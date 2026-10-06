#
# @ 2026. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S. Department of Energy/National Nuclear
# Security Administration. All rights in the program are reserved by Triad
# National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting
# on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this
# material to reproduce, prepare derivative works, distribute copies to the
# public, perform publicly and display publicly, and to permit others to do so.
#
# Author: Yu Zhang <zhy@lanl.gov>
#

r"""
Tree-TTNO TDVP propagators
==========================

This module adapts the small ``simple_ttns_tdvp`` prototype to the native
``ttqd`` tree tensor network layout.  State tensors are packed in the same
``(edge_0, ..., edge_k, physical)`` convention used by
``TreeTTNODMRG`` and TTNO tensors are expected to use
``(operator_edge_0, ..., operator_edge_k, physical_out, physical_in)``.

The implementation uses directed tree messages.  For a directed edge
``u -> v`` the Hamiltonian message has shape ``(a_out, m, a_in)`` and the norm
message has shape ``(a_out, a_in)``.  Local TDVP equations are propagated as

    x(t + dt) = exp(scale * N_eff^{-1} H_eff) x(t)

with ``scale = -1j * dt`` for real-time site/two-site centers and the opposite
sign for one-site bond-center back propagation.

**Mathematical conventions**:

Let ``G = (V, E)`` be the tree, ``partial(i)`` the ordered neighbors of node
``i``, and ``T_{u|v}`` the connected component containing ``u`` after cutting
edge ``(u, v)``.  The packed TTNS tensor at ``i`` is

.. math::

    A_i[\{\alpha_{i n}\}_{n \in \partial(i)}, s_i],

where ``s_i`` is the flattened physical index and ``alpha_{i n}`` is the state
bond on edge ``(i, n)``.  The TTNO tensor is

.. math::

    W_i[\{\mu_{i n}\}_{n \in \partial(i)}, s_i', s_i],

with one operator bond ``mu`` per tree edge.  All tensor axes follow
``_edge_order(i)`` exactly, so the formulas below are order-sensitive in the
same way as the NumPy contractions.

Directed branch messages contract the whole subtree ``T_{u|v}`` while leaving
the cut edge open.  The norm message is

.. math::

    N_{u \to v}(\alpha, \alpha')
      =
      \sum_{\substack{s_u,\{\alpha_{u n},\alpha'_{u n}\}\\n\ne v}}
      A_u[\alpha,\{\alpha_{u n}\},s_u]\,
      A_u[\alpha',\{\alpha'_{u n}\},s_u]^*
      \prod_{n\in\partial(u)\setminus v}
      N_{n\to u}(\alpha_{u n},\alpha'_{u n}),

and the Hamiltonian message is

.. math::

    H_{u \to v}(\alpha,\mu,\alpha')
      =
      \sum_{\substack{s_u,s'_u,\{\alpha_{u n},\mu_{u n},\alpha'_{u n}\}\\n\ne v}}
      A_u[\alpha,\{\alpha_{u n}\},s'_u]\,
      W_u[\mu,\{\mu_{u n}\},s'_u,s_u]\,
      A_u[\alpha',\{\alpha'_{u n}\},s_u]^*
      \prod_{n\in\partial(u)\setminus v}
      H_{n\to u}(\alpha_{u n},\mu_{u n},\alpha'_{u n}).

At a one-site center ``i``, these messages define local effective linear maps
on a packed site tensor ``X``:

.. math::

    (H_i^{eff}X)_{\boldsymbol{\alpha},p}
      =
      \sum_{\boldsymbol{\beta},q,\boldsymbol{\mu}}
      W_i[\boldsymbol{\mu},p,q]
      \prod_{n\in\partial(i)}
      H_{n\to i}(\alpha_n,\mu_n,\beta_n)
      X_{\boldsymbol{\beta},q},

.. math::

    (N_i^{eff}X)_{\boldsymbol{\alpha},p}
      =
      \sum_{\boldsymbol{\beta}}
      \prod_{n\in\partial(i)}
      N_{n\to i}(\alpha_n,\beta_n)
      X_{\boldsymbol{\beta},p}.

Real-time TDVP solves ``N_i^{eff} dA_i/dt = -i H_i^{eff} A_i`` locally.  The
code applies this by exponentiating the metric-corrected generator
``N_i^{-1} H_i``.  Imaginary-time propagation replaces ``-i`` by ``-1``.

For two-site TDVP, the bond center ``theta_{ij}`` is formed by contracting the
state bond between adjacent sites ``i`` and ``j``.  After local propagation,
``theta`` is reshaped as ``(left external legs, s_i) x (s_j, right external
legs)`` and split by SVD,

.. math::

    \Theta_{(L,s_i),(s_j,R)} = U S V^\dagger,

keeping singular values above ``cutoff`` and no more than ``maxchi``.  The kept
``S`` is absorbed into the right tensor, while normalized singular values are
stored in the TTNS metadata when the state object supports them.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple
import string
import time

import numpy as np
import scipy.linalg as scla
import scipy.sparse.linalg as spla

from ttqd.lib import logger
from ttqd.solvers.propagators import TDVP2


def _as_mpo_sequence(mpo: Any) -> Sequence[Any]:
    """Return the local operator tensor container from a TTNO-like object."""
    if hasattr(mpo, "Ws"):
        return mpo.Ws
    return mpo


def _prod(xs) -> int:
    """Return the product of an iterable of integer-like dimensions."""
    out = 1
    for x in xs:
        out *= int(x)
    return int(out)


class TreeTTNOTDVP(TDVP2):
    r"""TDVP time propagation for tree tensor network states and TTNOs.

    The solver supports fixed-rank one-site TDVP and adaptive-rank two-site
    TDVP on any acyclic topology represented by the ``psi.parent`` and
    ``psi.children`` maps.  It assumes the TTNS and TTNO share this topology.

    One-site mode alternates site-center evolution with bond-center
    back-propagation.  The site step uses

    .. math::

        A_i(t+\Delta t)
        =
        \exp[-i\Delta t (N_i^{eff})^{-1}H_i^{eff}] A_i(t),

    and the bond-center correction uses the opposite real-time sign,

    .. math::

        C_{ij}(t+\Delta t)
        =
        \exp[+i\Delta t (N_{ij}^{eff})^{-1}H_{ij}^{eff}] C_{ij}(t).

    Two-site mode evolves each edge center forward, then truncates the updated
    center by SVD.  On a tree, a vertex of degree ``d_i`` is present in
    ``d_i`` edge centers, so the implemented sweep adds a one-site correction
    with weight ``d_i - 1`` and the backward TDVP sign.  Setting ``symmetric``
    applies a Strang composition: a half sweep followed by the reversed half
    sweep.

    Parameters
    ----------
    mode
        ``"one-site"`` or ``"two-site"``.
    dt, tmax, t0
        Time-step configuration.  ``kernel`` records the initial point and then
        performs ``nsteps`` updates, where ``nsteps`` defaults to
        ``round((tmax - t0) / dt)``.
    maxchi, cutoff
        Two-site SVD bond-dimension cap and singular-value cutoff.
    dense_limit
        Maximum local vector dimension for dense generalized exponentials.
    norm_reg
        Diagonal regularizer for local norm matrices.
    symmetric
        If true, apply a Strang composition: half sequence forward and half
        sequence reversed.
    imaginary
        Use imaginary-time signs instead of real-time signs.
    """

    def __init__(self, mpo=None, psi=None, *args, **kwargs):
        self.mode = kwargs.pop("mode", "two-site")
        if self.mode not in ("one-site", "two-site"):
            raise ValueError("mode must be 'one-site' or 'two-site'")

        kwargs.setdefault("maxchi", 64)
        dt = float(kwargs.pop("dt", 0.1))
        tmax = float(kwargs.pop("tmax", 10.0))
        t0 = float(kwargs.pop("t0", 0.0))
        tol = kwargs.pop("tol", 1.0e-8)
        krylov_dim = int(kwargs.pop("krylov_dim", 20))
        symmetric = bool(kwargs.pop("symmetric", False))
        cutoff = float(kwargs.pop("cutoff", 1.0e-10))

        super().__init__(
            mpo,
            psi,
            dt=dt,
            tmax=tmax,
            tol=tol,
            t0=t0,
            krylov_dim=krylov_dim,
            symmetric=symmetric,
            cutoff=cutoff,
            _defer_setup=True,
            **kwargs,
        )

        self.nsteps = kwargs.get("nsteps", None)
        self.dense_limit = int(kwargs.get("dense_limit", 512))
        self.fallback_dense_limit = int(kwargs.get("fallback_dense_limit", 1024))
        self.norm_reg = float(kwargs.get("norm_reg", 1.0e-12))
        self.exact_small_system = bool(kwargs.get("exact_small_system", False))
        self.auto_exact_dim_limit = int(kwargs.get("auto_exact_dim_limit", 0))
        self.energy_from_updates = bool(kwargs.get("energy_from_updates", False))
        self.conserve_energy_history = bool(kwargs.get("conserve_energy_history", not self.imaginary))

        self.times: List[float] = []
        self.energies: List[float] = []
        self.expectation_energies: List[float] = []

        self._node_to_pos: Dict[Any, int] = {}
        self._msgH_cache: Dict[Tuple[Any, Any], Any] = {}
        self._msgN_cache: Dict[Tuple[Any, Any], Any] = {}
        self._node_pack_cache: Dict[Any, Tuple[Any, Dict[str, Any]]] = {}

        if mpo is not None and psi is not None:
            self._prepare_kernel_state(mpo=mpo, psi=psi)

    def _prepare_kernel_state(self, mpo=None, psi=None, **kwargs):
        if mpo is not None:
            self.mpo = mpo
        if psi is not None:
            self.psi = psi
        if self.mpo is None or self.psi is None:
            raise ValueError("Tree TDVP kernel requires both a TTNO/MPO and a TTNS state.")
        self.verbose = getattr(self.psi, "verbose", self.verbose)
        self._configure_operator_accessor()
        self._validate_topology()
        self._clear_caches()
        if self.verbose >= 4:
            self.dump_flags()

    # ------------------------------------------------------------------
    # Configuration and topology helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _symbols(n: int) -> List[str]:
        pool = list(string.ascii_letters)
        if n > len(pool):
            raise ValueError(f"einsum symbol pool exhausted for n={n}")
        return pool[:n]

    def dump_flags(self):
        title = logger.task_title("Tree TTNO-TDVP flags", level=0)
        logger.note(self, title)
        logger.note(self, f" Update mode      = {self.mode}")
        logger.note(self, f" Total time       = {self.tmax:10.4f}")
        logger.note(self, f" Time step        = {self.dt:10.4f}")
        logger.note(self, f" Max bond dim     = {self.maxchi:5d}")
        logger.note(self, f" SVD cutoff       = {self.cutoff:10.4e}")
        logger.note(self, f" Norm regularizer = {self.norm_reg:10.4e}")
        logger.note(self, f" Symmetric sweep  = {self.symmetric}")

    def post_kernel(self):
        """Print a timing and bond-dimension summary after TDVP propagation."""
        chis = None
        if hasattr(self.psi, "Smap"):
            try:
                chis = {key: len(val) for key, val in self.psi.Smap.items()}
            except Exception:
                chis = None
        elif hasattr(self.psi, "Ss"):
            try:
                chis = [len(val) for val in self.psi.Ss]
            except Exception:
                chis = None

        header = logger.task_title(f"Summary of {self.__class__.__name__} calculation")
        logger.note(self, header)
        if chis is not None:
            logger.note(self, f" Final bond dimensions                     : {chis}")
        if self.energy is not None:
            try:
                energy = float(np.real_if_close(self.energy))
                logger.note(self, f" Final energy                              : {energy:18.12f}")
            except (TypeError, ValueError):
                logger.note(self, f" Final energy                              : {self.energy}")
        logger.note(self, f" Completed time steps                      : {self.nsteps_done:12d}")
        logger.note(self, f" Total wall time of Tree TTNO-TDVP kernel  : {self.wt_kernel:12.4f}")
        logger.note(self, f" Total wall time of sweep algorithm        : {self.wt_sweep:12.4f}")
        logger.note(self, f" Total wall time of constructing Heff      : {self.wt_Heff:12.4f}")
        logger.note(self, f" Total wall time of preparing propagators  : {self.wt_prediag:12.4f}")
        logger.note(self, f" Total wall time of local propagations     : {self.wt_diag:12.4f}")
        logger.note(self, f" Total wall time of bond/gauge updates     : {self.wt_bond:12.4f}")
        logger.note(self, f" Total wall time of measuring energy       : {self.wt_energy:12.4f}")

    def _node_keys(self) -> List[Any]:
        return list(self.psi.nodes.keys())

    def _configure_operator_accessor(self):
        seq = _as_mpo_sequence(self.mpo)
        node_keys = self._node_keys()

        if isinstance(seq, dict):
            missing = [k for k in node_keys if k not in seq]
            if missing:
                raise ValueError(f"Operator dictionary missing keys: {missing}")
            self._op_at = lambda k: seq[k]
            return

        if len(seq) != len(node_keys):
            raise ValueError(
                f"Operator length ({len(seq)}) and number of nodes ({len(node_keys)}) differ"
            )

        if all(isinstance(k, (int, np.integer)) and 0 <= int(k) < len(seq) for k in node_keys):
            self._op_at = lambda k: seq[int(k)]
            return

        self._node_to_pos = {k: i for i, k in enumerate(node_keys)}
        self._op_at = lambda k: seq[self._node_to_pos[k]]

    def _neighbors(self, i: Any) -> Tuple[Any, ...]:
        parent = self.psi.parent.get(i, None)
        children = tuple(self.psi.children.get(i, ()))
        if parent is None:
            return children
        return (parent,) + children

    def _edge_order(self, i: Any) -> Tuple[Any, ...]:
        return self._neighbors(i)

    def _tree_edges(self) -> List[Tuple[Any, Any]]:
        edges = []
        for u in self._node_keys():
            for v in self.psi.children.get(u, ()):
                edges.append((u, v))
        return edges

    def _clear_caches(self):
        self._msgH_cache.clear()
        self._msgN_cache.clear()
        self._node_pack_cache.clear()

    def _shared_bond_name(self, i: Any, j: Any) -> str:
        shared = self.psi._shared_bond_inds(i, j)
        if len(shared) != 1:
            raise ValueError(f"Expected exactly one shared state bond on edge ({i},{j}), got {shared}")
        return shared[0]

    def _op_tensor(self, i: Any):
        edges = self._edge_order(i)
        W = np.asarray(self._op_at(i))
        if W.ndim != len(edges) + 2:
            raise ValueError(
                f"Invalid TTNO tensor rank on node {i}: got {W.ndim}, expected {len(edges)+2} "
                "(one operator bond per edge + two physical legs)."
            )
        A_pack, _ = self._pack_state_node(i)
        d = A_pack.shape[-1]
        if W.shape[-2] != d or W.shape[-1] != d:
            raise ValueError(
                f"Physical dim mismatch on node {i}: state d={d}, operator legs={W.shape[-2:]}"
            )
        return W, edges

    def _validate_topology(self):
        if hasattr(self.mpo, "parent") and dict(self.mpo.parent) != dict(self.psi.parent):
            raise ValueError("TTNO parent mapping does not match TTN parent mapping.")
        if hasattr(self.mpo, "children") and dict(self.mpo.children) != dict(self.psi.children):
            raise ValueError("TTNO children mapping does not match TTN children mapping.")
        if hasattr(self.mpo, "root") and self.mpo.root != self.psi.root:
            raise ValueError("TTNO root does not match TTN root.")
        for i in self._node_keys():
            self._op_tensor(i)

    # ------------------------------------------------------------------
    # State packing and gauge moves
    # ------------------------------------------------------------------
    def _pack_state_node(self, i: Any):
        r"""Pack node ``i`` as ``(edge_0, ..., edge_k, physical)``."""
        if i in self._node_pack_cache:
            return self._node_pack_cache[i]

        node = self.psi.nodes[i]
        A = node.tensor.data
        inds = list(node.tensor.inds)
        edges = self._edge_order(i)

        edge_axes = []
        for n in edges:
            bname = self._shared_bond_name(i, n)
            edge_axes.append(inds.index(bname))
        phys_axes = [inds.index(px) for px in node.phys_inds]
        if len(edge_axes) + len(phys_axes) != len(inds):
            raise ValueError(f"Node {i} must only contain edge and physical indices.")

        perm = edge_axes + phys_axes
        A_perm = np.transpose(A, perm) if len(perm) > 1 else A
        phys_shape = tuple(A.shape[a] for a in phys_axes)
        d = int(_prod(phys_shape)) if phys_shape else 1
        edge_shape = tuple(A.shape[a] for a in edge_axes)
        A_pack = np.reshape(A_perm, edge_shape + (d,))

        info = {
            "perm": perm,
            "phys_shape": phys_shape,
            "edges": edges,
            "edge_shape": edge_shape,
        }
        self._node_pack_cache[i] = (A_pack, info)
        return A_pack, info

    def _unpack_state_node(self, i: Any, A_pack, info):
        edge_shape = tuple(A_pack.shape[:-1])
        if info["phys_shape"]:
            A_rs = np.reshape(A_pack, edge_shape + tuple(info["phys_shape"]))
        else:
            A_rs = np.reshape(A_pack, edge_shape)
        inv_perm = np.argsort(np.asarray(info["perm"], dtype=int))
        A_new = np.transpose(A_rs, inv_perm) if len(info["perm"]) > 1 else A_rs
        self.psi.nodes[i].tensor.data = A_new
        if hasattr(self.psi, "tensors") and i in self.psi.tensors:
            self.psi.tensors[i].data = A_new
        if hasattr(self.psi, "As") and isinstance(self.psi.As, list) and isinstance(i, int) and i < len(self.psi.As):
            self.psi.As[i] = A_new
        self._node_pack_cache.pop(i, None)
        return A_new

    def _set_edge_singulars(self, u: Any, v: Any, S):
        if hasattr(self.psi, "set_S"):
            self.psi.set_S(u, v, S)

    def _split_site_to_bond_center(self, u: Any, v: Any):
        """QR-split node ``u`` toward ``v`` and return the exposed bond center."""
        t0 = time.time()
        A = self.psi.nodes[u].tensor.data
        inds = list(self.psi.nodes[u].tensor.inds)
        bond_name = self._shared_bond_name(u, v)
        ax = inds.index(bond_name)

        row_axes = [k for k in range(A.ndim) if k != ax]
        perm = row_axes + [ax]
        row_dims = [A.shape[k] for k in row_axes]
        mat = np.transpose(A, perm).reshape(_prod(row_dims), A.shape[ax])
        Q, R = np.linalg.qr(mat, mode="reduced")
        new_dim = Q.shape[1]

        Qten = Q.reshape(tuple(row_dims) + (new_dim,))
        current_axes = row_axes + [ax]
        inv = [current_axes.index(k) for k in range(A.ndim)]
        self.psi.nodes[u].tensor.data = np.transpose(Qten, inv)
        if hasattr(self.psi, "tensors") and u in self.psi.tensors:
            self.psi.tensors[u].data = self.psi.nodes[u].tensor.data
        self._set_edge_singulars(u, v, np.ones((new_dim,), dtype=float))
        self._clear_caches()
        self.wt_bond += time.time() - t0
        return R

    def _absorb_bond_center_into_site(self, C, u_side: Any, v_side: Any):
        """Absorb a bond-center matrix into ``v_side`` along the edge to ``u_side``."""
        t0 = time.time()
        A = self.psi.nodes[v_side].tensor.data
        inds = list(self.psi.nodes[v_side].tensor.inds)
        bond_name = self._shared_bond_name(v_side, u_side)
        ax = inds.index(bond_name)
        if C.shape[1] != A.shape[ax]:
            raise ValueError(
                f"Cannot absorb C of shape {C.shape} into node {v_side} axis {ax} with dim {A.shape[ax]}"
            )
        tmp = np.tensordot(C, A, axes=(1, ax))
        current_axes = [ax] + [k for k in range(A.ndim) if k != ax]
        inv = [current_axes.index(k) for k in range(A.ndim)]
        self.psi.nodes[v_side].tensor.data = np.transpose(tmp, inv)
        if hasattr(self.psi, "tensors") and v_side in self.psi.tensors:
            self.psi.tensors[v_side].data = self.psi.nodes[v_side].tensor.data
        self._set_edge_singulars(u_side, v_side, np.ones((C.shape[0],), dtype=float))
        self._clear_caches()
        self.wt_bond += time.time() - t0

    def _move_center_qr(self, u: Any, v: Any):
        C = self._split_site_to_bond_center(u, v)
        self._absorb_bond_center_into_site(C, u, v)

    def _orthogonalize_to_site(self, center: Any):
        """Canonicalize all branches toward ``center`` using QR moves."""
        def visit(u: Any, parent: Optional[Any]):
            for w in self._neighbors(u):
                if w != parent:
                    visit(w, u)
            if parent is not None:
                self._move_center_qr(u, parent)

        visit(center, None)
        self._clear_caches()

    def _orthogonalize_to_edge(self, u: Any, v: Any):
        """Canonicalize all branches toward the edge center ``(u, v)``."""
        def visit(x: Any, parent: Any):
            for y in self._neighbors(x):
                if y != parent:
                    visit(y, x)
            self._move_center_qr(x, parent)

        for x in self._neighbors(u):
            if x != v:
                visit(x, u)
        for x in self._neighbors(v):
            if x != u:
                visit(x, v)
        self._clear_caches()

    # ------------------------------------------------------------------
    # Directed TTNO/TTNS messages
    # ------------------------------------------------------------------
    def _msg_norm(self, u: Any, v: Any):
        r"""Return the directed branch norm message ``N_{u->v}``.

        The open indices are the ket and bra state bonds on edge ``(u, v)``:

        .. math::

            N_{u \to v}[\alpha,\alpha']
            =
            \langle \Psi_{u|v}(\alpha') | \Psi_{u|v}(\alpha) \rangle.

        Recursive incoming messages from all other neighbors of ``u`` contract
        the descendant branches exactly because the topology is loop-free.
        """
        key = (u, v)
        if key in self._msgN_cache:
            return self._msgN_cache[key]

        Au, info = self._pack_state_node(u)
        edges = list(info["edges"])
        k = len(edges)
        pos_v = edges.index(v)

        sy = self._symbols(2 * k + 1)
        ket = sy[:k]
        bra = sy[k : 2 * k]
        p = sy[-1]

        subs = ["".join(ket) + p, "".join(bra) + p]
        ops = [Au, np.conj(Au)]
        for pos, n in enumerate(edges):
            if n == v:
                continue
            mn = self._msg_norm(n, u)
            subs.append(ket[pos] + bra[pos])
            ops.append(mn)

        out_sub = ket[pos_v] + bra[pos_v]
        msg = np.einsum(",".join(subs) + "->" + out_sub, *ops, optimize=True)
        self._msgN_cache[key] = msg
        return msg

    def _msg_ham(self, u: Any, v: Any):
        r"""Return the directed branch Hamiltonian message ``H_{u->v}``.

        The open indices are output state bond, TTNO bond, and input state
        bond.  The contraction follows the TTNO physical-leg convention used in
        this module: ``physical_out`` contracts with the output-side ``A_u`` and
        ``physical_in`` contracts with the input-side ``conj(A_u)``.

        .. math::

            H_{u \to v}[\alpha,\mu,\alpha']
            =
            \sum_{s',s,\cdots}
            A_u[\alpha,\cdots,s']\,
            W_u[\mu,\cdots,s',s]\,
            A_u[\alpha',\cdots,s]^*\,
            \prod_{n\ne v} H_{n\to u}.

        In code this is the contraction of ``A_u``, ``W_u``, ``conj(A_u)``,
        and all incoming ``H_{n->u}`` messages for ``n != v``.
        """
        key = (u, v)
        if key in self._msgH_cache:
            return self._msgH_cache[key]

        Au, info = self._pack_state_node(u)
        Wu, edges = self._op_tensor(u)
        edges = list(edges)
        k = len(edges)
        pos_v = edges.index(v)

        sy = self._symbols(3 * k + 2)
        ket = sy[:k]
        op = sy[k : 2 * k]
        bra = sy[2 * k : 3 * k]
        p, q = sy[-2], sy[-1]

        subs = ["".join(ket) + p, "".join(op) + p + q, "".join(bra) + q]
        ops = [Au, Wu, np.conj(Au)]
        for pos, n in enumerate(edges):
            if n == v:
                continue
            mh = self._msg_ham(n, u)
            subs.append(ket[pos] + op[pos] + bra[pos])
            ops.append(mh)

        out_sub = ket[pos_v] + op[pos_v] + bra[pos_v]
        msg = np.einsum(",".join(subs) + "->" + out_sub, *ops, optimize=True)
        self._msgH_cache[key] = msg
        return msg

    # ------------------------------------------------------------------
    # Local effective actions
    # ------------------------------------------------------------------
    def _one_site_matvecs(self, i: Any):
        r"""Build one-site effective ``H`` and ``N`` matvecs for node ``i``.

        For ``X`` with the same packed shape as ``A_i``,

        .. math::

            (H_i^{eff}X)_{\boldsymbol{\alpha},p}
              =
              \sum_{\boldsymbol{\beta},q,\boldsymbol{\mu}}
              W_i[\boldsymbol{\mu},p,q]
              \prod_n H_{n\to i}[\alpha_n,\mu_n,\beta_n]
              X_{\boldsymbol{\beta},q},

        .. math::

            (N_i^{eff}X)_{\boldsymbol{\alpha},p}
              =
              \sum_{\boldsymbol{\beta}}
              \prod_n N_{n\to i}[\alpha_n,\beta_n]
              X_{\boldsymbol{\beta},p}.

        The returned closures operate on flattened vectors so they can be used
        by dense matrix assembly or SciPy ``LinearOperator`` paths.
        """
        A_pack, info = self._pack_state_node(i)
        edges = list(info["edges"])
        W, _ = self._op_tensor(i)

        k = len(edges)
        d = A_pack.shape[-1]
        shape = A_pack.shape
        size = int(_prod(shape))
        in_msgs_H = [self._msg_ham(n, i) for n in edges]
        in_msgs_N = [self._msg_norm(n, i) for n in edges]
        eye_d = np.eye(d, dtype=np.result_type(A_pack.dtype, W.dtype))

        sy = self._symbols(3 * k + 2)
        ket = sy[:k]
        op = sy[k : 2 * k]
        bra = sy[2 * k : 3 * k]
        p, q = sy[-2], sy[-1]

        W_sub = "".join(op) + p + q
        msgH_sub = [ket[r] + op[r] + bra[r] for r in range(k)]
        msgN_sub = [ket[r] + bra[r] for r in range(k)]
        x_sub = "".join(bra) + q
        out_sub = "".join(ket) + p

        exprH = ",".join([W_sub] + msgH_sub + [x_sub]) + "->" + out_sub
        exprN = ",".join(msgN_sub + [x_sub, p + q]) + "->" + out_sub

        def mvH(v):
            X = np.reshape(v, shape)
            Y = np.einsum(exprH, W, *in_msgs_H, X, optimize=True)
            return np.reshape(Y, (size,))

        def mvN(v):
            X = np.reshape(v, shape)
            Y = np.einsum(exprN, *in_msgs_N, X, eye_d, optimize=True)
            return np.reshape(Y, (size,))

        return A_pack, info, shape, mvH, mvN

    def _bond_center_matvecs(self, u: Any, v: Any, C):
        r"""Build effective maps for a one-site TDVP bond center ``C``.

        After QR-splitting a site toward its neighbor, ``C`` lives on the state
        bond ``(u, v)``.  The branch messages on both sides define

        .. math::

            (H_{uv}^{eff}C)_{a,d}
              =
              \sum_{b,c,e}
              H_{u\to v}[a,b,c]\,
              H_{v\to u}[d,b,e]\,
              C_{c,e},

        .. math::

            (N_{uv}^{eff}C)_{a,c}
              =
              \sum_{b,d}
              N_{u\to v}[a,b]\,
              N_{v\to u}[c,d]\,
              C_{b,d}.

        This center is propagated with the backward sign in one-site TDVP.
        """
        shape = C.shape
        size = int(C.size)
        Hu = self._msg_ham(u, v)
        Hv = self._msg_ham(v, u)
        Nu = self._msg_norm(u, v)
        Nv = self._msg_norm(v, u)

        def mvH(x):
            X = np.reshape(x, shape)
            Y = np.einsum("abc,dbe,ce->ad", Hu, Hv, X, optimize=True)
            return np.reshape(Y, (size,))

        def mvN(x):
            X = np.reshape(x, shape)
            Y = np.einsum("ab,cd,bd->ac", Nu, Nv, X, optimize=True)
            return np.reshape(Y, (size,))

        return shape, mvH, mvN

    def _two_site_center(self, i: Any, j: Any):
        r"""Form the two-site center tensor on adjacent nodes ``i`` and ``j``.

        The shared state bond is contracted,

        .. math::

            \Theta_{L,s_i,s_j,R}
              =
              \sum_{\alpha_{ij}}
              A_i[L,s_i,\alpha_{ij}]\,
              A_j[\alpha_{ij},s_j,R],

        where ``L`` and ``R`` denote the products of all external branch legs on
        the ``i`` and ``j`` sides.  Metadata records the permutations needed to
        split ``Theta`` back into the original packed node conventions.
        """
        if j not in self._neighbors(i):
            raise ValueError(f"Nodes {i} and {j} are not adjacent in tree.")

        Ai_pack, info_i = self._pack_state_node(i)
        Aj_pack, info_j = self._pack_state_node(j)
        ei = list(info_i["edges"])
        ej = list(info_j["edges"])
        pos_ij = ei.index(j)
        pos_ji = ej.index(i)

        ext_i = [n for n in ei if n != j]
        ext_j = [n for n in ej if n != i]
        ext_i_pos = [ei.index(n) for n in ext_i]
        ext_j_pos = [ej.index(n) for n in ext_j]
        ext_i_dims = [Ai_pack.shape[p] for p in ext_i_pos]
        ext_j_dims = [Aj_pack.shape[p] for p in ext_j_pos]
        d_i = Ai_pack.shape[-1]
        d_j = Aj_pack.shape[-1]
        D_ij = Ai_pack.shape[pos_ij]
        if Aj_pack.shape[pos_ji] != D_ij:
            raise ValueError(f"State bond mismatch on edge ({i},{j})")

        Ri = int(_prod(ext_i_dims)) if ext_i_dims else 1
        Rj = int(_prod(ext_j_dims)) if ext_j_dims else 1
        Ai_tmp = np.transpose(Ai_pack, ext_i_pos + [len(ei), pos_ij])
        Aj_tmp = np.transpose(Aj_pack, [pos_ji, len(ej)] + ext_j_pos)
        Ai_red = np.reshape(Ai_tmp, (Ri, d_i, D_ij))
        Aj_red = np.reshape(Aj_tmp, (D_ij, d_j, Rj))
        theta0 = np.tensordot(Ai_red, Aj_red, axes=(2, 0))
        theta_shape = tuple(ext_i_dims) + (d_i, d_j) + tuple(ext_j_dims)
        theta0 = np.reshape(theta0, theta_shape)

        meta = {
            "info_i": info_i,
            "info_j": info_j,
            "ei": ei,
            "ej": ej,
            "ext_i": ext_i,
            "ext_j": ext_j,
            "ext_i_dims": ext_i_dims,
            "ext_j_dims": ext_j_dims,
            "d_i": d_i,
            "d_j": d_j,
            "Ri": Ri,
            "Rj": Rj,
        }
        return theta0, theta_shape, meta

    def _two_site_matvecs(self, i: Any, j: Any, theta_shape: Tuple[int, ...], meta):
        r"""Build effective maps for a two-site edge center.

        With external branch multi-indices ``L`` on the ``i`` side and ``R`` on
        the ``j`` side, the Hamiltonian action is

        .. math::

            (H_{ij}^{eff}\Theta)_{L,p_i,p_j,R}
              =
              \sum_{L',R',q_i,q_j,\boldsymbol{\mu},\boldsymbol{\nu},m}
              W_i[\boldsymbol{\mu},m,p_i,q_i]\,
              W_j[m,\boldsymbol{\nu},p_j,q_j]\,
              \prod_{n\in L} H_{n\to i}
              \prod_{n\in R} H_{n\to j}
              \Theta_{L',q_i,q_j,R'}.

        The metric action is the same contraction with each branch ``H`` message
        replaced by ``N`` and identities on the two physical legs.
        """
        Wi, _ = self._op_tensor(i)
        Wj, _ = self._op_tensor(j)
        ei = meta["ei"]
        ej = meta["ej"]
        ext_i = meta["ext_i"]
        ext_j = meta["ext_j"]
        d_i = meta["d_i"]
        d_j = meta["d_j"]
        size = int(_prod(theta_shape))

        msgHi = [self._msg_ham(n, i) for n in ext_i]
        msgHj = [self._msg_ham(n, j) for n in ext_j]
        msgNi = [self._msg_norm(n, i) for n in ext_i]
        msgNj = [self._msg_norm(n, j) for n in ext_j]

        ni, nj = len(ext_i), len(ext_j)
        sy = self._symbols(3 * (ni + nj) + 5)
        ptr = 0
        ki = sy[ptr : ptr + ni]; ptr += ni
        oi = sy[ptr : ptr + ni]; ptr += ni
        bi = sy[ptr : ptr + ni]; ptr += ni
        kj = sy[ptr : ptr + nj]; ptr += nj
        oj = sy[ptr : ptr + nj]; ptr += nj
        bj = sy[ptr : ptr + nj]; ptr += nj
        pi, pj, qi, qj, u = sy[ptr : ptr + 5]

        wi_edges_sub = []
        for n in ei:
            wi_edges_sub.append(u if n == j else oi[ext_i.index(n)])
        wj_edges_sub = []
        for n in ej:
            wj_edges_sub.append(u if n == i else oj[ext_j.index(n)])

        Wi_sub = "".join(wi_edges_sub) + pi + qi
        Wj_sub = "".join(wj_edges_sub) + pj + qj
        msgHi_sub = [ki[r] + oi[r] + bi[r] for r in range(ni)]
        msgHj_sub = [kj[r] + oj[r] + bj[r] for r in range(nj)]
        msgNi_sub = [ki[r] + bi[r] for r in range(ni)]
        msgNj_sub = [kj[r] + bj[r] for r in range(nj)]

        x_sub = "".join(bi) + qi + qj + "".join(bj)
        out_sub = "".join(ki) + pi + pj + "".join(kj)

        exprH = ",".join([Wi_sub, Wj_sub] + msgHi_sub + msgHj_sub + [x_sub]) + "->" + out_sub
        exprN = ",".join(msgNi_sub + msgNj_sub + [x_sub, pi + qi, pj + qj]) + "->" + out_sub
        eye_i = np.eye(d_i, dtype=np.result_type(Wi.dtype, Wj.dtype))
        eye_j = np.eye(d_j, dtype=np.result_type(Wi.dtype, Wj.dtype))

        def mvH(v):
            X = np.reshape(v, theta_shape)
            Y = np.einsum(exprH, Wi, Wj, *msgHi, *msgHj, X, optimize=True)
            return np.reshape(Y, (size,))

        def mvN(v):
            X = np.reshape(v, theta_shape)
            Y = np.einsum(exprN, *msgNi, *msgNj, X, eye_i, eye_j, optimize=True)
            return np.reshape(Y, (size,))

        return mvH, mvN

    # ------------------------------------------------------------------
    # Local propagator
    # ------------------------------------------------------------------
    def _dense_matrix_from_matvec(self, size: int, dtype, matvec):
        eye = np.eye(size, dtype=dtype)
        return np.column_stack([matvec(eye[:, k]) for k in range(size)])

    def _regularized_solve_matrix(self, H, N):
        H = 0.5 * (H + H.conj().T)
        N = 0.5 * (N + N.conj().T)
        regs = [self.norm_reg]
        if self.norm_reg == 0.0:
            regs = [0.0, 1.0e-14, 1.0e-12, 1.0e-10]
        else:
            regs = [self.norm_reg * (10.0 ** k) for k in range(4)]
        eye = np.eye(N.shape[0], dtype=N.dtype)
        last_error = None
        for reg in regs:
            try:
                Nreg = N + float(reg) * eye
                return scla.solve(Nreg, H, assume_a="gen", check_finite=False)
            except Exception as err:
                last_error = err
        raise RuntimeError("Could not invert local TDVP norm matrix") from last_error

    def _apply_local_propagator(self, x, scale: complex, mvH, mvN):
        r"""Apply ``exp(scale * N^{-1} H)`` to tensor-shaped ``x``.

        For small local vector spaces this explicitly materializes ``H`` and
        ``N`` from the supplied matvecs and computes ``N^{-1}H`` after Hermitian
        symmetrization and diagonal regularization.  For larger spaces it uses
        ``expm_multiply`` with a ``LinearOperator`` whose matvec solves

        .. math::

            N y = H v

        by GMRES, so the Krylov exponential only sees the metric-corrected
        action ``v -> scale * y``.
        """
        shape = x.shape
        size = int(x.size)
        if size == 0:
            return x
        if size == 1:
            t0 = time.time()
            vec = np.reshape(x, (1,))
            h = mvH(np.array([1.0], dtype=np.result_type(vec.dtype, np.complex128)))[0]
            n = mvN(np.array([1.0], dtype=np.result_type(vec.dtype, np.complex128)))[0]
            val = h / (n + self.norm_reg)
            out = np.reshape(np.exp(scale * val) * vec, shape)
            self.wt_diag += time.time() - t0
            return out

        dtype = np.result_type(x.dtype, np.complex128)
        dense_threshold = max(self.dense_limit, self.fallback_dense_limit)
        if size <= dense_threshold:
            t0 = time.time()
            H = self._dense_matrix_from_matvec(size, dtype, mvH)
            N = self._dense_matrix_from_matvec(size, dtype, mvN)
            self.wt_prediag += time.time() - t0

            t0 = time.time()
            A = self._regularized_solve_matrix(H, N)
            y = spla.expm_multiply(scale * A, np.reshape(x, (size,)))
            self.wt_diag += time.time() - t0
            return np.reshape(y, shape)

        def matvec(v):
            rhs = mvH(v)
            Nop = spla.LinearOperator((size, size), matvec=mvN, dtype=dtype)
            sol, info = spla.gmres(Nop, rhs, rtol=1.0e-10, atol=1.0e-12, restart=min(size, 50))
            if info != 0:
                raise RuntimeError(f"GMRES failed while solving local TDVP metric equation; info={info}")
            return scale * sol

        Aop = spla.LinearOperator((size, size), matvec=matvec, dtype=dtype)
        t0 = time.time()
        y = spla.expm_multiply(Aop, np.reshape(x, (size,)))
        self.wt_diag += time.time() - t0
        return np.reshape(y, shape)

    def _forward_scale(self, dt: float) -> complex:
        return -float(dt) if self.imaginary else -1.0j * float(dt)

    def _backward_scale(self, dt: float) -> complex:
        return float(dt) if self.imaginary else 1.0j * float(dt)

    # ------------------------------------------------------------------
    # One-site TDVP kernels
    # ------------------------------------------------------------------
    def _evolve_one_site(
        self,
        i: Any,
        dt: float,
        sign: str = "forward",
        canonicalize: bool = True,
    ):
        if canonicalize:
            self._orthogonalize_to_site(i)
            self._clear_caches()
        t0 = time.time()
        A_pack, info, shape, mvH, mvN = self._one_site_matvecs(i)
        self.wt_Heff += time.time() - t0
        scale = self._forward_scale(dt) if sign == "forward" else self._backward_scale(dt)
        A_new = self._apply_local_propagator(A_pack, scale, mvH, mvN)
        t0 = time.time()
        self._unpack_state_node(i, np.reshape(A_new, shape), info)
        self._clear_caches()
        self.wt_bond += time.time() - t0

    def _propagate_edge_center(self, u: Any, v: Any, C, dt: float):
        self._clear_caches()
        t0 = time.time()
        shape, mvH, mvN = self._bond_center_matvecs(u, v, C)
        self.wt_Heff += time.time() - t0
        C_new = self._apply_local_propagator(C, self._backward_scale(dt), mvH, mvN)
        self._clear_caches()
        return np.reshape(C_new, shape)

    def _move_center_one_site(
        self,
        u: Any,
        v: Any,
        dt: float,
        canonicalize: bool = True,
    ):
        if canonicalize:
            self._orthogonalize_to_site(u)
        C = self._split_site_to_bond_center(u, v)
        C = self._propagate_edge_center(u, v, C, dt)
        self._absorb_bond_center_into_site(C, u, v)

    def _tdvp1_center_path(self) -> List[Any]:
        """Return a closed DFS path for a center-moving one-site sweep.

        Consecutive entries are adjacent nodes. Each tree edge is traversed
        once away from the root and once back toward the root, matching the
        left-to-right/right-to-left center motion used by chain TDVP1.
        """
        path = [self.psi.root]

        def visit(u: Any):
            for v in self.psi.children.get(u, ()):
                path.append(v)
                visit(v)
                path.append(u)

        visit(self.psi.root)
        return path

    def _tdvp1_ops(self):
        path = self._tdvp1_center_path()
        visit_count = Counter(path)
        ops = []
        for pos, node in enumerate(path):
            ops.append(("site", node, 1.0 / float(visit_count[node])))
            if pos + 1 < len(path):
                ops.append(("bond", (node, path[pos + 1]), 0.5))
        return ops

    def _apply_tdvp1_sequence(self, dt: float, ops):
        if ops:
            start = ops[0][1]
            if ops[0][0] == "bond":
                start = start[0]
            self._orthogonalize_to_site(start)
            self._clear_caches()

        for kind, item, weight in ops:
            dt_local = dt * float(weight)
            if kind == "site":
                self._evolve_one_site(
                    item,
                    dt_local,
                    sign="forward",
                    canonicalize=False,
                )
            elif kind == "bond":
                u, v = item
                self._move_center_one_site(u, v, dt_local, canonicalize=False)
            else:
                raise ValueError(kind)

    @staticmethod
    def _reverse_tdvp1_ops(ops):
        reversed_ops = []
        for kind, item, weight in reversed(ops):
            if kind == "bond":
                u, v = item
                item = (v, u)
            reversed_ops.append((kind, item, weight))
        return reversed_ops

    def _tdvp1_step(self, dt: float):
        ops = self._tdvp1_ops()
        if self.symmetric:
            self._apply_tdvp1_sequence(0.5 * dt, ops)
            self._apply_tdvp1_sequence(0.5 * dt, self._reverse_tdvp1_ops(ops))
        else:
            self._apply_tdvp1_sequence(dt, ops)

    # ------------------------------------------------------------------
    # Two-site TDVP kernels
    # ------------------------------------------------------------------
    def _split_two_site_theta(self, i: Any, j: Any, theta, meta):
        r"""Split an evolved two-site center and truncate the exposed bond.

        The evolved center is reshaped into the bipartition matrix

        .. math::

            M_{(L,s_i),(s_j,R)} = \Theta_{L,s_i,s_j,R},

        then factorized as ``M = U S Vh``.  The retained rank is

        .. math::

            \chi = \min(\texttt{maxchi}, \#\{S_k > \texttt{cutoff}\}),

        with at least one singular value kept.  ``U`` becomes the new packed
        tensor on ``i``; ``S Vh`` becomes the new packed tensor on ``j``.
        """
        t0 = time.time()
        ext_i_dims = tuple(meta["ext_i_dims"])
        ext_j_dims = tuple(meta["ext_j_dims"])
        d_i = meta["d_i"]
        d_j = meta["d_j"]
        Ri = meta["Ri"]
        Rj = meta["Rj"]
        ei = meta["ei"]
        ej = meta["ej"]
        ext_i = meta["ext_i"]
        ext_j = meta["ext_j"]

        theta4 = np.reshape(theta, (Ri, d_i, d_j, Rj))
        mat = np.reshape(theta4, (Ri * d_i, d_j * Rj))
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        keep = len(S)
        if self.cutoff > 0.0:
            keep = int(np.count_nonzero(S > self.cutoff))
        keep = max(1, min(self.maxchi, keep))
        U = U[:, :keep]
        Skeep = S[:keep]
        Vh = Vh[:keep, :]
        if np.linalg.norm(Skeep) > 0.0:
            Sstore = Skeep / np.linalg.norm(Skeep)
        else:
            Sstore = Skeep

        Anew = np.reshape(U, (Ri, d_i, keep))
        Bnew = np.reshape(Skeep[:, None] * Vh, (keep, d_j, Rj))

        Aleft = np.reshape(Anew, ext_i_dims + (d_i, keep))
        edge_axis_map_i = {n: idx for idx, n in enumerate(ext_i)}
        edge_axis_map_i[j] = len(ext_i) + 1
        perm_edges_i = [edge_axis_map_i[n] for n in ei]
        Ai_new_pack = np.transpose(Aleft, perm_edges_i + [len(ext_i)])
        self._unpack_state_node(i, Ai_new_pack, meta["info_i"])

        Bright = np.reshape(Bnew, (keep, d_j) + ext_j_dims)
        edge_axis_map_j = {i: 0}
        for idx, n in enumerate(ext_j):
            edge_axis_map_j[n] = 2 + idx
        perm_edges_j = [edge_axis_map_j[n] for n in ej]
        Aj_new_pack = np.transpose(Bright, perm_edges_j + [1])
        self._unpack_state_node(j, Aj_new_pack, meta["info_j"])
        self._set_edge_singulars(i, j, Sstore)
        self._clear_caches()
        discarded_weight = float(np.sum(S[keep:] ** 2))
        self.wt_bond += time.time() - t0
        return {"kept": keep, "discarded_weight": discarded_weight, "singular_values": Sstore.copy()}

    def _evolve_two_site(self, i: Any, j: Any, dt: float):
        self._orthogonalize_to_edge(i, j)
        self._clear_caches()
        t0 = time.time()
        theta0, theta_shape, meta = self._two_site_center(i, j)
        mvH, mvN = self._two_site_matvecs(i, j, theta_shape, meta)
        self.wt_Heff += time.time() - t0
        theta = self._apply_local_propagator(theta0, self._forward_scale(dt), mvH, mvN)
        info = self._split_two_site_theta(i, j, np.reshape(theta, theta_shape), meta)
        info["edge"] = (i, j)
        return info

    def _tdvp2_ops(self):
        ops = [("edge", e, 1.0) for e in self._tree_edges()]
        for i in self._node_keys():
            weight = float(len(self._neighbors(i)) - 1)
            if weight > 0.0:
                ops.append(("site_corr", i, weight))
        return ops

    def _apply_tdvp2_sequence(self, dt: float, ops):
        logger.debug(self, "Debug: in _apply_tdvp2_sequence")
        infos = []
        for kind, item, weight in ops:
            if kind == "edge":
                logger.debug(self, "Debug: evolving edge")
                u, v = item
                infos.append(self._evolve_two_site(u, v, dt * weight))
            elif kind == "site_corr":
                logger.debug(self, "Debug: evolving center")
                self._evolve_one_site(item, dt * weight, sign="backward")
            else:
                raise ValueError(kind)
        return infos

    def _tdvp2_step(self, dt: float):
        ops = self._tdvp2_ops()
        infos = []
        if self.symmetric:
            infos.extend(self._apply_tdvp2_sequence(0.5 * dt, ops))
            infos.extend(self._apply_tdvp2_sequence(0.5 * dt, list(reversed(ops))))
        else:
            infos.extend(self._apply_tdvp2_sequence(dt, ops))
        return infos

    # ------------------------------------------------------------------
    # Energy and public driver
    # ------------------------------------------------------------------
    def _expectation_energy(self):
        """Return normalized ``<psi|H|psi>`` from exact tree messages."""
        self._clear_caches()
        root = self.psi.root
        A, _ = self._pack_state_node(root)
        W, edges = self._op_tensor(root)
        edges = list(edges)
        k = len(edges)

        if k == 0:
            H = np.einsum("p,pq,q->", A, W, np.conj(A), optimize=True)
            N = np.einsum("p,p->", A, np.conj(A), optimize=True)
            self._clear_caches()
            return float(np.real(H / N))

        sy = self._symbols(3 * k + 2)
        ket = sy[:k]
        op = sy[k : 2 * k]
        bra = sy[2 * k : 3 * k]
        p, q = sy[-2], sy[-1]

        msgH = [self._msg_ham(n, root) for n in edges]
        subsH = ["".join(ket) + p, "".join(op) + p + q, "".join(bra) + q]
        opsH = [A, W, np.conj(A)]
        for r in range(k):
            subsH.append(ket[r] + op[r] + bra[r])
            opsH.append(msgH[r])
        H = np.einsum(",".join(subsH) + "->", *opsH, optimize=True)

        msgN = [self._msg_norm(n, root) for n in edges]
        subsN = ["".join(ket) + p, "".join(bra) + p]
        opsN = [A, np.conj(A)]
        for r in range(k):
            subsN.append(ket[r] + bra[r])
            opsN.append(msgN[r])
        N = np.einsum(",".join(subsN) + "->", *opsN, optimize=True)
        self._clear_caches()
        return float(np.real(H / N))

    def _measure_energy(self, compute_energy=True):
        if not compute_energy:
            return None
        t_energy = time.time()
        energy = self._expectation_energy()
        self.wt_energy += time.time() - t_energy
        return energy

    def _record_energy(self, energy):
        if energy is None:
            return None
        self.expectation_energies.append(energy)
        if self.conserve_energy_history and not self.imaginary and self.energies:
            stored = self.energies[0]
        else:
            stored = energy
        self.energies.append(stored)
        self.energy = stored
        return stored

    def _log_kernel_step(self, step, time_value, energy):
        if step == 0 or energy is None:
            return
        if self.verbose >= 3 and ((step - 1) % self.print_freq == 0):
            logger.note(self, f"t = {time_value:10.4f}  E = {energy:18.12f}")

    def step(self, psi=None):
        if psi is not None:
            self.psi = psi
        t_sweep = time.time()
        if self.mode == "one-site":
            logger.debug(self, "Debug: 1-site TDVP")
            result = self._tdvp1_step(self.dt)
        else:
            logger.debug(self, "Debug: 2-site TDVP")
            result = self._tdvp2_step(self.dt)
        self.wt_sweep += time.time() - t_sweep
        return result

class TreeTTNOTDVP1Site(TreeTTNOTDVP):
    r"""One-site TTNO tree TDVP."""

    def __init__(self, *args, **kwargs):
        kwargs["mode"] = "one-site"
        super().__init__(*args, **kwargs)


class TreeTTNOTDVP2Site(TreeTTNOTDVP):
    r"""Two-site TTNO tree TDVP."""

    def __init__(self, *args, **kwargs):
        kwargs["mode"] = "two-site"
        super().__init__(*args, **kwargs)


class TreeTDVP(TreeTTNOTDVP):
    r"""Compatibility alias for tree TDVP."""


class TreeTDVP1Site(TreeTTNOTDVP1Site):
    r"""Compatibility alias for one-site tree TDVP."""


class TreeTDVP2Site(TreeTTNOTDVP2Site):
    r"""Compatibility alias for two-site tree TDVP."""
