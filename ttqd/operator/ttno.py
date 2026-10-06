

r"""
Tree tensor network operator (TTNO) module
------------------------------------------

This module parallels :class:`~ttqd.operator.mpo.MPO` but for arbitrary
acyclic topologies.  A ``TTNO`` stores a list of local operator tensors
``Ws`` and the accompanying parent/children information describing the tree
structure.  It is primarily used in conjunction with
:class:`~ttqd.network.TreeTensorNetwork` (or any
:class:`~ttqd.network.BaseTreeTensorNetwork` subclass) to ensure that the
operator and state networks share the same geometry.

The ``TTNO`` does **not** impose any particular ordering of the tensor
indices – the user is responsible for providing ``Ws`` whose index structure
matches the topology of the associated tree.  For simple examples one can
recycle MPO-style 4‑index tensors when the tree reduces to a chain.
"""
#
# Note: we may not need this class if we transform the local update in MPS format
# by combing other legs into one, see Ref. JPC 138, 134113 (2013)
#
from __future__ import annotations
from typing import Any, Sequence, Union, Dict, Tuple
from ttqd.linalg import backend


class TTNO:
    """Tree Tensor Network Operator.

    Parameters
    ----------
    Ws : Sequence[Any]
        Sequence of operator tensors; the length must equal the number of
        nodes in the tree.  Elements are converted to the configured backend
    parent : Dict[Any, Any]
        Mapping from node key to parent key (``None`` for the root).
    children : Dict[Any, Tuple[Any, ...]]
        Mapping from node key to an ordered tuple of its children.
    root : Any
        Key of the root node.
    """

    def __init__(
        self,
        Ws: Sequence[Any],
        parent: Dict[Any, Any],
        children: Dict[Any, Tuple[Any, ...]],
        root: Any,
    ):
        self.be = backend
        # make a copy of the topology dictionaries so callers can mutate them
        self.parent: Dict[Any, Any] = dict(parent)
        self.children: Dict[Any, Tuple[Any, ...]] = {k: tuple(v) for k, v in children.items()}
        self.root = root

        self.Ws = [self.be.asarray(W) for W in Ws]
        self.L = len(self.Ws)
        self.nsites = self.L # make alias

        if self.L != len(self.parent) or self.L != len(self.children):
            raise ValueError("Length of Ws must match number of tree nodes")

    @classmethod
    def from_ttns(
        cls,
        ttn: "ttqd.network.TreeTensorNetwork",
        Ws: Sequence[Any],
    ) -> "TTNO":
        r"""Create a TTNO matching the topology of a previously built TTNS.

        Parameters
        ----------
        ttn : TreeTensorNetwork
            Source state network whose topology will be reused for the
            operator.  ``ttn.parent``, ``ttn.children`` and ``ttn.root`` are
            copied into the new object.
        Ws : Sequence[Any]
            Operator tensors in the same order as ``ttn.nodes.keys()``.  A
            simple way to obtain this ordering is ``list(ttn.nodes)`` which
            iterates over the node keys.
        """
        # ensure correct ordering of Ws matches node keys
        node_keys = list(ttn.nodes.keys())
        if len(Ws) != len(node_keys):
            raise ValueError("Ws length must equal number of nodes in TTN")
        # order Ws according to node key listing
        ordered = [Ws[i] for i in range(len(node_keys))]
        return cls(ordered, ttn.parent, ttn.children, ttn.root)

    def to_networkx(self):
        """Return a networkx graph of the operator topology (same as the
        underlying tree).
        """
        # the network structure is identical to the state graph; reuse its
        # construction logic.
        try:
            from ttqd.network import BaseTreeTensorNetwork
            dummy = BaseTreeTensorNetwork(
                nodes={}, parent=self.parent, children=self.children, root=self.root
            )
            return dummy._to_networkx_graph()
        except ImportError:  # pragma: no cover
            raise ImportError("networkx required to export graph")

    @property
    def phys_dim(self):
        """Return the physical output dimension of the first operator tensor
        (assumes uniform physical legs)."""
        W0 = self.Ws[0]
        # we look for the last two axes (out, in) by convention
        if W0.ndim < 2:
            return 1
        return int(W0.shape[-2])

    #
    def to_dense(self) -> Array:
        """Debug helper: contract TTNO into a dense matrix."""

        out_phys = [lab.new() for _ in range(self.nsites)]
        in_phys = [lab.new() for _ in range(self.nsites)]
        # contract from the parent?

# a simple cache analogous to TwoSiteMPOCache could be added later


def mpo_to_ttno_tree(
    ttn,
    mpo,
    site_order=None,
    tol=1.0e-14,
    max_terms=200000,
    compress_terms=True,
):
    """Convert a chain MPO into a TTNO on an arbitrary tree topology.

    Exact conversion strategy:
    1) If the TTN is a rooted chain matching ``site_order``, keep the original
       compact MPO auxiliary bonds and only absorb the boundary vectors.
    2) Otherwise, expand the MPO into a sum of product operators via
       virtual-bond paths.
    3) Encode that sum on the tree with a shared term-index flowing on each edge.
    """
    # from ttqd.operator.ttno import TTNO
    import numpy as np

    mpo_ws = mpo.Ws if hasattr(mpo, "Ws") else mpo
    if len(mpo_ws) != len(ttn.nodes):
        raise ValueError("MPO length must equal number of TTN nodes.")

    node_keys = list(ttn.nodes.keys())
    if site_order is None:
        site_order = list(node_keys)
    if sorted(site_order) != sorted(node_keys):
        raise ValueError("site_order must be a permutation of TTN node keys.")

    def rooted_chain_path():
        path = []
        node = ttn.root
        seen = set()
        while node is not None:
            if node in seen:
                return None
            seen.add(node)
            path.append(node)
            children = tuple(ttn.children.get(node, ()))
            if len(children) > 1:
                return None
            node = children[0] if children else None
        if len(path) != len(node_keys):
            return None
        return path

    def compact_chain_ttno():
        path = rooted_chain_path()
        if path is None or list(site_order) != list(path):
            return None

        dtype = np.result_type(*[np.asarray(W).dtype for W in mpo_ws], np.float64)
        Ws_by_node = {}
        nsite = len(path)
        for p, node in enumerate(path):
            W = np.asarray(mpo_ws[p], dtype=dtype)
            if W.ndim != 4:
                raise ValueError(f"MPO tensor {p} must be rank 4, got shape {W.shape}")

            if nsite == 1:
                eL = np.zeros((W.shape[0],), dtype=dtype)
                eR = np.zeros((W.shape[1],), dtype=dtype)
                eL[0] = 1.0
                eR[-1] = 1.0
                Ws_by_node[node] = np.einsum("a,abpq,b->pq", eL, W, eR)
            elif p == 0:
                eL = np.zeros((W.shape[0],), dtype=dtype)
                eL[0] = 1.0
                Ws_by_node[node] = np.tensordot(eL, W, axes=(0, 0))
            elif p == nsite - 1:
                eR = np.zeros((W.shape[1],), dtype=dtype)
                eR[-1] = 1.0
                Ws_by_node[node] = np.tensordot(W, eR, axes=(1, 0))
            else:
                Ws_by_node[node] = W

        return TTNO.from_ttns(ttn, [Ws_by_node[node] for node in node_keys])

    chain_ttno = compact_chain_ttno()
    if chain_ttno is not None:
        return chain_ttno

    local_dims = []
    for pos, W in enumerate(mpo_ws):
        d_out = int(W.shape[2])
        d_in = int(W.shape[3])
        if d_out != d_in:
            raise ValueError(
                f"Only square local physical dimensions are supported; "
                f"site {pos} has shape {W.shape[2:]}"
            )
        local_dims.append(d_out)

    dtype = np.result_type(*[np.asarray(W).dtype for W in mpo_ws], np.float64)
    wL0 = int(mpo_ws[0].shape[0])
    wR_last = int(mpo_ws[-1].shape[1])
    eL = np.zeros((wL0,), dtype=dtype)
    eR = np.zeros((wR_last,), dtype=dtype)
    eL[0] = 1.0
    eR[-1] = 1.0

    # items: (current virtual index, scalar prefactor, [ops for visited sites])
    states = [(a0, eL[a0], []) for a0 in range(wL0) if abs(eL[a0]) > tol]
    if not states:
        raise ValueError("No nonzero left boundary state found in MPO.")

    for W in mpo_ws:
        W = np.asarray(W, dtype=dtype)
        wL, wR, _, _ = W.shape
        nz_rows = {}
        for a in range(wL):
            rows = []
            for b in range(wR):
                op = W[a, b]
                if np.max(np.abs(op)) > tol:
                    rows.append((b, op))
            nz_rows[a] = rows

        new_states = []
        for a_prev, coeff, ops in states:
            if abs(coeff) <= tol:
                continue
            for a_next, op in nz_rows.get(a_prev, []):
                new_states.append((a_next, coeff, ops + [op]))
        states = new_states

        if max_terms is not None and len(states) > int(max_terms):
            raise RuntimeError(
                f"MPO->TTNO term expansion exceeded max_terms={max_terms}. "
                "Increase max_terms or reduce system size."
            )

    terms = []
    for a_last, coeff, ops in states:
        c = coeff * eR[a_last]
        if abs(c) > tol:
            terms.append((c, ops))

    if not terms:
        raise ValueError("No nonzero MPO paths survived boundary projection.")

    if compress_terms:
        merged = {}
        for c, ops in terms:
            key = tuple(
                (
                    np.asarray(op).shape,
                    np.asarray(op).dtype.str,
                    np.asarray(op).tobytes(),
                )
                for op in ops
            )
            if key in merged:
                merged[key][0] = merged[key][0] + c
            else:
                merged[key] = [c, ops]
        terms = []
        for c, ops in merged.values():
            if abs(c) > tol:
                terms.append((c, ops))

    M = len(terms)
    pos_of = {node: p for p, node in enumerate(site_order)}

    Ws = []
    for node in node_keys:
        parent = ttn.parent.get(node, None)
        deg = len(ttn.children.get(node, ())) + (0 if parent is None else 1)
        d = local_dims[pos_of[node]]

        if deg == 0:
            Wloc = np.zeros((d, d), dtype=dtype)
            for c, ops in terms:
                Wloc += c * np.asarray(ops[pos_of[node]], dtype=dtype)
            Ws.append(Wloc)
            continue

        shape = (M,) * deg + (d, d)
        print(f"shape for node {node} = ", shape)
        Wloc = np.zeros(shape, dtype=dtype)
        for m, (c, ops) in enumerate(terms):
            fac = c if node == ttn.root else 1.0
            Wloc[(m,) * deg] = fac * np.asarray(ops[pos_of[node]], dtype=dtype)
        Ws.append(Wloc)

    return TTNO.from_ttns(ttn, Ws)
