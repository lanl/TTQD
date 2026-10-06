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

This module provides a tree-aware sweep driver with both one-site and
two-site local updates. It reuses the local effective Hamiltonian operators
already implemented for MPS-style updates (`Heff_1site`, `Heff_2site`) and
applies them on edge-oriented effective tensors obtained from the TTN
(`get_As(i, j)` and `get_As(j, i, direction="right")`).

The implementation is intentionally robust:
- it builds path environments for the current traversal order,
- checks all tensor shape contracts before each environment update,
- falls back to identity environments when a branch-induced mismatch appears.

This keeps sweeps stable on arbitrary tree paths while remaining compatible
with chain/MPS objects.

"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import time
import string

import numpy as np
import scipy.sparse.linalg as spla
import scipy.linalg as scla

from ttqd.lib import logger
from ttqd.linalg.davidson import davidson_lowest_eigenpair, make_diagonal_preconditioner
from ttqd.solvers.dmrg import (
    BaseDMRG,
    Heff_1site,
    Heff_2site,
    split_truncate_theta,
    update_left,
    update_right,
)


def _as_mpo_sequence(mpo: Any) -> Sequence[Any]:
    """Return the local operator tensor container from an MPO-like object."""
    if hasattr(mpo, "Ws"):
        return mpo.Ws
    return mpo


def _prod(xs):
    """Return the product of an iterable of integer-like dimensions."""
    out = 1
    for x in xs:
        out *= int(x)
    return int(out)


def _einsum_path(expr: str, *operands):
    """Precompute a reusable contraction path for a fixed einsum signature."""
    return np.einsum_path(expr, *operands, optimize=True)[0]


def _apply_axis_matrices(tensor, matrices):
    """Apply one matrix to each tensor axis, preserving the original axis order."""
    out = tensor
    for axis, mat in enumerate(matrices):
        if mat is None:
            continue
        out = np.tensordot(mat, out, axes=(1, axis))
        out = np.moveaxis(out, 0, axis)
    return out


def _kron_metric(matrices, dtype):
    """Build the dense Kronecker-product metric in row-major flattening order."""
    out = np.asarray([[1.0]], dtype=dtype)
    for mat in matrices:
        out = np.kron(out, np.asarray(mat, dtype=dtype))
    return out


def _metric_axis_eigh(matrices, shape, dtype):
    """Diagonalize Kronecker metric factors axis-by-axis."""
    evals = []
    evecs = []
    for axis, (mat, dim) in enumerate(zip(matrices, shape)):
        M = np.asarray(mat, dtype=dtype)
        if M.shape != (dim, dim):
            raise ValueError(
                f"Metric factor on axis {axis} has shape {M.shape}, expected {(dim, dim)}"
            )
        I = np.eye(dim, dtype=dtype)
        if np.allclose(M, I, rtol=1.0e-12, atol=1.0e-14):
            evals.append(np.ones(dim, dtype=float))
            evecs.append(None)
            continue
        M = 0.5 * (M + M.conj().T)
        vals, vecs = scla.eigh(M, check_finite=False)
        evals.append(np.real(vals))
        evecs.append(vecs)
    return evals, evecs


def _metric_eigenvalue_grid(evals, reg):
    """Return eigenvalues of ``kron(metric_axes) + reg * I`` in tensor shape."""
    shape = tuple(len(v) for v in evals)
    grid = np.ones(shape, dtype=float)
    for axis, vals in enumerate(evals):
        view_shape = [1] * len(shape)
        view_shape[axis] = len(vals)
        grid *= np.asarray(vals, dtype=float).reshape(view_shape)
    grid += float(reg)
    max_abs = float(np.max(np.abs(grid))) if grid.size else 1.0
    min_val = float(np.min(grid)) if grid.size else 1.0
    tol = 100.0 * np.finfo(float).eps * max(1.0, max_abs)
    if min_val <= -tol:
        raise np.linalg.LinAlgError("Regularized metric is not positive definite.")
    return np.maximum(grid, tol)


def _apply_metric_eigen_scaling(v, shape, evecs, scale):
    """Apply ``Q f(D) Q^H`` for Kronecker-product eigenvectors."""
    X = np.reshape(v, shape)
    X = _apply_axis_matrices(X, [None if Q is None else Q.conj().T for Q in evecs])
    X = X * scale
    X = _apply_axis_matrices(X, evecs)
    return np.reshape(X, (-1,))


def _metric_axis_cholesky(matrices, shape, dtype):
    """Cholesky-factor Kronecker metric factors axis-by-axis."""
    factors = []
    for axis, (mat, dim) in enumerate(zip(matrices, shape)):
        M = np.asarray(mat, dtype=dtype)
        if M.shape != (dim, dim):
            raise ValueError(
                f"Metric factor on axis {axis} has shape {M.shape}, expected {(dim, dim)}"
            )
        I = np.eye(dim, dtype=dtype)
        if np.allclose(M, I, rtol=1.0e-12, atol=1.0e-14):
            factors.append(None)
            continue
        M = 0.5 * (M + M.conj().T)
        factors.append(scla.cholesky(M, lower=True, check_finite=False))
    return factors


def _apply_metric_cholesky_adjoint(v, shape, factors):
    """Apply ``kron(L_i).H`` without forming the dense Kronecker factor."""
    return np.reshape(
        _apply_axis_matrices(
            np.reshape(v, shape),
            [None if L is None else L.conj().T for L in factors],
        ),
        (-1,),
    )


def _solve_metric_cholesky(v, shape, factors, adjoint=False):
    """Solve against ``kron(L_i)`` or its adjoint axis-by-axis."""
    out = np.reshape(v, shape)
    for axis, L in enumerate(factors):
        if L is None:
            continue
        moved = np.moveaxis(out, axis, 0)
        mat = np.reshape(moved, (L.shape[0], -1))
        if adjoint:
            sol = scla.solve_triangular(
                L.conj().T,
                mat,
                lower=False,
                check_finite=False,
            )
        else:
            sol = scla.solve_triangular(
                L,
                mat,
                lower=True,
                check_finite=False,
            )
        out = np.moveaxis(np.reshape(sol, moved.shape), 0, axis)
    return np.reshape(out, (-1,))


# to be deprecated
class TreeDMRG(BaseDMRG):
    r"""DMRG sweep solver for tree tensor network states.

    **Main idea**:
    On a tree, cutting an edge ``(u, v)`` separates the network into two
    independent subtrees. The subtree on the ``u`` side can therefore be
    contracted into a compact *message* sent to ``v`` across that edge:

    - a norm message :math:`N^{(u \to v)}`, which summarizes overlap
      contributions of the ``u``-subtree,
    - a Hamiltonian message :math:`H^{(u \to v)}`, which summarizes the
      contribution of the same subtree to :math:`\langle \psi | \hat H | \psi \rangle`.

    These messages play the role of environments in ordinary MPS-DMRG. During a
    local update, all branches adjacent to the active node or active edge are
    replaced by their incoming messages, so the full optimization reduces to a
    local generalized eigenvalue problem.

    Parameters
    ----------
    mode : str
        `"one-site"` or `"two-site"`.
    maxiter : int
        Maximum number of sweeps in :meth:`kernel`.
    maxchi : int
        Maximum bond dimension retained by two-site SVD truncation.
    cutoff : float
        Singular-value cutoff in two-site splitting.
    lanczos_tol : float
        Tolerance for sparse eigensolver.
    lanczos_maxiter : int
        Max iterations for sparse eigensolver.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = kwargs.get("mode", "two-site")
        if self.mode not in ("one-site", "two-site"):
            raise ValueError("mode must be 'one-site' or 'two-site'")

        self.maxiter = int(kwargs.get("maxiter", 50))
        self.maxchi = int(kwargs.get("maxchi", 64))
        self.cutoff = float(kwargs.get("cutoff", 1.0e-10))
        self.lanczos_tol = float(kwargs.get("lanczos_tol", 1.0e-8))
        self.lanczos_maxiter = int(kwargs.get("lanczos_maxiter", 50))
        self.allow_branching_chain_mpo = bool(kwargs.get("allow_branching_chain_mpo", False))
        self._configure_local_solver(
            kwargs,
            tol_default=self.lanczos_tol,
            maxiter_default=self.lanczos_maxiter,
        )
        self.metric_transform_threshold = int(kwargs.get("metric_transform_threshold", 1024))
        self.ham_precontract_max_size = int(kwargs.get("ham_precontract_max_size", 1_000_000))

        self._node_to_pos: Dict[Any, int] = {}
        self._has_directional_get = False

    def _is_branching_state(self) -> bool:
        """Return ``True`` when the current TTN has a node of degree greater than two."""
        if not hasattr(self.psi, "children"):
            return False
        for u in self._node_keys():
            deg = len(self.psi.children.get(u, ()))
            par = None
            if hasattr(self.psi, "parent"):
                par = self.psi.parent.get(u, None)
            if par is not None:
                deg += 1
            if deg > 2:
                return True
        return False

    def _is_chain_mpo(self) -> bool:
        """Return ``True`` when every local operator tensor has standard MPO rank four."""
        seq = _as_mpo_sequence(self.mpo)
        if isinstance(seq, dict):
            vals = list(seq.values())
        else:
            vals = list(seq)
        if not vals:
            return False
        return all(getattr(W, "ndim", None) == 4 for W in vals)

    def _edge_order(self, i: Any) -> Tuple[Any, ...]:
        """Return the canonical edge order ``(parent, children...)`` for node ``i``."""
        parent = None
        if hasattr(self.psi, "parent"):
            parent = self.psi.parent.get(i, None)
        children = ()
        if hasattr(self.psi, "children"):
            children = tuple(self.psi.children.get(i, ()))
        if parent is None:
            return tuple(children)
        return (parent,) + tuple(children)

    def _is_true_ttno_layout(self) -> bool:
        """Check whether each local operator has one auxiliary leg per tree edge.

        True TTNO layout convention used by :class:`TreeTTNODMRG`:
        ``W_i.shape == (w_e0, ..., w_ek, d_out, d_in)``, where ``k`` is node degree.
        """
        try:
            keys = self._node_keys()
            for i in keys:
                W = self._op_at(i)
                if getattr(W, "ndim", None) != len(self._edge_order(i)) + 2:
                    return False
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------
    def dump_flags(self):
        """Print the solver configuration used for the current run."""
        title = logger.task_title("Tree DMRG flags", level=0)
        logger.note(self, title)
        logger.note(self, f" Update mode      = {self.mode}")
        logger.note(self, f" Convergence tol  = {self.tol:10.4e}")
        logger.note(self, f" Max sweeps       = {self.maxiter:5d}")
        logger.note(self, f" Max bond dim     = {self.maxchi:5d}")
        logger.note(self, f" SVD cutoff       = {self.cutoff:10.4e}")
        self._dump_local_solver_flags(
            eigsh_tol=self.lanczos_tol,
            eigsh_maxiter=self.lanczos_maxiter,
            prefix=" ",
        )

    def _node_keys(self) -> List[Any]:
        """Return node labels from the state object in a stable traversal order."""
        if hasattr(self.psi, "nodes"):
            return list(self.psi.nodes.keys())
        if hasattr(self.psi, "L"):
            return list(range(self.psi.L))
        raise ValueError("Could not infer node keys from psi")

    def _configure_operator_accessor(self):
        """Build ``self._op_at`` so local operators can be retrieved by node label."""
        seq = _as_mpo_sequence(self.mpo)
        node_keys = self._node_keys()

        if isinstance(seq, dict):
            missing = [k for k in node_keys if k not in seq]
            if missing:
                raise ValueError(f"Operator dictionary missing keys: {missing}")
            self._op_at = lambda k: seq[k]
            return

        if not hasattr(seq, "__len__"):
            raise ValueError("mpo must be a sequence, dict, or object with .Ws")

        if len(seq) != len(node_keys):
            raise ValueError(
                f"Operator length ({len(seq)}) and number of nodes ({len(node_keys)}) differ"
            )

        if all(isinstance(k, (int, np.integer)) and 0 <= int(k) < len(seq) for k in node_keys):
            self._op_at = lambda k: seq[int(k)]
            return

        self._node_to_pos = {k: i for i, k in enumerate(node_keys)}
        self._op_at = lambda k: seq[self._node_to_pos[k]]

    # ------------------------------------------------------------------
    # State tensor adapters (TreeTensorNetwork and MPS compatible)
    # ------------------------------------------------------------------
    def _get_As_left(self, i: Any, j: Any):
        """Return the tensor on node ``i`` oriented toward edge ``(i, j)``."""
        try:
            return self.psi.get_As(i, j)
        except TypeError:
            return self.psi.get_As(i)

    def _get_As_right(self, i: Any, j: Any):
        """Return the tensor on node ``i`` viewed from the right side of edge ``(i, j)``."""
        if self._has_directional_get:
            return self.psi.get_As(i, j, direction="right")
        return self.psi.get_As(i, j)

    def _set_As_left(self, i: Any, j: Any, A):
        """Write an oriented tensor back to node ``i`` along edge ``(i, j)``."""
        try:
            self.psi.set_As(i, j, A)
        except TypeError:
            self.psi.set_As(i, A)

    def _set_As_plain(self, i: Any, A):
        """Write a non-oriented tensor back to node ``i``."""
        self.psi.set_As(i, A)

    def _set_edge_s(self, i: Any, j: Any, S):
        """Store Schmidt values on edge ``(i, j)`` using the state API available."""
        try:
            self.psi.set_S(i, j, S)
        except TypeError:
            if hasattr(self.psi, "Ss"):
                self.psi.Ss[j] = S
            else:
                raise

    # ------------------------------------------------------------------
    # Environment helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _left_identity(dim: int, wdim: int, dtype):
        """Return a left boundary environment representing the identity channel."""
        out = np.zeros((dim, wdim, dim), dtype=dtype)
        out[:, 0, :] = np.eye(dim, dtype=dtype)
        return out

    @staticmethod
    def _right_identity(dim: int, wdim: int, dtype):
        """Return a right boundary environment representing the identity channel."""
        out = np.zeros((dim, wdim, dim), dtype=dtype)
        out[:, wdim - 1, :] = np.eye(dim, dtype=dtype)
        return out

    @staticmethod
    def _compatible_left(Li, A, W):
        """Check whether a left environment matches tensor ``A`` and operator ``W``."""
        return (
            Li.shape[0] == A.shape[0]
            and Li.shape[2] == A.shape[0]
            and Li.shape[1] == W.shape[0]
            and A.shape[1] == W.shape[2]
            and A.shape[1] == W.shape[3]
        )

    @staticmethod
    def _compatible_right(Ri, B, W):
        """Check whether a right environment matches tensor ``B`` and operator ``W``."""
        return (
            Ri.shape[0] == B.shape[2]
            and Ri.shape[2] == B.shape[2]
            and Ri.shape[1] == W.shape[1]
            and B.shape[1] == W.shape[2]
            and B.shape[1] == W.shape[3]
        )

    def _sanitize_left_env(self, Li, A, W):
        """Replace a missing or incompatible left environment by an identity one."""
        dtype = np.result_type(A.dtype, W.dtype)
        if Li is None:
            return self._left_identity(A.shape[0], W.shape[0], dtype)
        if not self._compatible_left(Li, A, W):
            return self._left_identity(A.shape[0], W.shape[0], dtype)
        return Li

    def _sanitize_right_env(self, Ri, B, W):
        """Replace a missing or incompatible right environment by an identity one."""
        dtype = np.result_type(B.dtype, W.dtype)
        if Ri is None:
            return self._right_identity(B.shape[2], W.shape[1], dtype)
        if not self._compatible_right(Ri, B, W):
            return self._right_identity(B.shape[2], W.shape[1], dtype)
        return Ri

    def _propagate_left_env(self, Li, A, W):
        """Advance a left environment through one site along the current path."""
        dtype = np.result_type(A.dtype, W.dtype)
        if self._compatible_left(Li, A, W):
            return update_left(W, A, Li, np.conj(A))
        return self._left_identity(A.shape[2], W.shape[1], dtype)

    def _propagate_right_env(self, Ri, B, W):
        """Advance a right environment through one site along the current path."""
        dtype = np.result_type(B.dtype, W.dtype)
        if self._compatible_right(Ri, B, W):
            return update_right(W, B, Ri, np.conj(B))
        return self._right_identity(B.shape[0], W.shape[0], dtype)

    def _build_path_envs(self, path: Sequence[Any]):
        r"""Build left/right effective environments for a traversal path.

        Returns dictionaries keyed by directed edges `(i, j)` where `i` and `j`
        are consecutive nodes in `path`.
        """
        ledge: Dict[Tuple[Any, Any], Any] = {}
        redge: Dict[Tuple[Any, Any], Any] = {}

        if len(path) < 2:
            return ledge, redge

        # left-to-right pass
        i0, j0 = path[0], path[1]
        A0 = self._get_As_left(i0, j0)
        W0 = self._op_at(i0)
        Lcur = self._left_identity(A0.shape[0], W0.shape[0], np.result_type(A0.dtype, W0.dtype))
        for k in range(len(path) - 1):
            i, j = path[k], path[k + 1]
            A = self._get_As_left(i, j)
            W = self._op_at(i)
            Lcur = self._sanitize_left_env(Lcur, A, W)
            ledge[(i, j)] = Lcur
            Lcur = self._propagate_left_env(Lcur, A, W)

        # right-to-left pass
        ia, ib = path[-2], path[-1]
        Blast = self._get_As_right(ib, ia)
        Wb = self._op_at(ib)
        Rcur = self._right_identity(Blast.shape[2], Wb.shape[1], np.result_type(Blast.dtype, Wb.dtype))
        for k in range(len(path) - 2, -1, -1):
            i, j = path[k], path[k + 1]
            B = self._get_As_right(j, i)
            W = self._op_at(j)
            Rcur = self._sanitize_right_env(Rcur, B, W)
            redge[(i, j)] = Rcur
            Rcur = self._propagate_right_env(Rcur, B, W)

        return ledge, redge

    # ------------------------------------------------------------------
    # Local solves
    # ------------------------------------------------------------------
    def _diag_local(self, Heff, guess):
        """Solve the lowest-eigenvalue local problem for an effective Hamiltonian."""
        dim = int(Heff.shape[0])
        v0 = np.asarray(guess).reshape(dim)

        if dim == 1:
            Hv = Heff.matvec(np.array([1.0], dtype=v0.dtype))
            val = np.real_if_close(Hv[0])
            return float(np.real(val)), np.asarray([1.0], dtype=v0.dtype).reshape(Heff.theta_shape)

        if self.local_solver == "davidson":
            preconditioner = None
            if hasattr(Heff, "diagonal"):
                preconditioner = make_diagonal_preconditioner(
                    Heff.diagonal(),
                    min_denom=self.davidson_min_denom,
                )
            try:
                energy, vec, stats = davidson_lowest_eigenpair(
                    Heff.matvec,
                    v0,
                    preconditioner=preconditioner,
                    size=dim,
                    dtype=getattr(Heff, "dtype", v0.dtype),
                    tol=self.davidson_tol,
                    maxiter=self.davidson_maxiter,
                    max_subspace=self.davidson_max_subspace,
                )
                self._record_davidson_stats(stats)
                return float(np.real(energy)), np.reshape(vec, Heff.theta_shape)
            finally:
                self.num_diag_matvec += getattr(Heff, "nmatvec", 0)

        try:
            vals, vecs = spla.eigsh(
                Heff,
                k=1,
                which="SA",
                v0=v0,
                tol=self.lanczos_tol,
                maxiter=self.lanczos_maxiter,
                return_eigenvectors=True,
            )
            self.num_diag_matvec += getattr(Heff, "nmatvec", 0)
            return float(np.real(vals[0])), np.reshape(vecs[:, 0], Heff.theta_shape)
        except Exception:
            # Dense fallback for small local spaces.
            if dim > 1024:
                raise
            eye = np.eye(dim, dtype=v0.dtype)
            H = np.column_stack([Heff.matvec(eye[:, k]) for k in range(dim)])
            H = 0.5 * (H + H.conj().T)
            evals, evecs = np.linalg.eigh(H)
            idx = int(np.argmin(np.real(evals)))
            return float(np.real(evals[idx])), np.reshape(evecs[:, idx], Heff.theta_shape)

    def _update_edge_one_site(self, i: Any, j: Any, path: Sequence[Any]):
        """Perform a one-site update on node ``i`` while sweeping across edge ``(i, j)``."""
        ledge, redge = self._build_path_envs(path)

        Ai = self._get_As_left(i, j)
        Wi = self._op_at(i)
        Li = self._sanitize_left_env(ledge.get((i, j)), Ai, Wi)

        # Build right env for site i by folding site j into the edge-right env.
        Bj = self._get_As_right(j, i)
        Wj = self._op_at(j)
        Rj = self._sanitize_right_env(redge.get((i, j)), Bj, Wj)
        if self._compatible_right(Rj, Bj, Wj):
            RP = update_right(Wj, Bj, Rj, np.conj(Bj))
        else:
            RP = self._right_identity(Ai.shape[2], Wi.shape[1], np.result_type(Ai.dtype, Wi.dtype))

        if RP.shape[0] != Ai.shape[2] or RP.shape[1] != Wi.shape[1] or RP.shape[2] != Ai.shape[2]:
            RP = self._right_identity(Ai.shape[2], Wi.shape[1], np.result_type(Ai.dtype, Wi.dtype))

        Heff = Heff_1site(Li, RP, Wi)
        t0 = time.time()
        energy, Aopt = self._diag_local(Heff, Ai)
        self.wt_diag += time.time() - t0
        t1 = time.time()
        self._set_As_left(i, j, Aopt)
        self.wt_bond += time.time() - t1
        return energy

    def _update_edge_two_site(self, i: Any, j: Any, path: Sequence[Any]):
        """Perform a two-site update on the active path edge ``(i, j)``."""
        ledge, redge = self._build_path_envs(path)

        Ai = self._get_As_left(i, j)
        Bj = self._get_As_right(j, i)
        Wi = self._op_at(i)
        Wj = self._op_at(j)

        Li = self._sanitize_left_env(ledge.get((i, j)), Ai, Wi)
        Rj = self._sanitize_right_env(redge.get((i, j)), Bj, Wj)

        Heff = Heff_2site(Li, Rj, Wi, Wj)
        theta0 = np.tensordot(Ai, Bj, axes=(2, 0))
        t0 = time.time()
        energy, theta = self._diag_local(Heff, theta0)
        self.wt_diag += time.time() - t0

        t1 = time.time()
        Anew, Snew, Bnew = split_truncate_theta(theta, self.maxchi, self.cutoff)

        # Delegate gauge bookkeeping to the network implementation.
        # - MPS.update_bond applies the legacy Schmidt-gauge convention.
        # - BaseTreeTensorNetwork.update_bond stores edge Schmidt values and
        #   writes oriented tensors through set_As(..., direction="right").
        self.psi.update_bond(i, j, Anew, Snew, Bnew)
        self.wt_bond += time.time() - t1

        return energy

    # ------------------------------------------------------------------
    # Sweep drivers
    # ------------------------------------------------------------------
    def _sweep_path(self, path: Sequence[Any]):
        """Apply local updates along a directed sweep path and return the last energy."""
        if len(path) < 2:
            return 0.0

        energy = 0.0
        for i, j in zip(path[:-1], path[1:]):
            if self.mode == "one-site":
                energy = self._update_edge_one_site(i, j, path)
            else:
                energy = self._update_edge_two_site(i, j, path)
        print(f"Debug: seeping along edge {i} {j}")
        return float(np.real(energy))

    def sweep(self):
        """Run one full down-and-up sweep of the current tree state."""
        t0 = time.time()
        down = list(self.psi._downward_order())
        up = list(self.psi._upward_order())

        energy = self._sweep_path(down)
        if len(up) >= 2:
            energy = self._sweep_path(up)

        self.wt_sweep += time.time() - t0
        return energy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def kernel(self, mpo, psi, maxiter: int | None = None, tol: float | None = None):
        """Run the tree-DMRG optimization until convergence or ``maxiter`` sweeps."""
        t0 = time.time()
        self.psi = psi
        self.mpo = mpo
        if maxiter is not None:
            self.maxiter = int(maxiter)
        if tol is not None:
            self.tol = float(tol)
        self.verbose = getattr(psi, "verbose", self.verbose)

        self._configure_operator_accessor()
        if self._is_true_ttno_layout():
            solver = TreeTTNODMRG(
                mode=self.mode,
                maxiter=self.maxiter,
                maxchi=self.maxchi,
                cutoff=self.cutoff,
                tol=self.tol,
                lanczos_tol=self.lanczos_tol,
                lanczos_maxiter=self.lanczos_maxiter,
                local_solver=self.local_solver,
                davidson_tol=self.davidson_tol,
                davidson_maxiter=self.davidson_maxiter,
                davidson_max_subspace=self.davidson_max_subspace,
                davidson_min_denom=self.davidson_min_denom,
            )
            solver.kernel(mpo, psi, maxiter=self.maxiter, tol=self.tol)
            self._copy_run_stats_from(solver)
            return self.energy
        if self._is_branching_state() and self._is_chain_mpo() and (not self.allow_branching_chain_mpo):
            raise NotImplementedError(
                "TreeDMRG with a branching TTN and a 1D chain MPO is not supported by the "
                "current edge-local Heff factorization. Use MPS/chain topology for chain MPOs, "
                "or provide a tree-structured operator and tree-specific environments."
            )

        # Detect directional get_As interface (TreeTensorNetwork style).
        self._has_directional_get = False
        probe = list(self.psi._downward_order())
        if len(probe) >= 2:
            try:
                _ = self.psi.get_As(probe[1], probe[0], direction="right")
                self._has_directional_get = True
            except TypeError:
                self._has_directional_get = False

        self.dump_flags()
        print(f"\n*******DMRG sweep with {psi.__class__.__name__}***********")
        self._run_sweep_iterations(
            self.sweep,
            maxiter=self.maxiter,
            convergence_message="Tree DMRG sweep converged!",
        )
        self.wt_kernel += time.time() - t0
        self.post_kernel()
        return self.energy

    def post_kernel(self):
        chis = None
        if hasattr(self.psi, "Smap"):
            chis = [len(v) for v in self.psi.Smap.values()]
        elif hasattr(self.psi, "Ss"):
            chis = [len(v) for v in self.psi.Ss]
        header = logger.task_title(f"Summary of {self.__class__.__name__} calculation")
        logger.note(self, header)
        if chis is not None:
            logger.note(self, f" Final bond dimensions {chis}")
        logger.note(self, f" Final energy                               : {self.energy:18.12f}")
        logger.note(self, f" Total wall time of TreeDMRG kernel        : {self.wt_kernel:12.4f}")
        logger.note(self, f" Total wall time of sweep algorithm         : {self.wt_sweep:12.4f}")
        logger.note(self, f" Total wall time of local diagonalizations  : {self.wt_diag:12.4f}")
        self._log_diag_summary()
        logger.note(self, f" Total wall time of bond updates            : {self.wt_bond:12.4f}")


class TreeTTNODMRG(BaseDMRG):
    r"""True TTNO-based DMRG for branching TTNs.

    Assumes each local TTNO tensor follows:
      ``W_i.shape == (w_e0, ..., w_ek, d_out, d_in)``
    where ``e0..ek`` follow edge order `(parent, children...)` of node ``i``.

    The central construction is the tree *message*. For any directed edge
    ``u -> v``, the whole subtree rooted at ``u`` and cut away from ``v`` is
    contracted into:

    - :math:`N^{(u \to v)}`, a reduced norm/overlap tensor on the bond index,
    - :math:`H^{(u \to v)}`, an effective Hamiltonian environment carrying both
      bond and TTNO auxiliary indices.

    Because a tree has no loops, these messages can be built recursively and
    reused throughout the sweep. The local one-site and two-site DMRG updates
    then assemble effective operators from the incoming branch messages instead
    of re-contracting the entire network each time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = kwargs.get("mode", "two-site")
        if self.mode not in ("one-site", "two-site"):
            raise ValueError("mode must be 'one-site' or 'two-site'")

        self.maxiter = int(kwargs.get("maxiter", 50))
        self.maxchi = int(kwargs.get("maxchi", 64))
        self.cutoff = float(kwargs.get("cutoff", 1.0e-10))
        self.lanczos_tol = float(kwargs.get("lanczos_tol", 1.0e-10))
        self.lanczos_maxiter = int(kwargs.get("lanczos_maxiter", 400))
        self.norm_reg = float(kwargs.get("norm_reg", 0.0))
        self.dense_limit = int(kwargs.get("dense_limit", 1024))
        self._configure_local_solver(
            kwargs,
            tol_default=self.lanczos_tol,
            maxiter_default=self.lanczos_maxiter,
        )
        self.metric_transform_threshold = int(kwargs.get("metric_transform_threshold", 1024))
        self.ham_precontract_max_size = int(kwargs.get("ham_precontract_max_size", 1_000_000))
        self._node_to_pos: Dict[Any, int] = {}
        self._msgH_cache: Dict[Tuple[Any, Any], Any] = {}
        self._msgN_cache: Dict[Tuple[Any, Any], Any] = {}
        self._node_pack_cache: Dict[Any, Tuple[Any, Dict[str, Any]]] = {}

    @staticmethod
    def _symbols(n: int) -> List[str]:
        """Return a pool of einsum symbols of length ``n``."""
        pool = list(string.ascii_letters)
        if n > len(pool):
            raise ValueError(f"einsum symbol pool exhausted for n={n}")
        return pool[:n]

    def dump_flags(self):
        """Print the TTNO-tree DMRG configuration for the current run."""
        title = logger.task_title("Tree TTNO-DMRG flags", level=0)
        logger.note(self, title)
        logger.note(self, f" Update mode      = {self.mode}")
        logger.note(self, f" Convergence tol  = {self.tol:10.4e}")
        logger.note(self, f" Max sweeps       = {self.maxiter:5d}")
        logger.note(self, f" Max bond dim     = {self.maxchi:5d}")
        logger.note(self, f" SVD cutoff       = {self.cutoff:10.4e}")
        logger.note(self, f" Norm regularizer = {self.norm_reg:10.4e}")
        logger.note(self, f" H precontract max= {self.ham_precontract_max_size:5d}")
        self._dump_local_solver_flags(
            eigsh_tol=self.lanczos_tol,
            eigsh_maxiter=self.lanczos_maxiter,
            prefix=" ",
        )
        if self.local_solver == "davidson":
            logger.note(self, f" Metric transform = {self.metric_transform_threshold:5d}")

    def _node_keys(self) -> List[Any]:
        """Return node labels from the TTN state."""
        return list(self.psi.nodes.keys())

    def _configure_operator_accessor(self):
        """Build ``self._op_at`` so TTNO tensors can be fetched by node label."""
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
        """Return neighbors of ``i`` in the TTNO leg order ``(parent, children...)``."""
        parent = self.psi.parent.get(i, None)
        children = tuple(self.psi.children.get(i, ()))
        if parent is None:
            return children
        return (parent,) + children

    def _edge_order(self, i: Any) -> Tuple[Any, ...]:
        """Return the canonical ordering of operator/state bond legs for node ``i``."""
        return self._neighbors(i)

    def _clear_caches(self):
        """Clear cached subtree messages and packed tensor views."""
        self._msgH_cache.clear()
        self._msgN_cache.clear()
        self._node_pack_cache.clear()

    def _shared_bond_name(self, i: Any, j: Any) -> str:
        """Return the unique state-bond identifier shared by adjacent nodes ``i`` and ``j``."""
        shared = self.psi._shared_bond_inds(i, j)
        if len(shared) != 1:
            raise ValueError(f"Expected exactly one shared state bond on edge ({i},{j}), got {shared}")
        return shared[0]

    def _pack_state_node(self, i: Any):
        r"""Pack node ``i`` into ``(edge_1, \ldots, edge_k, physical)`` format.

        All physical legs are fused into a single trailing index, while virtual
        bond legs are ordered according to :meth:`_edge_order`.
        """
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
            raise ValueError(
                f"Node {i} must only contain edge and physical indices for TTNO-DMRG."
            )

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
        """Undo :meth:`_pack_state_node` and store the tensor back into the TTN."""
        edge_shape = tuple(A_pack.shape[:-1])
        if info["phys_shape"]:
            A_rs = np.reshape(A_pack, edge_shape + tuple(info["phys_shape"]))
        else:
            A_rs = np.reshape(A_pack, edge_shape)
        inv_perm = np.argsort(np.asarray(info["perm"], dtype=int))
        A_new = np.transpose(A_rs, inv_perm) if len(info["perm"]) > 1 else A_rs
        self.psi.nodes[i].tensor.data = A_new
        if hasattr(self.psi, "As") and isinstance(self.psi.As, list) and i < len(self.psi.As):
            self.psi.As[i] = A_new

    def _op_tensor(self, i: Any):
        """Return the TTNO tensor on node ``i`` after validating local dimensions."""
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
        """Check that TTN and TTNO share the same rooted-tree topology."""
        if hasattr(self.mpo, "parent") and dict(self.mpo.parent) != dict(self.psi.parent):
            raise ValueError("TTNO parent mapping does not match TTN parent mapping.")
        if hasattr(self.mpo, "children") and dict(self.mpo.children) != dict(self.psi.children):
            raise ValueError("TTNO children mapping does not match TTN children mapping.")
        if hasattr(self.mpo, "root") and self.mpo.root != self.psi.root:
            raise ValueError("TTNO root does not match TTN root.")
        for i in self._node_keys():
            self._op_tensor(i)

    def _msg_norm(self, u: Any, v: Any):
        r"""Return the norm message propagated from subtree ``u`` toward node ``v``.

        The result leaves open only the state bond connecting ``u`` and ``v``.
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
        r"""Return the Hamiltonian message propagated from subtree ``u`` to ``v``.

        The result leaves open the state bond toward ``v`` and the TTNO
        auxiliary leg on the same cut edge.
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

    def _solve_local_generalized(
        self,
        mvH,
        mvN,
        guess,
        metric_matrix=None,
        metric_axes=None,
        metric_shape=None,
    ):
        r"""Solve the matrix-free generalized problem ``H x = \lambda N x``.

        Small local spaces are treated densely; larger ones use sparse Lanczos
        with the norm operator as metric and dense regularized fallbacks.
        """
        v0 = np.asarray(guess).reshape(-1)
        n = v0.size
        dtype = np.result_type(v0.dtype, np.complex128 if np.iscomplexobj(v0) else np.float64)

        def _nmat(v):
            if self.norm_reg != 0.0:
                return mvN(v) + self.norm_reg * v
            return mvN(v)

        def _nmat_reg(v, reg):
            return mvN(v) + reg * v

        def _metric_matrix(reg):
            if metric_matrix is None:
                if metric_axes is None:
                    eye = np.eye(n, dtype=dtype)
                    N = np.column_stack([_nmat_reg(eye[:, k], reg) for k in range(n)])
                else:
                    N = _kron_metric(metric_axes, dtype)
                    if reg != 0.0:
                        N.flat[:: n + 1] += reg
            else:
                N = np.array(metric_matrix, dtype=dtype, copy=True)
                if reg != 0.0:
                    N.flat[:: n + 1] += reg
            return 0.5 * (N + N.conj().T)

        def _dense_solve(reg):
            eye = np.eye(n, dtype=dtype)
            H = np.column_stack([mvH(eye[:, k]) for k in range(n)])
            N = _metric_matrix(reg)
            H = 0.5 * (H + H.conj().T)
            evals, evecs = scla.eigh(H, N, check_finite=False)
            idx = int(np.argmin(np.real(evals)))
            x = evecs[:, idx]
            nrm = np.sqrt(np.real(np.vdot(x, N @ x)))
            x = x / nrm
            return float(np.real(evals[idx])), x

        def _regularization_candidates():
            if self.norm_reg == 0.0:
                base = 1.0e-14
                return [0.0] + [base * (10.0 ** k) for k in range(8)]
            base = abs(float(self.norm_reg))
            return [base * (10.0 ** k) for k in range(8)]

        def _davidson_metric_solve(reg):
            N = _metric_matrix(reg)
            L = scla.cholesky(N, lower=True, check_finite=False)

            def to_x(y):
                return scla.solve_triangular(
                    L.conj().T,
                    y,
                    lower=False,
                    check_finite=False,
                )

            def to_y(x):
                return scla.solve_triangular(
                    L,
                    x,
                    lower=True,
                    check_finite=False,
                )

            h_matvecs = 0

            def matvec_metric(y):
                nonlocal h_matvecs
                h_matvecs += 1
                return to_y(mvH(to_x(y)))

            y0 = L.conj().T @ v0
            energy, yopt, stats = davidson_lowest_eigenpair(
                matvec_metric,
                y0,
                preconditioner=None,
                size=n,
                dtype=dtype,
                tol=self.davidson_tol,
                maxiter=self.davidson_maxiter,
                max_subspace=self.davidson_max_subspace,
            )
            x = to_x(yopt)
            nrm = np.sqrt(np.real(np.vdot(x, N @ x)))
            x = x / nrm
            self.num_diag_matvec += h_matvecs
            self._record_davidson_stats(stats)
            return float(np.real(energy)), x

        def _davidson_metric_kron_cholesky_solve():
            if metric_axes is None or metric_shape is None:
                raise ValueError("Kronecker metric factors are required.")
            factors = _metric_axis_cholesky(metric_axes, metric_shape, dtype)

            def to_x(y):
                return _solve_metric_cholesky(y, metric_shape, factors, adjoint=True)

            def to_y(x):
                return _solve_metric_cholesky(x, metric_shape, factors, adjoint=False)

            h_matvecs = 0

            def matvec_metric(y):
                nonlocal h_matvecs
                h_matvecs += 1
                return to_y(mvH(to_x(y)))

            y0 = _apply_metric_cholesky_adjoint(v0, metric_shape, factors)
            energy, yopt, stats = davidson_lowest_eigenpair(
                matvec_metric,
                y0,
                preconditioner=None,
                size=n,
                dtype=dtype,
                tol=self.davidson_tol,
                maxiter=self.davidson_maxiter,
                max_subspace=self.davidson_max_subspace,
            )
            x = to_x(yopt)
            nrm = np.linalg.norm(yopt)
            if nrm != 0.0:
                x = x / nrm
            self.num_diag_matvec += h_matvecs
            self._record_davidson_stats(stats)
            return float(np.real(energy)), x

        metric_eig = None

        def _davidson_metric_spectral_solve(reg):
            nonlocal metric_eig
            if metric_axes is None or metric_shape is None:
                raise ValueError("Kronecker metric factors are required.")
            if metric_eig is None:
                metric_eig = _metric_axis_eigh(metric_axes, metric_shape, dtype)
            evals, evecs = metric_eig
            weights = _metric_eigenvalue_grid(evals, reg)
            scale_mhalf = weights ** -0.5
            scale_phalf = weights ** 0.5

            def to_x(y):
                return _apply_metric_eigen_scaling(y, metric_shape, evecs, scale_mhalf)

            h_matvecs = 0

            def matvec_metric(y):
                nonlocal h_matvecs
                h_matvecs += 1
                return to_x(mvH(to_x(y)))

            y0 = _apply_metric_eigen_scaling(v0, metric_shape, evecs, scale_phalf)
            energy, yopt, stats = davidson_lowest_eigenpair(
                matvec_metric,
                y0,
                preconditioner=None,
                size=n,
                dtype=dtype,
                tol=self.davidson_tol,
                maxiter=self.davidson_maxiter,
                max_subspace=self.davidson_max_subspace,
            )
            x = to_x(yopt)
            nrm = np.linalg.norm(yopt)
            if nrm != 0.0:
                x = x / nrm
            self.num_diag_matvec += h_matvecs
            self._record_davidson_stats(stats)
            return float(np.real(energy)), x

        if n == 1:
            x = np.asarray([1.0], dtype=dtype)
            denom = _nmat(x)[0]
            e = np.real_if_close(mvH(x)[0] / denom)
            return float(np.real(e)), x

        if self.local_solver == "davidson":
            if metric_axes is not None and float(self.norm_reg) == 0.0:
                try:
                    return _davidson_metric_kron_cholesky_solve()
                except Exception:
                    pass
            if metric_axes is not None and n >= self.metric_transform_threshold:
                for reg in _regularization_candidates():
                    try:
                        return _davidson_metric_spectral_solve(reg)
                    except Exception:
                        continue
            for reg in _regularization_candidates():
                try:
                    return _davidson_metric_solve(reg)
                except Exception:
                    continue
            if n > self.dense_limit:
                # If the metric transform is ill-conditioned, fall through to
                # sparse generalized Lanczos before dense fallback.
                pass
            else:
                for reg in _regularization_candidates():
                    try:
                        return _dense_solve(reg)
                    except Exception:
                        continue
                raise RuntimeError("Davidson and dense generalized eigensolves failed.")

        if n <= self.dense_limit:
            for reg in _regularization_candidates():
                try:
                    return _dense_solve(reg)
                except Exception:
                    continue
            raise RuntimeError("Dense generalized eigensolve failed for all regularization levels.")

        h_matvecs = 0

        def mvH_counted(v):
            nonlocal h_matvecs
            h_matvecs += 1
            return mvH(v)

        Aop = spla.LinearOperator((n, n), matvec=mvH_counted, dtype=dtype)
        Mop = spla.LinearOperator((n, n), matvec=_nmat, dtype=dtype)
        try:
            vals, vecs = spla.eigsh(
                Aop,
                k=1,
                M=Mop,
                which="SA",
                v0=v0,
                tol=self.lanczos_tol,
                maxiter=self.lanczos_maxiter,
                return_eigenvectors=True,
            )
            x = vecs[:, 0]
            nrm = np.sqrt(np.real(np.vdot(x, _nmat(x))))
            x = x / nrm
            self.num_diag_matvec += h_matvecs
            return float(np.real(vals[0])), x
        except Exception:
            self.num_diag_matvec += h_matvecs
            # Fallback to dense generalized solve with adaptive regularization.
            for reg in _regularization_candidates():
                try:
                    return _dense_solve(reg)
                except Exception:
                    continue
            raise RuntimeError("Generalized eigensolver failed in both sparse and dense fallback paths.")

    def _update_one_site(self, i: Any):
        r"""
        Perform a one-site tree-DMRG update at node ``i``.

        Let :math:`x \equiv \mathrm{vec}(A_i)` denote the packed tensor at the
        active node after all neighboring subtrees have been contracted into
        incoming Hamiltonian and norm messages. If node ``i`` has neighbors
        :math:`e_1,\dots,e_k`, we write the local state tensor as

        .. math::

            A_i \equiv A^{[i]}_{\alpha_1 \cdots \alpha_k,\, p},

        where :math:`\alpha_r` labels the virtual bond on edge :math:`e_r` and
        :math:`p` is the physical index. The packed vector :math:`x` is obtained
        by flattening the multi-index :math:`(\alpha_1,\dots,\alpha_k,p)`.

        For each neighboring branch :math:`e_r`, the recursion over the
        corresponding subtree produces a norm message
        :math:`N^{(e_r \to i)}_{\alpha_r \beta_r}` and a Hamiltonian message
        :math:`H^{(e_r \to i)}_{\alpha_r\, \omega_r\, \beta_r}`, where
        :math:`\omega_r` is the local TTNO auxiliary index on edge :math:`e_r`.
        In terms of these messages, the effective norm acts as

        .. math::

            \bigl(N_{\mathrm{eff}}^{(i)} x\bigr)_{\alpha_1 \cdots \alpha_k,\, p}
            =
            \sum_{\beta_1,\dots,\beta_k}
            \left[
                \prod_{r=1}^k
                N^{(e_r \to i)}_{\alpha_r \beta_r}
            \right]
            x_{\beta_1 \cdots \beta_k,\, p},

        while the effective Hamiltonian acts as

        .. math::

            \bigl(H_{\mathrm{eff}}^{(i)} x\bigr)_{\alpha_1 \cdots \alpha_k,\, p}
            =
            \sum_{\substack{\beta_1,\dots,\beta_k\\ \omega_1,\dots,\omega_k\\ q}}
            W^{[i]}_{\omega_1 \cdots \omega_k,\, p q}
            \left[
                \prod_{r=1}^k
                H^{(e_r \to i)}_{\alpha_r\, \omega_r\, \beta_r}
            \right]
            x_{\beta_1 \cdots \beta_k,\, q}.

        Equivalently, the branch messages are the partial contractions of all
        descendant tensors outside the active node. For a neighbor ``u`` of
        ``i``, they satisfy the recursive relations

        .. math::

            N^{(u \to i)}_{\alpha \beta}
            =
            \sum_{p, \{\gamma\}, \{\delta\}}
            A^{[u]}_{\{\gamma\},\, \alpha,\, p}\,
            \overline{A^{[u]}_{\{\delta\},\, \beta,\, p}}
            \prod_{n \in \partial u \setminus i}
            N^{(n \to u)}_{\gamma_n \delta_n},

        .. math::

            H^{(u \to i)}_{\alpha\, \omega\, \beta}
            =
            \sum_{\substack{p,q\\ \{\gamma\}, \{\delta\}\\ \{\omega_n\}}}
            A^{[u]}_{\{\gamma\},\, \alpha,\, p}\,
            W^{[u]}_{\{\omega_n\},\, \omega,\, p q}\,
            \overline{A^{[u]}_{\{\delta\},\, \beta,\, q}}
            \prod_{n \in \partial u \setminus i}
            H^{(n \to u)}_{\gamma_n\, \omega_n\, \delta_n}.

        The local variational problem is then

        .. math::

            E[x] = \frac{x^\dagger H_{\mathrm{eff}}^{(i)} x}
                        {x^\dagger N_{\mathrm{eff}}^{(i)} x},

        which leads to the generalized eigenvalue equation

        .. math::

            H_{\mathrm{eff}}^{(i)} x = \lambda\, N_{\mathrm{eff}}^{(i)} x.

        Here :math:`H_{\mathrm{eff}}^{(i)}` is assembled from the TTNO tensor on
        node ``i`` together with the Hamiltonian messages from every adjacent
        branch, while :math:`N_{\mathrm{eff}}^{(i)}` is assembled from the norm
        messages of those same branches. After solving for the lowest-energy
        local vector :math:`x`, the result is reshaped and unpacked back into the
        tensor stored on node ``i``.
        """
        self._clear_caches()
        A_pack, info = self._pack_state_node(i)
        edges = list(info["edges"])
        W, _ = self._op_tensor(i)

        k = len(edges)
        d = A_pack.shape[-1]
        edge_dims = A_pack.shape[:-1]
        in_msgs_H = [self._msg_ham(n, i) for n in edges]
        in_msgs_N = [self._msg_norm(n, i) for n in edges]
        metric_dtype = np.result_type(A_pack.dtype, W.dtype)
        eye_d = np.eye(d, dtype=metric_dtype)

        sy = self._symbols(3 * k + 2)
        ket = sy[:k]
        op = sy[k : 2 * k]
        bra = sy[2 * k : 3 * k]
        p, q = sy[-2], sy[-1]

        W_sub = "".join(op) + p + q
        msgH_sub = [ket[r] + op[r] + bra[r] for r in range(k)]
        x_sub = "".join(bra) + q
        out_sub = "".join(ket) + p

        exprH = ",".join([W_sub] + msgH_sub + [x_sub]) + "->" + out_sub

        shape = edge_dims + (d,)
        size = int(_prod(shape))
        probe = np.empty(shape, dtype=metric_dtype)
        pathH = _einsum_path(exprH, W, *in_msgs_H, probe)
        metric_axes = list(in_msgs_N) + [eye_d]
        metric_matrix = None

        def mvH(v):
            X = np.reshape(v, shape)
            Y = np.einsum(exprH, W, *in_msgs_H, X, optimize=pathH)
            return np.reshape(Y, (size,))

        def mvN(v):
            X = np.reshape(v, shape)
            Y = _apply_axis_matrices(X, metric_axes)
            return np.reshape(Y, (size,))

        t0 = time.time()
        energy, xopt = self._solve_local_generalized(
            mvH,
            mvN,
            np.reshape(A_pack, (size,)),
            metric_matrix=metric_matrix,
            metric_axes=metric_axes,
            metric_shape=shape,
        )
        self.wt_diag += time.time() - t0

        t1 = time.time()
        self._unpack_state_node(i, np.reshape(xopt, shape), info)
        self.wt_bond += time.time() - t1
        self._clear_caches()
        return energy

    def _update_two_site(self, i: Any, j: Any):
        r"""
        Perform a two-site tree-DMRG update on the edge ``(i, j)``.

        The tensors on nodes ``i`` and ``j`` are first packed into a joint
        two-site center tensor

        .. math::

            \Theta_{(\alpha_i)\,p_i\,p_j\,(\alpha_j)}
            = \sum_{\beta_{ij}}
              A^{[i]}_{(\alpha_i)\,p_i\,\beta_{ij}}
              A^{[j]}_{\beta_{ij}\,p_j\,(\alpha_j)},

        where :math:`(\alpha_i)` and :math:`(\alpha_j)` collect all external
        branch indices attached to ``i`` and ``j`` other than the bond
        :math:`(i,j)`. After packing :math:`x \equiv \mathrm{vec}(\Theta)`, the
        local variational problem is

        .. math::

            E[x] = \frac{x^\dagger H_{\mathrm{eff}}^{(ij)} x}
                        {x^\dagger N_{\mathrm{eff}}^{(ij)} x},

        with generalized eigenvalue equation

        .. math::

            H_{\mathrm{eff}}^{(ij)} x = \lambda\, N_{\mathrm{eff}}^{(ij)} x.

        As in the one-site case, the effective operators are obtained by
        contracting all spectator branches into Hamiltonian and norm messages,
        leaving only the active edge open. Once the optimal two-site tensor
        :math:`\Theta_{\mathrm{opt}}` is found, it is factorized as

        .. math::

            \Theta_{\mathrm{opt}}
            \approx A_{\mathrm{new}}\, \mathrm{diag}(S_{\mathrm{new}})\, B_{\mathrm{new}},

        and the singular values are absorbed to the left tensor before the two
        updated node tensors are written back onto the tree.
        """
        if j not in self._neighbors(i):
            raise ValueError(f"Nodes {i} and {j} are not adjacent in tree.")
        self._clear_caches()

        t0 = time.time()

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

        Wi, _ = self._op_tensor(i)
        Wj, _ = self._op_tensor(j)
        msgHi = [self._msg_ham(n, i) for n in ext_i]
        msgHj = [self._msg_ham(n, j) for n in ext_j]
        msgNi = [self._msg_norm(n, i) for n in ext_i]
        msgNj = [self._msg_norm(n, j) for n in ext_j]

        Ri = int(_prod(ext_i_dims)) if ext_i_dims else 1
        Rj = int(_prod(ext_j_dims)) if ext_j_dims else 1

        # Current two-site center tensor as initial guess.
        Ai_tmp = np.transpose(Ai_pack, ext_i_pos + [len(ei), pos_ij])  # ext_i, pi, ij
        Aj_tmp = np.transpose(Aj_pack, [pos_ji, len(ej)] + ext_j_pos)  # ij, pj, ext_j
        Ai_red = np.reshape(Ai_tmp, (Ri, d_i, D_ij))
        Aj_red = np.reshape(Aj_tmp, (D_ij, d_j, Rj))
        theta0 = np.tensordot(Ai_red, Aj_red, axes=(2, 0))  # Ri, pi, pj, Rj
        theta_shape = tuple(ext_i_dims) + (d_i, d_j) + tuple(ext_j_dims)
        theta0 = np.reshape(theta0, theta_shape)
        size = int(_prod(theta_shape))

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

        msgHi_sub = [ki[r] + oi[r] + bi[r] for r in range(ni)]
        msgHj_sub = [kj[r] + oj[r] + bj[r] for r in range(nj)]

        x_sub = "".join(bi) + qi + qj + "".join(bj)
        out_sub = "".join(ki) + pi + pj + "".join(kj)
        probe = np.empty(theta_shape, dtype=np.result_type(theta0.dtype, Wi.dtype, Wj.dtype))

        wij_shape = (
            tuple(Wi.shape[a] for a in range(Wi.ndim) if a != pos_ij)
            + tuple(Wj.shape[a] for a in range(Wj.ndim) if a != pos_ji)
        )
        use_precontract = int(_prod(wij_shape)) <= self.ham_precontract_max_size
        if use_precontract:
            Wij = np.tensordot(Wi, Wj, axes=(pos_ij, pos_ji))
            Wij_sub = "".join(oi) + pi + qi + "".join(oj) + pj + qj
            h_operands = (Wij, *msgHi, *msgHj)
            h_probe_operands = (Wij, *msgHi, *msgHj, probe)
            exprH = ",".join([Wij_sub] + msgHi_sub + msgHj_sub + [x_sub]) + "->" + out_sub
        else:
            wi_edges_sub = [
                u if n == j else oi[ext_i.index(n)]
                for n in ei
            ]
            wj_edges_sub = [
                u if n == i else oj[ext_j.index(n)]
                for n in ej
            ]
            Wi_sub = "".join(wi_edges_sub) + pi + qi
            Wj_sub = "".join(wj_edges_sub) + pj + qj
            h_operands = (Wi, Wj, *msgHi, *msgHj)
            h_probe_operands = (Wi, Wj, *msgHi, *msgHj, probe)
            exprH = ",".join([Wi_sub, Wj_sub] + msgHi_sub + msgHj_sub + [x_sub]) + "->" + out_sub

        # print("exprH = ", exprH)
        eye_i = np.eye(d_i, dtype=np.result_type(theta0.dtype, Wi.dtype, Wj.dtype))
        eye_j = np.eye(d_j, dtype=np.result_type(theta0.dtype, Wi.dtype, Wj.dtype))
        pathH = _einsum_path(exprH, *h_probe_operands)
        metric_axes = list(msgNi) + [eye_i, eye_j] + list(msgNj)
        metric_matrix = None
        self.wt_prediag += time.time() - t0

        def mvH(v):
            X = np.reshape(v, theta_shape)
            Y = np.einsum(exprH, *h_operands, X, optimize=pathH)
            return np.reshape(Y, (size,))

        def mvN(v):
            X = np.reshape(v, theta_shape)
            Y = _apply_axis_matrices(X, metric_axes)
            return np.reshape(Y, (size,))

        t0 = time.time()
        logger.debug(self, f"DEBUG: Solving local Heff with theta_spae {theta_shape}, size = {size}")
        energy, xopt = self._solve_local_generalized(
            mvH,
            mvN,
            np.reshape(theta0, (size,)),
            metric_matrix=metric_matrix,
            metric_axes=metric_axes,
            metric_shape=theta_shape,
        )
        logger.debug(self, f"DEBUG: Solving local Heff --- Done!")
        self.wt_diag += time.time() - t0

        t1 = time.time()
        theta_opt = np.reshape(xopt, theta_shape)
        theta_opt = np.reshape(theta_opt, (Ri, d_i, d_j, Rj))
        Anew, Snew, Bnew = split_truncate_theta(theta_opt, self.maxchi, self.cutoff)
        Anew = np.tensordot(Anew, np.diag(Snew), axes=(2, 0))  # absorb S to left

        # Put updated left tensor back to node i.
        Aleft = np.reshape(Anew, tuple(ext_i_dims) + (d_i, Anew.shape[2]))  # ext_i, pi, ij
        edge_axis_map_i = {n: idx for idx, n in enumerate(ext_i)}
        edge_axis_map_i[j] = len(ext_i) + 1
        perm_edges_i = [edge_axis_map_i[n] for n in ei]
        Ai_new_pack = np.transpose(Aleft, perm_edges_i + [len(ext_i)])  # edges_i..., pi
        self._unpack_state_node(i, Ai_new_pack, info_i)

        # Put updated right tensor back to node j.
        Bright = np.reshape(Bnew, (Bnew.shape[0], d_j) + tuple(ext_j_dims))  # ij, pj, ext_j
        edge_axis_map_j = {i: 0}
        for idx, n in enumerate(ext_j):
            edge_axis_map_j[n] = 2 + idx
        perm_edges_j = [edge_axis_map_j[n] for n in ej]
        Aj_new_pack = np.transpose(Bright, perm_edges_j + [1])  # edges_j..., pj
        self._unpack_state_node(j, Aj_new_pack, info_j)

        if hasattr(self.psi, "set_S"):
            try:
                self.psi.set_S(i, j, Snew)
            except TypeError:
                if hasattr(self.psi, "Ss") and isinstance(j, int) and j < len(self.psi.Ss):
                    self.psi.Ss[j] = Snew
        self.wt_bond += time.time() - t1
        self._clear_caches()
        return energy

    def _sweep_path(self, path: Sequence[Any]):
        """Apply TTNO local updates along one directed sweep path."""
        if len(path) < 2:
            return 0.0
        energy = 0.0
        for i, j in zip(path[:-1], path[1:]):
            logger.debug(self, f"DEBUG: Sweep along the ege {i} - {j}")
            if self.mode == "one-site":
                energy = self._update_one_site(i)
            else:
                energy = self._update_two_site(i, j)
        return float(np.real(energy))

    def sweep(self):
        """Run one full TTNO sweep using downward and upward traversals."""
        t0 = time.time()
        down = list(self.psi._downward_order())
        up = list(self.psi._upward_order())
        logger.debug(self, f"DEBUG: Sweep (down) path is: {down}")
        energy = self._sweep_path(down)
        if len(up) >= 2:
            energy = self._sweep_path(up)
        self.wt_sweep += time.time() - t0
        return energy

    def kernel(self, mpo, psi, maxiter: int | None = None, tol: float | None = None):
        """Run TTNO-based tree DMRG until convergence or the sweep limit is reached."""
        t0 = time.time()
        self.psi = psi
        self.mpo = mpo
        if maxiter is not None:
            self.maxiter = int(maxiter)
        if tol is not None:
            self.tol = float(tol)
        self.verbose = getattr(psi, "verbose", self.verbose)

        self._configure_operator_accessor()
        self._validate_topology()

        self.dump_flags()
        print(f"\n*******DMRG sweep with {psi.__class__.__name__}***********")
        self._run_sweep_iterations(
            self.sweep,
            maxiter=self.maxiter,
            convergence_message="Tree TTNO-DMRG sweep converged!",
        )
        self.wt_kernel += time.time() - t0
        self.post_kernel()
        return self.energy

    def post_kernel(self):
        """Print a timing and bond-dimension summary after the solver finishes."""
        #
        chis = None
        if hasattr(self.psi, "Smap"):
            chis = {key: len(v) for key, v in self.psi.Smap.items()}
        elif hasattr(self.psi, "Ss"):
            chis = [len(v) for v in self.psi.Ss]
        header = logger.task_title(f"Summary of {self.__class__.__name__} calculation")
        logger.note(self, header)
        if chis is not None:
            logger.note(self, f" Final bond dimensions {chis}")
        logger.note(self, f" Final energy                               : {self.energy:18.12f}")
        logger.note(self, f" Total wall time of Tree TTNO-DMRG kernel   : {self.wt_kernel:12.4f}")
        logger.note(self, f" Total wall time of sweep algorithm         : {self.wt_sweep:12.4f}")
        logger.note(self, f" Total wall time of preparing diag          : {self.wt_prediag:12.4f}")
        logger.note(self, f" Total wall time of local diagonalizations  : {self.wt_diag:12.4f}")
        self._log_diag_summary()
        logger.note(self, f" Total wall time of bond updates            : {self.wt_bond:12.4f}")


class TreeTTNODMRG1Site(TreeTTNODMRG):
    r"""True TTNO-DMRG with one-site local updates."""

    def __init__(self, *args, **kwargs):
        kwargs["mode"] = "one-site"
        super().__init__(*args, **kwargs)


class TreeTTNODMRG2Site(TreeTTNODMRG):
    r"""True TTNO-DMRG with two-site local updates."""

    def __init__(self, *args, **kwargs):
        kwargs["mode"] = "two-site"
        super().__init__(*args, **kwargs)


class TreeDMRG1Site(TreeDMRG):
    r"""Tree DMRG with one-site local updates."""

    def __init__(self, *args, **kwargs):
        kwargs["mode"] = "one-site"
        super().__init__(*args, **kwargs)


class TreeDMRG2Site(TreeDMRG):
    r"""Tree DMRG with two-site local updates."""

    def __init__(self, *args, **kwargs):
        kwargs["mode"] = "two-site"
        super().__init__(*args, **kwargs)
