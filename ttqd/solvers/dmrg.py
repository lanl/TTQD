import numpy as np
from ttqd.lib import logger
from ttqd.linalg.davidson import davidson_lowest_eigenpair, make_diagonal_preconditioner
from ttqd.solvers.basesolver import BaseSolver
import scipy.sparse.linalg
import scipy.sparse as sparse
import time

# ---------- function for updating environments ----------


def update_left(Wi, Ai, Li, Bi):
    r"""
    tensor contraction from the left hand side
    +-(j)   (i)----A---(j)
    |       |      |
    |       |     (p)
    L'(m) = L(k)---M---(m)
    |       |      |
    |       |     (q)
    +-(n)   (l)----B---(n)

    Ai and Bi are bra and ket states, respectively. For most case
    Bi = conj(Ai), but we here still Bi to make it general that bra state
    can be anyother state

    .. math::

        L'_{m,j,n} = sum_{kil, pq} Li[kil] Ai[i,p,j] Wi[k,m,p,q] Bi[l,q,n])

    Li: (Dl, wL, Dl)
    Ai/Bi: (Dl, d, Dr)
    Wi: (wL, wR, d_out, d_in)
    returns L(i+1): (Dr, wR, Dr)
    """
    # contracting efficiently in a sequence
    # (vL, wL, vL) * (vL, p, vR), i.e. \sum_j Li_{iaj} * Ai_{jpk} -> {iapk}
    tmp = np.tensordot(Li, Ai, axes=(2, 0))
    # sampe as (but more efficient than) out = np.einsum('iaj,jpk.->iapk', Li, Ai)

    # Wi_{km pq} T_{iapk} einsum("abpq, iapk->bqik", Wi, tmp)
    tmp = np.tensordot(Wi, tmp, axes=([0, 3], [1, 2]))    # [wL] wR i [i*], vL [wL*] [i] vR

    # einsum("iqj, ->", Bi, tmp)
    return np.tensordot(Bi, tmp, axes=([0, 1], [2, 1]))  # [vL*] [i*] vR*, wR [i] [vL] vR

    tmp = np.einsum("ipj, ikl->pkjl", Ai, Li)  # (w,ljk)
    tmp = np.einsum("pkjl,kmpq->qmjl", tmp, Wi)  # (t, mjk)
    return np.einsum("qmjl,lqn->jmn", tmp, Bi)  # (mjl)
    # return np.einsum("kil, ipj, kmpq, lqn->mjn", Li, Ai, Wi, Bi, optimize=True)


def update_right(Wi, Ai, Ri, Bi):
    r"""
    (i)-+     (i)---A----(j)
        |           |     |
        |          (p)    |
    (m)-R' =  (m)---M-(k)-R
        |           |     |
        |          (q)    |
    (n)-+     (n)---B----(l)

    .. math::
        L'_{min} = \sum_{jkl, pq) B_{nql} W_{mk,pq}  A_{ipj} R_{kjl}

    Ri: (Dr, wR, Dr)
    Ai/Bi: (Dl, d, Dr)
    Wi: (wL, wR, d_out, d_in)
    return R(i): (Dl, wL, Dl)
    """
    # contracting efficiently in a sequence (starting at an edge and moving progressively along the network)
    tmp = np.tensordot(Ai, Ri, axes=(2, 0))  # vL i [vR], [vR*] wR* vR
    tmp = np.tensordot(tmp, Wi, axes=([1, 2], [3, 1]))  # vL [i] [wR*] vR, wL [wR] i [i*]
    return np.tensordot(tmp, Bi, axes=([1, 3], [2, 1]))  # vL [vR] wL [i], vL* [i*] [vR*]

    # same as above
    tmp = np.einsum("ipj,jkl->pkil", Ai, Ri)  # d m_s^3 m_o
    tmp = np.einsum("pkil,mkpq->qmil", tmp, Wi)  # d^2 m_s^2 m_o^2
    return np.einsum("qmil,nql->imn", tmp, Bi)  # d m_s^3 m_o
    # return np.einsum("kjl, ipj, mkpq, nql->min", Ri, Ai, Wi, Bi, optimize=True)


## two-site optimization of MPS A,B with respect to MPO W1,W2 and
## environment tensors L,R
## dir = 'left' or 'right' for a left-moving or right-moving sweep

def update_twosites(i, j, psi, mpo, L, R, m, direction):
    r"""Deprecated
    TODO: remove the dependence on a 6-rank tensor
    """
    W1, W2 = mpo[i], mpo[j]

    # A, B = psi.As[i], psi.As[j]
    A, B = psi.get_As(i), psi.get_As(j)
    W1, W2 = mpo[i], mpo[j]

    W = coarse_grain_MPO(W1, W2)
    AA = coarse_grain_MPS(A, B)

    # construct effective Hamiltonian
    # print("L.shape ", L.shape)
    # print("R.shape ", R.shape)
    H = HamiltonianMultiply(L, W, R)
    # print("H.shape i =", H.shape)
    # print("AA.shape =", AA.shape)

    # update Tensors
    Ene, V = sparse.linalg.eigsh(H, 1, v0=AA, which="SA")

    AA = np.reshape(V[:, 0], H.req_shape)
    A, S, B = split_truncate_MPS(AA, [A.shape[1], B.shape[1]])
    A, S, B, trunc, m = truncate_SVD(A, S, B, m)

    if direction == "right":
        B = np.einsum("ij,jsk->isk", np.diag(S), B)
    else:
        assert direction == "left"
        A = np.einsum("isj,jk->isk", A, np.diag(S))
    # Gi = np.tensordot(np.diag(psi.Ss[i]**(-1)), A, axes=(1, 0))  # vL [vL*], [vL] i vC
    # A = np.tensordot(Gi, np.diag(S), axes=(2, 0))  # vL i [vC], [vC*] vC

    # udpate psi
    psi.As[i], psi.As[j] = A, B
    psi.Ss[j] = S
    return Ene[0] #, trunc, m


# 2-1 coarse-graining of two site MPO into one site
#  |     |  |
# -R- = -W--X-
#  |     |  |
def coarse_grain_MPO(W, X):
    return np.reshape(
        np.einsum("abst,bcuv->acsutv", W, X),
        [W.shape[0], X.shape[1], W.shape[2] * X.shape[2], W.shape[3] * X.shape[3]],
    )


# 2-1 coarse-graining of two-site MPS into one site
#   |     |  |
#  -R- = -A--B-
def coarse_grain_MPS(A, B):
    return np.reshape(
        np.einsum("isj,jtk->stik", A, B),
        [A.shape[1] * B.shape[1], A.shape[0], B.shape[2]],
    )


# opposite of coarse graining, decompose using SVD
def split_truncate_MPS(A, dims):
    assert A.shape[0] == dims[0] * dims[1]
    Theta = np.transpose(np.reshape(A, dims + [A.shape[1], A.shape[2]]), (0, 2, 1, 3))
    M = np.reshape(Theta, (dims[0] * A.shape[1], dims[1] * A.shape[2]))
    U, S, V = np.linalg.svd(M, full_matrices=0)
    # print("U.shape = ", U.shape)
    # print("dims[0] =", dims[0])
    U = np.reshape(U, (A.shape[1], dims[0], -1))
    # print("U.shape after resphae = ", U.shape)
    # print("V.shape before resphae = ", V.shape)
    V = np.reshape(V, (-1, dims[1], A.shape[2]))
    # print("V.shape after resphae = ", V.shape)
    return U, S, V



# truncate the matrices from an SVD to at most m states
def truncate_SVD(U, S, V, m, eps=1.e-10):
    m = min(len(S), m, np.sum(S > eps))
    trunc = np.sum(S[m:])
    S = S[0:m]
    U = U[:, :, 0:m]
    V = V[0:m, :, :]
    return U, S, V, trunc, m


def split_truncate_theta(theta, chi_max, eps):
    """Split and truncate a two-site wave function in mixed canonical form.

    Split a two-site wave function as follows::
          vL --(theta)-- vR     =>    vL --(A)--diag(S)--(B)-- vR
                |   |                       |             |
                i   j                       i             j

    Afterwards, truncate in the new leg (labeled ``vC``).

    Parameters
    ----------
    theta : np.Array[ndim=4]
        Two-site wave function in mixed canonical form, with legs ``vL, i, j, vR``.
    chi_max : int
        Maximum number of singular values to keep
    eps : float
        Discard any singular values smaller than that.

    Returns
    -------
    A : np.Array[ndim=3]
        Left-canonical matrix on site i, with legs ``vL, i, vC``
    S : np.Array[ndim=1]
        Singular/Schmidt values.
    B : np.Array[ndim=3]
        Right-canonical matrix on site j, with legs ``vC, j, vR``
    """
    chivL, dL, dR, chivR = theta.shape
    theta = np.reshape(theta, [chivL * dL, dR * chivR])
    U, S, V = np.linalg.svd(theta, full_matrices=False)
    # truncate
    chivC = min(chi_max, np.sum(S > eps))
    assert chivC >= 1
    piv = np.argsort(S)[::-1][:chivC]  # keep the largest `chivC` singular values
    U, S, V = U[:, piv], S[piv], V[piv, :]
    # renormalize
    S = S / np.linalg.norm(S)  # == S/sqrt(sum(S**2))
    # split legs of U and V
    A = np.reshape(U, [chivL, dL, chivC])
    B = np.reshape(V, [chivC, dR, chivR])
    return A, S, B

# Functor to evaluate the Hamiltonian matrix-vector multiply
#        +--A--+
#        |  |  |
# -R- =  L--W--R
#  |     |  |  |
#        +-   -+
class HamiltonianMultiply(sparse.linalg.LinearOperator):
    def __init__(self, L, W, R):
        self.L = L
        self.W = W
        self.R = R
        self.req_shape = [W.shape[2], L.shape[0], R.shape[2]]
        self.size = self.req_shape[0] * self.req_shape[1] * self.req_shape[2]
        super().__init__(dtype=np.dtype("d"), shape=(self.size, self.size))

    def _matvec(self, A):
        # contracting efficiently
        R = np.einsum("iaj,sik->ajsk", self.L, np.reshape(A, self.req_shape))
        R = np.einsum("ajsk,abst->bjtk", R, self.W)
        R = np.einsum("bjtk,kbl->tjl", R, self.R)
        return np.reshape(R, -1)


class Heff_0site(scipy.sparse.linalg.LinearOperator):
    """Class for the effective Hamiltonian.

    Basically the same as d_dmrg.Heff_1site, but acts on the zero-site wave function::

        .--vL*   vR*--.
        |             |
        |             |
       (LP)----------(RP)
        |             |
        |             |
        .--vL     vR--.
    """
    def __init__(self, LP, RP, prefactor=1.):
        self.LP = LP  # vL wL* vL*
        self.RP = RP  # vR* wR* vR
        chi1, chi2 = LP.shape[0], RP.shape[2]
        self.theta_shape = (chi1, chi2)  # vL vR
        size = chi1 * chi2
        super().__init__(dtype=LP.dtype, shape=(size, size))

    def _matvec(self, theta):
        """Calculate |theta'> = H_eff |theta>."""
        x = np.reshape(theta, self.theta_shape)  # vL vR
        x = np.tensordot(self.LP, x, axes=(2, 0))  # vL wL* [vL*], [vL] vR
        x = np.tensordot(x, self.RP, axes=([1, 2], [1, 0]))  # vL [wL*] [vL*] , [vR*] [wR*] vR
        x = np.reshape(x, self.shape[0])
        return x

    def _adjoint(self):
        """Define self as hermitian."""
        return self

    def trace(self):
        """The trace of the operator.

        Only needed for expm_multiply in scipy version > 1.9.0 to avoid warnings,
        but cheap to calculate anyways.
        """
        return np.inner(np.trace(self.LP, axis1=0, axis2=2),  # [vL] wL* [vL*]
                        np.trace(self.RP, axis1=0, axis2=2))  # [vR*] wR* [vR]


class Heff_1site(scipy.sparse.linalg.LinearOperator):
    """Class for the effective Hamiltonian on 1 site.

    Basically the same as dmrg.Heff2, but acts on a single site::

        .--vL*     vR*--.
        |       i*      |
        |       |       |
       (LP)----(W1)----(RP)
        |       |       |
        |       i       |
        .--vL       vR--.
    """
    def __init__(self, LP, RP, W1, prefactor=1.):
        self.LP = LP  # vL wL* vL*
        self.RP = RP  # vR* wR* vR
        self.W1 = W1  # wL wR i i*
        chi1, chi2 = LP.shape[0], RP.shape[2]
        d1 = W1.shape[2]
        self.theta_shape = (chi1, d1, chi2)  # vL i vR
        size = chi1 * d1 * chi2
        super().__init__(dtype=W1.dtype, shape=(size, size))

    def _matvec(self, theta):
        """Calculate |theta'> = H_eff |theta>."""
        x = np.reshape(theta, self.theta_shape)  # vL i vR
        x = np.tensordot(self.LP, x, axes=(2, 0))  # vL wL* [vL*], [vL] i vR
        x = np.tensordot(x, self.W1, axes=([1, 2], [0, 3]))  # vL [wL*] [i] vR, [wL] wR i [i*]
        x = np.tensordot(x, self.RP, axes=([1, 2], [0, 1]))  # vL [vR] [wR] i, [vR*] [wR*] vR
        x = np.reshape(x, self.shape[0])
        return x

    def _adjoint(self):
        """Define self as hermitian."""
        return self

    def trace(self):
        """The trace of the operator.

        Only needed for expm_multiply in scipy version > 1.9.0 to avoid warnings,
        but cheap to calculate anyways.
        """
        return np.inner(np.trace(self.LP, axis1=0, axis2=2),          # [vL] wL* [vL*]
                        np.dot(np.trace(self.W1, axis1=2, axis2=3),   # wL wR [i] [i*]
                               np.trace(self.RP, axis1=0, axis2=2)))  # [vR*] wR* [vR]


class Heff_2site(sparse.linalg.LinearOperator):
    r"""Makeing effective Hamiltonain class without
    explicitly constructing 6-index tensor
    """
    def __init__(self, Li, Rj, Wi, Wj, use_mpo_pair=True):
        self.Li = Li
        self.Rj = Rj
        self.Wi = Wi
        self.Wj = Wj
        self.nmatvec = 0

        chii, chij = Li.shape[0], Rj.shape[2]
        di, dj = Wi.shape[2], Wj.shape[2]
        self.theta_shape = (chii, di, dj, chij)  # vL i j vR
        size = chii * di * dj * chij
        dtype = np.result_type(Li, Rj, Wi, Wj)
        super().__init__(dtype=dtype, shape=(size, size))
        self.Wij = None
        self._diagonal = None
        if use_mpo_pair:
            # Combine the two local MPO tensors once. This removes one
            # contraction from every ARPACK matvec.
            Wij = np.tensordot(Wi, Wj, axes=(1, 0))
            self.Wij = np.ascontiguousarray(np.transpose(Wij, (0, 3, 1, 4, 2, 5)))

    def _matvec(self, x):
        self.nmatvec += 1
        y = np.reshape(x, self.theta_shape)  # vL i j vR
        if self.Wij is not None:
            y = np.tensordot(self.Li, y, axes=(2, 0))  # vL wL [vL*], [vL] i j vR
            y = np.tensordot(y, self.Wij, axes=([1, 2, 3], [0, 4, 5]))
            y = np.tensordot(y, self.Rj, axes=([1, 2], [0, 1]))
            return np.reshape(y, self.shape[0])

        y = np.tensordot(self.Li, y, axes=(2, 0))  # vL wL* [vL*], [vL] i j vR
        y = np.tensordot(y, self.Wi, axes=([1, 2], [0, 3]))  # vL [wL*] [i] j vR, [wL] wC i [i*]
        y = np.tensordot(y, self.Wj, axes=([3, 1], [0, 3]))  # vL [j] vR [wC] i, [wC] wR j [j*]
        y = np.tensordot(y, self.Rj, axes=([1, 3], [0, 1]))  # vL [vR] i [wR] j, [vR*] [wR*] vR
        y = np.reshape(y, self.shape[0])
        return y

    def diagonal(self):
        r"""Return the diagonal of this effective Hamiltonian.

        This is used as the Davidson preconditioner. For basis state
        ``(a, p, q, b)`` the diagonal is

            sum_{w,v} L[a,w,a] W_pair[w,v,p,q,p,q] R[b,v,b].
        """
        if self._diagonal is None:
            if self.Wij is None:
                Wij = np.tensordot(self.Wi, self.Wj, axes=(1, 0))
                Wij = np.transpose(Wij, (0, 3, 1, 4, 2, 5))
            else:
                Wij = self.Wij

            ldiag = np.einsum("awa->aw", self.Li)
            rdiag = np.einsum("bvb->bv", self.Rj)
            wdiag = np.einsum("wvpqpq->wvpq", Wij)
            diag = np.einsum("aw,wvpq,bv->apqb", ldiag, wdiag, rdiag, optimize=True)
            self._diagonal = np.real_if_close(np.reshape(diag, self.shape[0]))
        return self._diagonal

    def _adjoint(self):
        """Define self as hermitian."""
        return self

    def trace(self):
        """The trace of the operator.

        Only needed in e_tdvp.py in expm_multiply for scipy version > 1.9.0 to avoid warnings,
        but cheap to calculate anyways.
        """
        return np.inner(np.trace(self.Li, axis1=0, axis2=2),                  # [vL] wL* [vL*]
                        np.dot(np.trace(self.Wi, axis1=2, axis2=3),           # wL wC [i] [i*]
                               np.dot(np.trace(self.Wj, axis1=2, axis2=3),    # wC wR [j] [j*]
                                      np.trace(self.Rj, axis1=0, axis2=2))))  # [vR*] wR* [vR]


class BaseDMRG(BaseSolver):
    """Common DMRG driver state shared by MPS and tree solvers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.psi = None
        self.mpo = None
        self.energy = None
        self.reset_diag_stats()

    def reset_diag_stats(self):
        """Reset local diagonalization counters."""
        self.num_diag_matvec = 0
        self.num_diag_iter = 0
        self.max_diag_residual = 0.0

    def _configure_local_solver(
        self,
        kwargs,
        *,
        tol_default,
        maxiter_default,
        solver_default="eigsh",
    ):
        """Initialize shared eigsh/Davidson local-solver options."""
        self.local_solver = kwargs.get("local_solver", solver_default).lower()
        if self.local_solver == "arpack":
            self.local_solver = "eigsh"
        if self.local_solver not in ("eigsh", "davidson"):
            raise ValueError("local_solver must be 'eigsh' or 'davidson'")

        self.davidson_tol = kwargs.get("davidson_tol", None)
        self._davidson_tol_is_auto = self.davidson_tol is None
        if self.davidson_tol is None:
            self.davidson_tol = tol_default
        self.davidson_maxiter = int(kwargs.get("davidson_maxiter", maxiter_default))
        self.davidson_max_subspace = int(kwargs.get("davidson_max_subspace", 24))
        self.davidson_min_denom = float(kwargs.get("davidson_min_denom", 1.0e-12))
        self.reset_diag_stats()

    def _set_auto_davidson_tol(self, tol_default):
        """Refresh Davidson tolerance when it follows the local eigensolver tolerance."""
        if getattr(self, "_davidson_tol_is_auto", False):
            self.davidson_tol = tol_default

    def _record_davidson_stats(self, stats):
        """Accumulate Davidson iteration and residual diagnostics."""
        self.num_diag_iter += int(stats.get("iterations", 0))
        self.max_diag_residual = max(
            self.max_diag_residual,
            float(stats.get("residual_norm", 0.0)),
        )

    def _dump_local_solver_flags(
        self,
        *,
        eigsh_tol=None,
        eigsh_ncv=None,
        eigsh_maxiter=None,
        prefix="",
    ):
        """Print common local-solver flags."""
        logger.note(self, f"{prefix}Local solver    =  {self.local_solver}")
        if eigsh_tol is not None:
            logger.note(self, f"{prefix}Local eigsh tol =  {eigsh_tol:10.4e}")
        if eigsh_ncv is not None:
            logger.note(self, f"{prefix}Local eigsh ncv =  {eigsh_ncv: 5d}")
        if eigsh_maxiter is not None:
            logger.note(self, f"{prefix}Local eigsh maxiter = {eigsh_maxiter: 5d}")
        if self.local_solver == "davidson":
            logger.note(self, f"{prefix}Davidson tol    =  {self.davidson_tol:10.4e}")
            logger.note(self, f"{prefix}Davidson maxiter = {self.davidson_maxiter: 5d}")
            logger.note(self, f"{prefix}Davidson subspace = {self.davidson_max_subspace: 5d}")

    def _log_diag_summary(self):
        """Print common local diagonalization diagnostics."""
        logger.note(self, f" Total number of H matvecs in diagonalization: {self.num_diag_matvec:12d}")
        if self.local_solver == "davidson":
            logger.note(self, f" Total Davidson subspace iterations          : {self.num_diag_iter:12d}")
            logger.note(self, f" Max Davidson residual norm                  : {self.max_diag_residual:12.4e}")

    def _copy_run_stats_from(self, other):
        """Copy timing and diagonalization statistics from a delegated solver."""
        for name in (
            "energy",
            "wt_kernel",
            "wt_sweep",
            "wt_Heff",
            "wt_envs",
            "wt_prediag",
            "wt_diag",
            "wt_bond",
            "num_diag_matvec",
            "num_diag_iter",
            "max_diag_residual",
        ):
            if hasattr(other, name):
                setattr(self, name, getattr(other, name))

    def _run_sweep_iterations(
        self,
        sweep_fn,
        *,
        maxiter=None,
        convergence_message="DMRG sweep converged!",
        check_first_sweep=False,
    ):
        """Run a standard energy-convergence sweep loop."""
        nsweep = self.maxiter if maxiter is None else int(maxiter)
        logger.note(self, f"Sweep {' ' * 14} Energy {' ' * 12} Error")

        old_energy = 0.0 if check_first_sweep else None
        energy = 0.0
        for isweep in range(nsweep):
            energy = float(np.real(sweep_fn()))
            err = abs(energy - old_energy) if old_energy is not None else abs(energy)
            logger.note(self, f"Sweep {isweep:5d} {energy:18.12f} {err:18.12f}")
            if old_energy is not None and err < self.tol:
                logger.note(self, convergence_message)
                break
            old_energy = energy

        self.energy = float(np.real(energy))
        return self.energy


# ==========================
# DMRG class
# ==========================
class DMRG(BaseDMRG):
    r"""
    DMRG sweep algorithm using tree traversal order

    This implementation no longer assumes that the MPS is represented as a
    bare Python list of arrays.  Instead the solver works with the
    :class:`~ttqd.network.MPS` container and relies on its built-in tree
    structure.  Sweeps traverse the network using ``_downward_order`` and
    ``_upward_order`` which for a linear chain reduce to the familiar left‑to‑
    right and right‑to‑left passes.  Environment tensors are recomputed after
    each local update via the helper methods provided by ``MPS``.
    """

    # may add sys as an input
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.L = None
        self.R = None
        self.maxiter = kwargs.get("maxiter", 50)
        self.maxchi = kwargs.get("maxchi", 50)
        self.adapativebond = kwargs.get("adapativebond", False)
        # other options TBA
        self.chi = 1
        self.boundary = kwargs.get("boundary", "finite")
        self.threshold = kwargs.get("threshold", 1.e-10)
        self.eigsh_tol = kwargs.get("eigsh_tol", self.tol)
        if self.eigsh_tol is None:
            self.eigsh_tol = 0.0
        self._eigsh_tol_is_auto = "eigsh_tol" not in kwargs
        self.eigsh_ncv = kwargs.get("eigsh_ncv", None)
        self.eigsh_maxiter = kwargs.get("eigsh_maxiter", None)
        self.use_mpo_pair = kwargs.get("use_mpo_pair", True)
        self._configure_local_solver(
            kwargs,
            tol_default=self.eigsh_tol,
            maxiter_default=80,
        )

    def dump_flags(self):
        title = logger.task_title("DMRG flags", level=0)
        logger.note(self, title)
        logger.note(self, f"Convergence tol =  {self.tol:10.4e}")
        logger.note(self, f"Max iteration   =  {self.maxiter: 5d}")
        logger.note(self, f"Paired MPO matvec = {str(self.use_mpo_pair):>5}")
        self._dump_local_solver_flags(
            eigsh_tol=self.eigsh_tol,
            eigsh_ncv=self.eigsh_ncv,
            eigsh_maxiter=self.eigsh_maxiter,
        )

    def check_convergence(self):
        pass

    def diag(self, Heff, guess):
        guess = np.reshape(guess, [Heff.shape[1]])
        if self.local_solver == "davidson":
            preconditioner = make_diagonal_preconditioner(
                Heff.diagonal(),
                min_denom=self.davidson_min_denom,
            )
            try:
                E, V, stats = davidson_lowest_eigenpair(
                    Heff.matvec,
                    guess,
                    preconditioner=preconditioner,
                    size=Heff.shape[0],
                    dtype=Heff.dtype,
                    tol=self.davidson_tol,
                    maxiter=self.davidson_maxiter,
                    max_subspace=self.davidson_max_subspace,
                )
                self._record_davidson_stats(stats)
            finally:
                self.num_diag_matvec += getattr(Heff, "nmatvec", 0)
            return E, np.reshape(V, Heff.theta_shape)

        kwargs = {
            "k": 1,
            "which": "SA",
            "return_eigenvectors": True,
            "v0": guess,
            "tol": self.eigsh_tol,
        }
        if self.eigsh_ncv is not None:
            kwargs["ncv"] = self.eigsh_ncv
        if self.eigsh_maxiter is not None:
            kwargs["maxiter"] = self.eigsh_maxiter

        try:
            E, V = sparse.linalg.eigsh(Heff, **kwargs)
        except sparse.linalg.ArpackNoConvergence as err:
            if (
                err.eigenvalues is None
                or err.eigenvectors is None
                or len(err.eigenvalues) == 0
            ):
                raise
            idx = np.argmin(np.real(err.eigenvalues))
            E = np.array([err.eigenvalues[idx]])
            V = err.eigenvectors[:, [idx]]
        finally:
            self.num_diag_matvec += getattr(Heff, "nmatvec", 0)

        return E[0], np.reshape(V[:, 0], Heff.theta_shape)

    def compute_envs(self):
        r"""Recompute left/right MPO environments from scratch.

        The environments are 3‑index arrays ``(v, w, v)`` containing the
        contraction of the MPS and MPO up to a given bond.  This mimics the
        historic behaviour of :func:`update_left`/``update_right`` and is
        required by ``Heff_2site`` which expects the MPO leg to remain open.
        """
        pass

    def init_envs(self, mpo, psi):
        r"""initialize the left/right environment for each site"""
        self.Nsites = len(mpo)
        # assert self.Nsites == len(psi.As)
        assert self.Nsites == len(psi.nodes)

        down = psi._downward_order()
        # set number of bonds in the chain
        if self.boundary == "finite":
            # self.num_bonds = self.Nsites - 1
            self.num_bonds = len(down) - 1
        else:
            # self.num_bonds = self.Nsites
            self.num_bonds = len(down)
        logger.info(self, f"num_bonds from down = {self.num_bonds}")

        up = psi._upward_order()
        dn = psi._downward_order()
        A0 = psi.get_As(dn[0], dn[1])
        AN = psi.get_As(up[0], up[1])

        # logger.debug(self, f"A0.shape : {A0.shape}")
        # logger.debug(self, f"AN.shape : {AN.shape}")
        chiL = A0.shape[0]
        chiR = AN.shape[0]

        # self.chi = chi = psi.As[0].shape[0]  # MPS virtual bond dimension (vD)
        self.chi = chi = psi.nodes[0].tensor.data.shape[0]  # MPS virtual bond dimension (vD)
        self.wD = wD = mpo[0].shape[0]  # wD

        L0 = np.zeros((chiL, wD, chiL))  # (vL, wL, vL)
        R0 = np.zeros((chiR, wD, chiR))  # (vR, wR, vR)

        self.Lenv = [None] * self.Nsites
        self.Renv = [None] * self.Nsites
        L0[:, 0, :] = np.eye(chiL)
        R0[:, wD - 1, :] = np.eye(chiR)

        self.Lenv[dn[0]] = L0   # fist node in the downward order
        self.Renv[dn[-1]] = R0  # last node in the downward order

        # use edge as variable (directed)
        edge0 = (-1, dn[1])
        edgen = (dn[-1], -1)
        psi.init_envs_map()
        psi.Lenv[edge0] = L0
        psi.Renv[edgen] = R0
        # print("edge0/n = ", edge0, edgen)
        # print(f"***Lenv[{edge0}] is created with size {chiL}")
        # print(f"***Renv[{edgen}] is created with size {chiR}")
        # -------------------------------
        for idx in range(len(dn) - 1):
            i, j = dn[idx], dn[idx+1]
            child_j = self.psi.children.get(j, ())
            Aj = psi.get_As(j, i)
            chi_j = Aj.shape[0]
            wD = mpo[j].shape[0]  # wD
            L0[:, 0, :] = np.eye(chi_j)
            R0[:, wD - 1, :] = np.eye(chi_j)
            if len(child_j) == 0:
                psi.Renv[(j, -1)] = R0
                psi.Lenv[(-1, j)] = L0
                # print(f"***Lenv[(-1, {j})] is created with size {chi_j}")
                # print(f"***Renv[({j}, -1)] is created with size {chi_j}")
            # print(f"childeren of node {j} is {child_j}")
        # -------------------------------

        # self.Lenv = init_left_envs(mpo)
        # self.Renv = init_right_envs(mpo, psi)
        # initialize right envs
        logger.debug(self, f"Debug: upward order   = {up}")
        logger.debug(self, f"Debug: length of Lenv = {len(self.Lenv)}")
        logger.debug(self, f"Debug: length of Renv = {len(self.Renv)}")

        #for i in range(self.Nsites - 1, 1, -1):
            # j = (i - 1) % self.Nsites
        for idx in range(len(up) - 1):
            i = up[idx]
            j = up[idx+1]
            # self.Renv[j] = update_right(mpo[i], psi.As[i], self.Renv[i], psi.As[i].conj())
            # self.update_right(i, psi.As[i])
            logger.debug(self, f"\n**********updating the Renv along the bond {i} - {j}*********")
            Ai = psi.get_As(i, j)
            # Ai = psi.get_As(i, j, direction="right")
            logger.debug(self, f"A[{i}].shape = {Ai.shape}")
            logger.debug(self, f"R[{i}].shape = {self.Renv[i].shape}")
            self.update_right(i, j, Ai)

        # new version for TTN
        prev_edge = edgen
        for idx in range(len(up) - 1):
            i = up[idx]
            j = up[idx+1]
            curr_edge = (j, i)  # directed edge
            # curr_edge = (i, j)  # directed edge
            Ai = psi.get_As(i, j)
            if len(psi.children.get(i, ())) == 0:
                prev_edge = (i, -1)
                # print(f"may update edge from laef edge {prev_edge_leaf}")

            logger.debug(self, f"A[{i}].shape = {Ai.shape}")
            logger.debug(self, f"updating edge {curr_edge} from previous edge {prev_edge}")
            logger.debug(self, f"R[{prev_edge}].shape = {psi.Renv[prev_edge].shape}")

            self.update_right_edge(i, prev_edge, curr_edge, Ai, psi.Renv)
            prev_edge = curr_edge
        #exit()

    def update_all_right_envs(self, psi):
        r"""Update all lef/right envs"""
        up = psi._upward_order()
        dn = psi._downward_order()

        edgen = (dn[-1], -1)
        prev_edge = edgen
        print("***********Debug: updating the right environments********")
        for idx in range(len(up) - 1):
            i = up[idx]
            j = up[idx+1]
            curr_edge = (j, i)  # directed edge
            Ai = psi.get_As(i, j, direction="right")
            if len(psi.children.get(i, ())) == 0:
                prev_edge = (i, -1)
                # print(f"may update edge from laef edge {prev_edge_leaf}")

            logger.debug(self, f"A[{i}].shape = {Ai.shape}")
            logger.debug(self, f"updating edge {curr_edge} from previous edge {prev_edge}")
            logger.debug(self, f"R[{prev_edge}].shape = {psi.Renv[prev_edge].shape}")

            self.update_right_edge(i, prev_edge, curr_edge, Ai, psi.Renv)
            prev_edge = curr_edge

        return
        # exit()
        print("***********Debug: updating the left environments********")
        edge0 = (-1, dn[1])
        prev_edge = edge0
        for idx in range(len(dn) - 1):
            i = dn[idx]
            j = dn[idx+1]
            curr_edge = (i, j)  # directed edge
            Ai = psi.get_As(i, j)
            if len(psi.children.get(i, ())) == 0:
                prev_edge = (-1, i)
                # print(f"may update edge from laef edge {prev_edge_leaf}")

            logger.debug(self, f"A[{i}].shape = {Ai.shape}")
            logger.debug(self, f"updating edge {curr_edge} from previous edge {prev_edge}")
            logger.debug(self, f"L[{prev_edge}].shape = {psi.Lenv[prev_edge].shape}")

            self.update_left_edge(i, prev_edge, curr_edge, Ai, psi.Lenv)
            prev_edge = curr_edge
        #exit()

    def update_right_edge(self, i, prev_edge, curr_edge, B, Renv):
        r"""Compute the effective right environment along bonds """
        Ri = Renv[prev_edge]  # vR* wR* vR
        # B legs:     vL i vR
        Bc = B.conj()  # vL* i* vR*
        Wi = self.mpo[i]  # wL wR i i*
        Renv[curr_edge] = update_right(Wi, B, Ri, Bc)
        logger.debug(self, f"Debug: Right environment of edge {curr_edge} is updated from edge {prev_edge}")

    def update_left_edge(self, i, prev_edge, curr_edge, A, Lenv):
        r"""Compute the effective left environment along bonds """
        Li = Lenv[prev_edge]  # vL wL vL*
        Ac = A.conj()  # vL* i* vR*
        Wi = self.mpo[i]  # wL wR i i*
        Lenv[curr_edge] = update_left(Wi, A, Li, Ac)
        logger.debug(self, f"Debug: Left environment of edge {curr_edge} is updated from edge {prev_edge}")


    def update_right(self, i, j, B):
        """Calculate right environment on the right right of site `i-1`.

        Uses RP right of `i` and the given, right-canonical `B` on site `i`.
        """
        # j = (i - 1) % self.psi.L # replaced by the order in up/downward_order

        Ri = self.Renv[i]  # vR* wR* vR
        # B legs:     vL i vR
        Bc = B.conj()  # vL* i* vR*
        Wi = self.mpo[i]  # wL wR i i*
        self.Renv[j] = update_right(Wi, B, Ri, Bc)


    def update_left(self, i, j, A):
        """Calculate left environment on the left of site `i+1`.

        Uses the LP left of site `i` and the given, left-canonical `A` on site `i`."""
        # j = (i + 1) % self.psi.L # replaced by up/downward function that is campatible with all tensor network structure
        Li = self.Lenv[i]  # vL wL vL*
        # A legs:    vL i vR
        Ac = A.conj()  # vL* i* vR*
        Wi = self.mpo[i]  # wL wR i i*
        self.Lenv[j] = update_left(Wi, A, Li, Ac)

    def get_effective_renvs(self, i, j, psi):
        parent = self.psi.parent.get(j, ())
        children = self.psi.children.get(j, ())
        if len(children) == 0: children = (-1)
        parent = parent if isinstance(parent, tuple) else (parent, )
        children = children if isinstance(children, tuple) else (children, )
        for k in parent + children:
            if k == i: continue
            edge = (j, k)
            # print(f"{j} node has Renv[{edge}] with shape of {psi.Renv[edge].shape}")

    def update_local(self, i, j):
        r"""Form local Hamiltonian and update the bonds

        TODO: make it easy to switch between 1- and 2-site sweep
        """
        # ----------------------
        # two_site algorithm:
        # ----------------------

        # 1) get effective Hamiltonian
        logger.debug(self, f"\n{'*'*50}\n Debug: update bond between {i} and {j}\n{'*'*50}")
        t0 = time.time()

        # print("prev_ledge: ", self.prev_ledge)
        # print("prev_redge: ", self.prev_redge)
        # print(f"Lenv[{self.prev_ledge}].shape: ", self.psi.Lenv[self.prev_ledge].shape)
        # print(f"Renv[{self.prev_redge}].shape: ", self.psi.Renv[self.prev_redge].shape)
        Heff = Heff_2site(
            self.psi.Lenv[self.prev_ledge],
            self.psi.Renv[self.prev_redge],
            self.mpo[i],
            self.mpo[j],
            use_mpo_pair=self.use_mpo_pair,
        )
        # Heff = Heff_2site(self.Lenv[i], self.Renv[j], self.mpo[i], self.mpo[j])
        t1 = time.time()
        self.wt_Heff += t1 - t0

        # 2) Diagonalize Heff and update ground state
        v0 = self.psi.get_psi_2site(i, j)
        logger.debug(self, f"Debug: v0.shape   = {v0.shape}")
        logger.debug(self, f"Debug: Heff.shape = {Heff.shape}")

        E, V = self.diag(Heff, v0)
        t2 = time.time()
        self.wt_diag += t2 - t1

        children = self.psi.children.get(i, ())
        # print(f"Debug:children of {i} node is {children}")

        # 3) split and truncate
        Ai, Sj, Bj = split_truncate_theta(V, self.maxchi, 1.e-8)
        logger.debug(self, f"\nAfter split and truncate:")
        logger.debug(self, f"Debug: new A[{i}].shape is : {Ai.shape}")
        logger.debug(self, f"Debug: new S[{i}].shape is : {Sj.shape}")
        logger.debug(self, f"Debug: new A[{j}].shape is : {Bj.shape}")

        # 4) put back into MPS/TTN (moved into update_bond)
        # Gi = np.tensordot(np.diag(self.psi.Ss[i]**(-1)), Ai, axes=(1, 0))  # vL [vL*], [vL] i vC
        # self.psi.As[i] = np.tensordot(Gi, np.diag(Sj), axes=(2, 0))  # vL i [vC], [vC*] vC
        # self.psi.Ss[j] = Sj  # vC
        # self.psi.As[j] = Bj  # vC j vR
        self.psi.update_bond(i, j, Ai, Sj, Bj)
        t3 = time.time()
        self.wt_bond += t3 - t2

        # 5) update left/right envs
        # ----------------- new code ---------------
        curr_edge = (i, j)
        # print(f"get Lenv[{curr_edge}] from Lenv[{self.prev_ledge}]")
        self.update_left_edge(i, self.prev_ledge, curr_edge, Ai, self.psi.Lenv)

        # print(f"get Renv[{curr_edge}] from Renv[{self.prev_redge}]")
        adj_i = self.psi._get_adj(i)
        adj_j = self.psi._get_adj(j)
        #print(f"adjacent of site {i} = {adj_i}")
        #print(f"adjacent of site {j} = {adj_j}")

        #for k in adj_i:
        #    curr_edge(i, k):
        #    self.update_right_edge(j, self.prev_redge, curr_edge, Bj, self.psi.Renv)
        self.update_right_edge(j, self.prev_redge, curr_edge, Bj, self.psi.Renv)

        # self.update_all_right_envs(self.psi)

        # ----------------- end of new code ---------------
        # self.update_left(i, j, Ai)
        # self.update_right(j, i, Bj)

        t4 = time.time()
        self.wt_envs += t4 - t3
        return E


    def sweep(self, mpo, psi):
        t0 = time.time()
        # from left to right
        # self.Ss = [None] * self.num_bonds

        energy = None
        sweep_order = psi._downward_order()
        logger.debug(self, f"Debug: sweep down order is {sweep_order}")
        # TODO: make the following sweep compatible with any tree structure
        # TODO: using down/up order list

        edge0 = (-1, sweep_order[1])
        edgen = (sweep_order[-1], -1)
        #print("\nSeep downward")
        # TODO: to be replaced by better local update algorihtm
        for i in range(self.num_bonds - 1):
            j = (i + 1) # % self.Nsites
            # print(f"sweep from site {i} to {j}")
            if i == 0:
                self.prev_ledge = edge0
            else:
                self.prev_ledge = (sweep_order[i-1], sweep_order[i])
            self.prev_redge = (sweep_order[j], sweep_order[j+1])
            #
            if len(psi.children.get(sweep_order[i], ())) == 0:
                self.prev_ledge = (-1, sweep_order[i])
                # print("prev_ledge is changed to", self.prev_ledge)

            if len(psi.children.get(sweep_order[j], ())) == 0:
                self.prev_redge = (sweep_order[j], -1)
                # print("prev_redge is changed to", self.prev_redge)

            energy = self.update_local(sweep_order[i], sweep_order[j])

        # 2) sweep from right to left
        #print("\nSeep upward")
        # up = psi._upward_order()
        for i in range(self.num_bonds - 1, 0, -1):
            j = (i + 1) # % self.Nsites
            # print(f"sweep from site {sweep_order[i]} to {sweep_order[j]}")
            if j == self.num_bonds:
                self.prev_redge = edgen
            else:
                self.prev_redge = (sweep_order[j], sweep_order[j+1])
            self.prev_ledge = (sweep_order[i-1], sweep_order[i])
            energy = self.update_local(sweep_order[i], sweep_order[j])

        self.wt_sweep += time.time() - t0
        return energy

    @staticmethod
    def _is_general_tree_state(psi):
        """Return True for TTN states that should use the tree-DMRG backend."""
        return (
            hasattr(psi, "nodes")
            and hasattr(psi, "parent")
            and hasattr(psi, "children")
            and not hasattr(psi, "As")
        )

    def _run_tree_kernel(self, mpo, psi):
        """Dispatch a general TTN state to the TTNO tree-DMRG implementation."""
        from ttqd.operator import mpo_to_ttno_tree
        from ttqd.operator.ttno import TTNO
        from ttqd.solvers.dmrg_tree import TreeTTNODMRG2Site

        t0 = time.time()
        tree_mpo = mpo if isinstance(mpo, TTNO) else mpo_to_ttno_tree(psi, mpo)
        lanczos_maxiter = self.eigsh_maxiter
        if lanczos_maxiter is None:
            lanczos_maxiter = max(self.davidson_maxiter, 400)

        solver = TreeTTNODMRG2Site(
            maxiter=self.maxiter,
            maxchi=self.maxchi,
            cutoff=self.threshold,
            tol=self.tol,
            lanczos_tol=self.eigsh_tol,
            lanczos_maxiter=lanczos_maxiter,
            local_solver=self.local_solver,
            davidson_tol=self.davidson_tol,
            davidson_maxiter=self.davidson_maxiter,
            davidson_max_subspace=self.davidson_max_subspace,
            davidson_min_denom=self.davidson_min_denom,
        )
        solver.kernel(tree_mpo, psi, maxiter=self.maxiter, tol=self.tol)
        self.psi = psi
        self.mpo = tree_mpo
        self._copy_run_stats_from(solver)
        self.wt_kernel = time.time() - t0
        return self.energy

    def kernel(self, mpo, psi, maxiter=None, tol=1.0e-8):
        if maxiter is not None:
            self.maxiter = int(maxiter)
        if tol is not None and tol < self.tol:
            self.tol = tol
        if self._eigsh_tol_is_auto:
            self.eigsh_tol = self.tol
        self._set_auto_davidson_tol(self.eigsh_tol)

        if self._is_general_tree_state(psi):
            return self._run_tree_kernel(mpo, psi)

        t0 = time.time()
        self.dump_flags()
        logger.info(self, f"DMRG sweep with {psi.__class__.__name__}")
        print(f"\n*******DMRG sweep with {psi.__class__.__name__}***********")
        self.psi = psi
        self.mpo = mpo
        self.verbose = psi.verbose

        # 1) initialize environments
        self.init_envs(mpo, psi)

        self._run_sweep_iterations(
            lambda: self.sweep(mpo, psi),
            maxiter=maxiter,
            convergence_message="DMRG sweep converged!",
            check_first_sweep=True,
        )

        self.wt_kernel += time.time() - t0
        self.post_kernel()
        return self.energy

    def post_kernel(self):
        final_chi = [A.shape[2] for A in self.psi.As]  # .As]
        header = logger.task_title(f"Summary of {self.__class__.__name__} calculation")
        logger.note(self, header)
        logger.note(self, f" Final bond dimension {final_chi}")
        logger.note(self, f" Total wall time of DMRG kernel              : {self.wt_kernel:12.4f}")
        logger.note(self, f" Total wall time of sweep algorithm          : {self.wt_sweep:12.4f}")
        logger.note(self, f" Total wall time of constructing effective H : {self.wt_Heff:12.4f}")
        logger.note(self, f" Total wall time of constructing environments: {self.wt_envs:12.4f}")
        logger.note(self, f" Total wall time of diagonalizating H        : {self.wt_diag:12.4f}")
        self._log_diag_summary()
        logger.note(self, f" Total wall time of updating bonds           : {self.wt_bond:12.4f}")
        logger.note(self, f"\n * Hooray, the job is done!")
        fotter = logger.task_title("The end").split("\n")[1] + "\n"
        logger.note(self, fotter)


# Tree-DMRG implementations live in ``dmrg_tree`` for compatibility with older
# imports, but are re-exported here so ``ttqd.solvers.dmrg`` is the public DMRG
# module for both MPS and TTNS solvers.
try:
    from ttqd.solvers.dmrg_tree import (
        TreeDMRG,
        TreeDMRG1Site,
        TreeDMRG2Site,
        TreeTTNODMRG,
        TreeTTNODMRG1Site,
        TreeTTNODMRG2Site,
    )
except ImportError:
    # ``dmrg_tree`` imports this module for shared MPS helpers. During that
    # circular import the tree classes are not available yet; direct imports from
    # ``ttqd.solvers.dmrg`` populate them on normal module import.
    pass


__all__ = [
    "BaseDMRG",
    "DMRG",
    "TreeDMRG",
    "TreeDMRG1Site",
    "TreeDMRG2Site",
    "TreeTTNODMRG",
    "TreeTTNODMRG1Site",
    "TreeTTNODMRG2Site",
]


if __name__ == "__main__":
    d = 4  # local bond dimension, |0>, |up>, |dn>, |up,dn>
    N = 6  # number of sites
    t = -1  # hopping element
    U = 2  # on-site repulsion
    chi = 10  #

    # initialize MPS with bond dimension = 1
    InitialA1 = np.zeros((d, 1, 1))
    InitialA1[1, 0, 0] = 1
    InitialA2 = np.zeros((d, 1, 1))
    InitialA2[2, 0, 0] = 1

    MPS = [InitialA1, InitialA2] * int(N / 2)
    # print("initial state =", len(MPS))
    # for i in range(N):
    #    print(f"MPS at site {i}: ", MPS[i])

    # local operator and MPO

    MPO = None
    dmrg = DMRG()
    dmrg.kernel(MPO, MPS, maxiter=50)
