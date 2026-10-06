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
Propagators
-----------
"""

import sys
import time
from typing import Dict, Tuple, Optional, List, Any, Iterable
from typing import Sequence, Callable
from ttqd.lib import logger
from ttqd.linalg import backend
from ttqd.network import MPS
from ttqd.network.mps import split_truncate_theta

from ttqd.operator import MPO, TwoSiteMPOCache
from ttqd.solvers.dmrg import update_left, update_right, Heff_0site, Heff_1site, Heff_2site
import scipy
import scipy.sparse.linalg
from scipy.sparse.linalg import expm
import numpy as np


def lanczos_expm_multiply(matvec, v, dt: complex, m: int = 20, eps: float = 1e-12):
    """
    Compute exp(dt * H) v for (approximately) Hermitian H using Lanczos.
    Works on any shaped v (tensor), using vdot/norm.
    """
    beta0 = be.norm(v)
    if float(be.to_numpy(beta0)) < eps:
        return v

    q_prev = None
    q = v / beta0

    alphas = []
    betas = []
    Q = [q]

    for j in range(m):
        w = matvec(q)
        alpha = be.vdot(q, w)
        w = w - alpha * q
        if q_prev is not None:
            w = w - betas[-1] * q_prev

        beta = be.norm(w)

        alphas.append(alpha)
        if j < m - 1:
            betas.append(beta)

        if float(be.to_numpy(beta)) < eps:
            break

        q_prev = q
        q = w / beta
        Q.append(q)

    k = len(alphas)

    # Build small tridiagonal T on host (stable) then cast
    import numpy as np
    T = np.zeros((k, k), dtype=np.complex128)
    for i in range(k):
        T[i, i] = complex(np.asarray(be.to_numpy(alphas[i])))
        if i < k - 1:
            b = float(np.asarray(be.to_numpy(betas[i])))
            T[i, i + 1] = b
            T[i + 1, i] = b
    T = be.asarray(T)

    w, U = be.eigh(T)
    ew = be.exp(dt * w)

    e1 = be.asarray(np.array([1.0] + [0.0] * (k - 1), dtype=np.complex128))
    Ut_e1 = be.einsum("ij,j->i", be.conj(be.transpose(U, (1, 0))), e1) if hasattr(be, "contract") else (be.conj(be.transpose(U, (1, 0))) @ e1)
    coeffs = (U * ew[None, :]) @ Ut_e1

    y = 0
    for j in range(k):
        y = y + coeffs[j] * Q[j]
    return beta0 * y





class PropagatorBase(object):
    """Class implementing propagators.

    Parameters
    ----------
    mpo : tensor network Hamiltonian (MPO or xx)
    psi : tensor network state
        Initial state.
    dt : float, optional
        Default time step, cannot be set as well as ``tol``.
    tmax : float, optional
        Final time. Defaults to 10.0.
    tol : float, optional
        Default target error for each evolution, cannot be set as well as
        ``dt``.
    t0 : float, optional
        Initial time. Defaults to 0.0.
    """

    def __init__(
        self,
        mpo,
        psi,
        dt=0.1,
        tmax=10.0,
        tol=None,
        t0=0.0,
        **kwargs,
    ):
        self.psi = psi
        self.mpo = mpo
        self.dt = float(dt)
        self.tol = tol
        self.t0 = float(t0)
        self.tmax = float(tmax)

        # update kwargs
        self.imaginary = bool(kwargs.get("imaginary", False))
        self.adaptive_bond = bool(kwargs.get("adaptive_bond", False))
        self.maxchi = int(kwargs.get("maxchi", 50))

        self.print_freq = kwargs.get("print_freq", 1) # may overwrite it with ttn.verbose
        self.verbose = kwargs.get("verbose", 3) # may overwrite it with ttn.verbose
        self.stdout = sys.stdout
        self.energy = None
        self.nsteps_done = 0

        self.wt_setup = 0.0   # wall time of propagator setup
        self.wt_kernel = 0.0  # total wall time
        self.wt_sweep = 0.0   # wall time of sweep/step algorithms
        self.wt_Heff = 0.0    # wall time of constructing effective Hamiltonians
        self.wt_envs = 0.0    # wall time of constructing/updating environments
        self.wt_prediag = 0.0 # wall time of preparing local propagators
        self.wt_diag = 0.0    # wall time of local exponentials
        self.wt_bond = 0.0    # wall time of splitting/truncating/updating bonds
        self.wt_energy = 0.0  # wall time of energy measurements

    def step(self, state: Any) -> None:
        raise NotImplementedError("This function should be implemeted in inherited class")


    def run(self, state: Any, nsteps: int, callback: Optional[Callable[[int, Any], None]] = None) -> None:
        for n in range(nsteps):
            self.step(state)
            if callback is not None:
                callback(n, state)

    def dump_flags(self):
        print(f"\n{'*' * 60}\n{' ' * 8} Tensornetwork dynamics with {self.__class__.__name__}\n{'*' * 60}")
        print(f"Total time is:        {self.tmax:10.3f}")
        print(f"Time step is:         {self.dt:10.3f}")
        print(f"Maximal chi :         {self.maxchi:10d}")

    def _final_bond_dimensions(self):
        psi = getattr(self, "psi", None)
        if psi is None:
            return None
        if hasattr(psi, "get_chi"):
            try:
                return psi.get_chi()
            except Exception:
                pass
        if hasattr(psi, "Smap"):
            try:
                return {key: len(val) for key, val in psi.Smap.items()}
            except Exception:
                pass
        if hasattr(psi, "Ss"):
            try:
                return [len(val) for val in psi.Ss]
            except Exception:
                pass
        if hasattr(psi, "As"):
            try:
                return [A.shape[-1] for A in psi.As]
            except Exception:
                pass
        return None

    def post_kernel(self):
        header = logger.task_title(f"Summary of {self.__class__.__name__} calculation")
        logger.note(self, header)
        chis = self._final_bond_dimensions()
        if chis is not None:
            logger.note(self, f" Final bond dimensions                     : {chis}")
        if self.energy is not None:
            try:
                energy = float(np.real_if_close(self.energy))
                logger.note(self, f" Final energy                              : {energy:18.12f}")
            except (TypeError, ValueError):
                logger.note(self, f" Final energy                              : {self.energy}")
        logger.note(self, f" Completed time steps                      : {self.nsteps_done:12d}")
        logger.note(self, f" Total wall time including setup           : {self.wt_setup + self.wt_kernel:12.4f}")
        logger.note(self, f" Total wall time of propagator setup       : {self.wt_setup:12.4f}")
        logger.note(self, f" Total wall time of propagation kernel     : {self.wt_kernel:12.4f}")
        logger.note(self, f" Total wall time of sweep algorithm        : {self.wt_sweep:12.4f}")
        logger.note(self, f" Total wall time of constructing Heff      : {self.wt_Heff:12.4f}")
        logger.note(self, f" Total wall time of updating environments  : {self.wt_envs:12.4f}")
        logger.note(self, f" Total wall time of local exponentials     : {self.wt_diag:12.4f}")
        logger.note(self, f" Total wall time of updating bonds         : {self.wt_bond:12.4f}")
        logger.note(self, f" Total wall time of measuring energy       : {self.wt_energy:12.4f}")



class TEBD(PropagatorBase):
    """Class implementing Time Evolving Block Decimation (TEBD) :cite:`Vidal:2003aa`.

    """

    #def __init__(self, *args, **kwargs):
    #    super().__init__(*args, **kwargs)

    def __init__(
        self,
        h2_bonds: Sequence[Any],   # length L-1, each (d^2,d^2) or (d,d,d,d)
        psi, # TTN
        dt: float = 0.1,
        tmax: float = 10.0,
        t0: float = 0.0,
        order: int = 2,
        imaginary: bool = False,
        cutoff: float = 1e-12,
        **kwargs,
    ):

        super().__init__(h2_bonds, psi, dt=dt, tmax=tmax, tol=tol, t0=t0, **kwargs)

        self.h2_bonds = list(h2_bonds)
        self.order = int(order)
        self.imaginary = bool(imaginary)
        self.cutoff = float(cutoff)
        self._gates_even = None
        self._gates_odd = None
        self._gates_even_half = None

        #
        self.psi = psi


    def _build_gate(self, h2: Any, dt: float, d: int):
        H = be.asarray(h2)
        if H.ndim == 4:
            H = be.reshape(H, (d * d, d * d))
        # Hermitian exp via eigh
        Hh = 0.5 * (H + be.conj(be.transpose(H, (1, 0))))
        w, V = be.eigh(Hh)
        if self.imaginary:
            ew = be.exp((-dt) * w)
        else:
            ew = be.exp((-1j * dt) * w)
        U = (V * ew[None, :]) @ be.conj(be.transpose(V, (1, 0)))
        return be.reshape(U, (d, d, d, d))  # (p'i,p'j,pi,pj)

    def _ensure_gates(self, mps: MPS):
        if self._gates_even is not None:
            return
        be = mps.be
        d = mps.d
        self._gates_even = []
        self._gates_odd = []
        for i, h2 in enumerate(self.h2_bonds):
            U = self._build_gate(be, h2, self.dt, d)
            (self._gates_even if (i % 2 == 0) else self._gates_odd).append((i, U))
        if self.order == 2:
            self._gates_even_half = []
            for i, h2 in enumerate(self.h2_bonds):
                if i % 2 == 0:
                    U = self._build_gate(be, h2, 0.5 * self.dt, d)
                    self._gates_even_half.append((i, U))

    def step(self, mps: MPS) -> None:
        r"""core steping algorithm"""

        # TBA
        self._ensure_gates(mps)
        if self.order == 1:
            for i, U in self._gates_even:
                mps.apply_two_site_gate(i, U, self.maxchi, self.cutoff)
            for i, U in self._gates_odd:
                mps.apply_two_site_gate(i, U, self.maxchi, self.cutoff)
        elif self.order == 2:
            for i, U in self._gates_even_half:
                mps.apply_two_site_gate(i, U, self.maxchi, self.cutoff)
            for i, U in self._gates_odd:
                mps.apply_two_site_gate(i, U, self.maxchi, self.cutoff)
            for i, U in self._gates_even_half:
                mps.apply_two_site_gate(i, U, self.maxchi, self.cutoff)
        else:
            raise ValueError("order must be 1 or 2")



# class TVDP(PropagatorBase):
class TDVP1(PropagatorBase):
    r"""Class implementing Time-Dependent Variational Principle (TDVP) :cite:`Haegeman:2011aa`

    """
    #def __init__(self, *args, **kwargs):
    #    super().__init__(*args, **kwargs)

    def __init__(self,
        mpo: MPO = None,
        psi=None,
        dt: float = 0.1,
        tmax: float = 10.0,
        tol=None,
        t0: float = 0.0,
        krylov_dim: int = 20,
        symmetric: bool = False,
        **kwargs,
    ):
        # initialize base attributes (tmax, maxchi, flags, etc.)
        super().__init__(mpo, psi, dt=dt, tmax=tmax, tol=tol, t0=t0, **kwargs)
        self.Hbonds = kwargs.get("Hbonds", None)
        self.krylov_dim = int(krylov_dim)
        self.symmetric = bool(symmetric)
        self.nsteps = kwargs.get("nsteps", None)
        self.times = []
        self.energies = []
        self.expectation_energies = []
        self.conserve_energy_history = bool(kwargs.get("conserve_energy_history", False))
        self.mps = psi
        self.Nsites = None
        self.Lenv = None
        self.Renv = None

        if not bool(kwargs.get("_defer_setup", False)) and mpo is not None and psi is not None:
            self._configure_state(mpo, psi)

    def _configure_state(self, mpo, psi):
        """Configure the MPS-specific TDVP state and environments."""
        self.mpo = mpo
        self.Nsites = len(mpo.Ws)
        self.mps = self.psi = psi
        assert self.Nsites == len(psi.As)

        self.chi = chi = psi.As[0].shape[0]  # MPS virtual bond dimension (vD)
        self.wD = wD = mpo.Ws[0].shape[0]  # wD

        L0 = np.zeros((chi, wD, chi))  # (vL, wL, vL)
        R0 = np.zeros((chi, wD, chi))  # (vR, wR, vR)

        self.Lenv = [None] * self.Nsites
        self.Renv = [None] * self.Nsites
        L0[:, 0, :] = np.eye(chi)
        R0[:, wD - 1, :] = np.eye(chi)

        self.Lenv[0] = L0
        self.Renv[-1] = R0

        t_setup = time.time()
        for i in range(self.Nsites - 1, 0, -1):
            self.update_right(i, psi.As[i])
        self.wt_setup += time.time() - t_setup

    def _prepare_kernel_state(self, mpo=None, psi=None, **kwargs):
        """Prepare state before the generic TDVP kernel loop."""
        if mpo is not None or psi is not None:
            self._configure_state(
                self.mpo if mpo is None else mpo,
                self.psi if psi is None else psi,
            )
        if self.psi is None:
            raise ValueError("TDVP kernel requires a state.")
        self.verbose = getattr(self.psi, "verbose", self.verbose)

    def _safe_nsteps(self):
        if self.dt == 0.0:
            return 0
        return max(0, int(round((float(self.tmax) - float(self.t0)) / float(self.dt))))


    def dump_flags(self):
        super().dump_flags()
        print(f"Krylov dim:           {self.krylov_dim:10d}")
        print(f"Symmetric?            {self.symmetric}")
        print("")


    def _right_envs(self, mps: MPS):
        be = mps.be
        W = self.mpo.Ws
        L = mps.L

        Renv = [None] * (L + 1)
        Renv[L] = be.asarray(1.0).reshape(1, 1, 1)  # (Dr,wR,Dr)
        for i in range(L - 1, -1, -1):
            A = mps.As[i]
            Ac = be.conj(A)
            Renv[i] = mps.be.einsum("apb,cqd,xyqp,byd->axc", A, Ac, W[i], Renv[i + 1])
        return Renv


    def update_left(self, i, A):
        """Calculate left environment on the left of site `i+1`.

        Uses the LP left of site `i` and the given, left-canonical `A` on site `i`."""

        """
        A = mps.As[i]
        Ac = mps.be.conj(A)
        Wi = self.mpo.Ws[i]
        return mps.be.einsum("axb,apc,bqd,xyqp->cyd", Lenv, A, Ac, Wi)
        """

        j = (i + 1) % self.mps.L
        Li = self.Lenv[i]  # vL wL vL*
        # A legs:    vL i vR
        Ac = A.conj()  # vL* i* vR*
        Wi = self.mpo.Ws[i]  # wL wR i i*
        t0 = time.time()
        self.Lenv[j] = update_left(Wi, A, Li, Ac)
        self.wt_envs += time.time() - t0


    def update_right(self, i, B):
        """Calculate right environment on the right right of site `i-1`.

        Uses RP right of `i` and the given, right-canonical `B` on site `i`.
        """
        j = (i - 1) % self.mps.L
        Ri = self.Renv[i]  # vR* wR* vR
        # B legs:     vL i vR
        Bc = B.conj()  # vL* i* vR*
        Wi = self.mpo.Ws[i]  # wL wR i i*
        t0 = time.time()
        self.Renv[j] = update_right(Wi, B, Ri, Bc)
        self.wt_envs += time.time() - t0

    def _sweep_old(self, mps: MPS, dt_local: float):
        r"""deprecated
        """
        be = mps.be
        mps.make_mixed_canonical(center=0)
        Renv = self._right_envs(mps)
        Lenv = be.asarray(1.0).reshape(1, 1, 1)  # (Dl,wL,Dl)

        for i in range(mps.L):
            A = mps.As[i]
            W = self.mpo.Ws[i]
            R = Renv[i + 1]

            def heff_mv(X):
                # Y[a,q,c] = sum_{x,y,b,p,t} L[a,x,b] W[x,y,q,p] X[b,p,t] R[t,y,c]
                return mps.be.einsum("axb,xyqp,bpt,tyc->aqc", Lenv, W, X, R)

            A_new = lanczos_expm_multiply(heff_mv, A, be, dt=(-1j * dt_local), m=self.krylov_dim)
            mps.As[i] = A_new

            if i < mps.L - 1:
                # shift center right (QR)
                Dl, d, Dr = mps.As[i].shape
                M = be.reshape(mps.As[i], (Dl * d, Dr))
                Q, Rq = be.qr(M)
                chi = Q.shape[1]
                mps.As[i] = be.reshape(Q, (Dl, d, chi))
                mps.As[i + 1] = mps.be.einsum("ab,bpc->apc", Rq, mps.As[i + 1])

                Lenv = self._left_update(mps, Lenv, i)

    def update_energy(self, psi):
        r"""Compute the energy
        """
        t0 = time.time()
        E = np.sum(psi.expval_bonds_from_psi2(self.Hbonds))
        self.wt_energy += time.time() - t0
        return E


    def _sweep(self, psi, dt):
        r"""TDVP sweep algorithm
        """
        t_sweep = time.time()
        L = psi.L

        # 1) sweep from left to right
        theta = psi.get_psi_1site(0)
        for i in range(L - 1):
            # print(f"debug-yz: update site {i}", self.Lenv[i].shape, self.Renv[i].shape)

            theta = self.evolve_one_site(i, 0.5*dt, theta)  # forward
            Ai, theta = self.split_one_site_theta(i, theta, move_right=True)
            # here theta is zero-site between site i and i+1
            psi.As[i] = Ai  # not in right canonical form, but expect this in right-to-left sweep
            self.update_left(i, Ai)
            theta = self.evolve_zero_site(i, -0.5*dt, theta)  # backward
            j = i + 1
            Bj = self.psi.As[j]
            theta = np.tensordot(theta, Bj, axes=(1, 0))  # vL [vL'], [vL] j vR
            # here theta is one-site on site j = i + 1
        # right boundary
        i = L - 1
        theta = self.evolve_one_site(i, dt, theta)  # forward
        theta, Bi = self.split_one_site_theta(i, theta, move_right=False)
        self.psi.As[i] = Bi
        self.update_right(i, Bi)

        # 2) sweep from right to left
        for i in reversed(range(L - 1)):
            theta = self.evolve_zero_site(i, -0.5*dt, theta)  # backward
            Ai = self.psi.As[i]  # still in left-canonical A form from the above right-sweep!
            theta = np.tensordot(Ai, theta, axes=(2, 0))  # vL i [vR], [vR'] vR
            theta = self.evolve_one_site(i, 0.5*dt, theta)  # forward
            theta, Bi = self.split_one_site_theta(i, theta, move_right=False)
            self.psi.As[i] = Bi
            self.update_right(i, Bi)
        assert theta.shape == (1, 1)
        assert abs(abs(theta[0]) - 1.) < 1.e-10
        # put it back into the tensor to keep track of the phase
        self.psi.As[0] *= theta[0, 0]
        self.wt_sweep += time.time() - t_sweep

    def expm_multiply(self, H, psi0, dt):
        from scipy.sparse.linalg import expm_multiply
        from packaging import version
        if version.parse(scipy.__version__) >= version.parse('1.9.0'):
            traceH = H.trace()  # new argument introduced in scipy 1.9.0
            return expm_multiply((-1.j*dt) * H, psi0, traceA =1.j*dt*traceH)
        return expm_multiply((-1.j*dt) * H, psi0)

    def evolve_zero_site(self, i, dt, theta):
        """Evolve zero-site `theta` with Heff_0site right of site `i`."""
        t0 = time.time()
        Heff = Heff_0site(self.Lenv[i + 1], self.Renv[i])
        self.wt_Heff += time.time() - t0
        theta = np.reshape(theta, [Heff.shape[0]])
        t0 = time.time()
        theta = self.expm_multiply(Heff, theta, dt)
        self.wt_diag += time.time() - t0
        # no truncation necessary!
        return np.reshape(theta, Heff.theta_shape)

    def evolve_one_site(self, i, dt, theta):
        """Evolve one-site `theta` with Heff_1site on site i."""
        # get effective Hamiltonian
        t0 = time.time()
        Heff = Heff_1site(self.Lenv[i], self.Renv[i], self.mpo.Ws[i])
        self.wt_Heff += time.time() - t0
        theta = np.reshape(theta, [Heff.shape[0]])
        t0 = time.time()
        theta = self.expm_multiply(Heff, theta, dt)
        self.wt_diag += time.time() - t0
        # no truncation necessary!
        return np.reshape(theta, Heff.theta_shape)

    def split_one_site_theta(self, i, theta, move_right=True):
        """Split a one-site theta into `Ai, theta` (right move) or ``theta, Bi`` (left move)."""
        t0 = time.time()
        chivL, d, chivR = theta.shape
        if move_right:
            # group i to the left
            theta = np.reshape(theta, [chivL * d, chivR])
            A, S, V = scipy.linalg.svd(theta, full_matrices=False)  # vL vC, vC, vC i vR
            S /= np.linalg.norm(S)
            self.psi.Ss[i + 1] = S
            chivC = len(S)  # no truncation necessary!
            A = np.reshape(A, [chivL, d, chivC])
            theta = np.tensordot(np.diag(S), V, axes=(1, 0)) # vC [vC'], [vC] vR
            self.wt_bond += time.time() - t0
            return A, theta
        else:
            # group i to the right
            theta = np.reshape(theta, [chivL, d * chivR])
            U, S, B = scipy.linalg.svd(theta, full_matrices=False)  #  vL i vC, vC, vC vR
            S /= np.linalg.norm(S)
            self.psi.Ss[i] = S
            chivC = len(S)  # no truncation necessary!
            B = np.reshape(B, [chivC, d, chivR])
            theta = np.tensordot(U, np.diag(S), axes=(1, 0)) # vL [vC], [vC'] vC
            self.wt_bond += time.time() - t0
            return theta, B



    def step(self, mps: MPS) -> None:
        if len(self.mpo.Ws) != mps.L:
            raise ValueError("MPO length mismatch")
        if self.symmetric:
            self._sweep(mps, 0.5 * self.dt)
            # reverse sweep can be added (like earlier), but many users do L->R only for TDVP1
            self._sweep(mps, 0.5 * self.dt)
        else:
            self._sweep(mps, self.dt)

    def _measure_energy(self, compute_energy=True):
        if not compute_energy:
            return None
        return self.update_energy(self.psi)

    def _record_energy(self, energy):
        if energy is None:
            return None
        self.expectation_energies.append(energy)
        if self.conserve_energy_history and self.energies:
            stored = self.energies[0]
        else:
            stored = energy
        self.energies.append(stored)
        self.energy = stored
        return stored

    def _log_kernel_step(self, step, time_value, energy):
        if energy is None:
            return
        if step % self.print_freq == 0:
            chi_msg = ""
            if hasattr(self.psi, "get_chi"):
                try:
                    chi_msg = f" , max(chi) = {max(self.psi.get_chi())}"
                except Exception:
                    chi_msg = ""
            print(f"t = {time_value:7.3f}  E = {energy:15.8f}{chi_msg}")

    def kernel(
        self,
        mpo=None,
        psi=None,
        nsteps: Optional[int] = None,
        dt: Optional[float] = None,
        compute_energy: bool = True,
        callback=None,
        **kwargs,
    ):
        r"""Run the TDVP propagation loop.

        MPS callers can keep using ``kernel()`` after construction.  Tree
        subclasses can pass ``mpo`` and ``psi`` here and override the setup,
        stepping, and energy hooks while sharing the time-loop bookkeeping.
        """
        t_kernel = time.time()
        if dt is not None:
            self.dt = float(dt)
        if nsteps is not None:
            self.nsteps = int(nsteps)
        if "tmax" in kwargs:
            self.tmax = float(kwargs["tmax"])

        self._prepare_kernel_state(mpo=mpo, psi=psi, **kwargs)
        total_steps = int(self.nsteps) if self.nsteps is not None else self._safe_nsteps()

        self.times = [self.t0]
        self.energies = []
        self.expectation_energies = []

        energy = self._measure_energy(compute_energy=compute_energy)
        stored = self._record_energy(energy)
        self._log_kernel_step(0, self.t0, stored)

        for step in range(total_steps):
            self.step(self.psi)
            self.nsteps_done += 1
            time_value = self.t0 + (step + 1) * self.dt
            self.times.append(time_value)
            energy = self._measure_energy(compute_energy=compute_energy)
            stored = self._record_energy(energy)
            self._log_kernel_step(step + 1, time_value, stored)
            if callback is not None:
                callback(step + 1, time_value, self.psi)
        self.wt_kernel += time.time() - t_kernel
        self.post_kernel()
        return self.energy


# 2-site center + bond-center evolution term
# TODO: For a true symmetric 2-site TDVP projector-splitting integrator, we should implement a right-to-left sweep
class TDVP2(TDVP1):
    """
    2-site TDVP projector-splitting (Haegeman/Lubich/Oseledets style):
      - evolve two-site center tensor forward
      - split by SVD => left iso A_i, bond matrix C, right iso B_{i+1}
      - evolve bond center C backward (bond-center term)
      - absorb C into next tensor and continue sweep
    """
    def __init__(
        self,
        mpo: "MPO",
        psi,
        dt: float = 0.1,
        tmax: float = 10.0,
        tol=None,
        t0: float = 0.0,
        krylov_dim: int = 20,
        symmetric: bool = False,
        # tdpv2 args
        cutoff: float = 1e-12,
        use_two_site_mpo_cache: bool = True,
        **kwargs,
    ):
        # initialize base attributes (tmax, maxchi, flags, etc.)
        super().__init__(mpo, psi, dt=dt, tmax=tmax, tol=tol, t0=t0, krylov_dim=krylov_dim, symmetric=symmetric, **kwargs)

        # TDVP2-specific attributes
        self.use_two_site_mpo_cache = bool(use_two_site_mpo_cache)
        self.cutoff = cutoff
        # self.maxchi = maxchi
        # self.cutoff = float(cutoff)
        self._cache: Optional[TwoSiteMPOCache] = None

    # ---- environments ----
    def _right_envs(self, mps: MPS):
        be = mps.be
        W = self.mpo.Ws
        L = mps.L
        Renv = [None] * (L + 1)
        Renv[L] = be.asarray(1.0).reshape(1, 1, 1)
        for i in range(L - 1, -1, -1):
            A = mps.As[i]
            Ac = be.conj(A)
            Renv[i] = mps.be.einsum("apb,cqd,xyqp,byd->axc", A, Ac, W[i], Renv[i + 1])
        return Renv

    def evolve_split_two_site(self, i, dt, theta):
        """Evolve two-site `theta` with Heff2 on sites i and i + 1."""
        j = i + 1
        # get effective Hamiltonian
        t0 = time.time()
        Heff = Heff_2site(self.Lenv[i], self.Renv[j], self.mpo.Ws[i], self.mpo.Ws[j])
        self.wt_Heff += time.time() - t0
        theta = np.reshape(theta, [Heff.shape[0]])  # group legs
        t0 = time.time()
        theta = self.expm_multiply(Heff, theta, dt)
        self.wt_diag += time.time() - t0
        theta = np.reshape(theta, Heff.theta_shape)  # split legs
        # truncation necessary!
        t0 = time.time()
        Ai, S, Bj = split_truncate_theta(theta, self.maxchi, eps=self.cutoff)
        # print("Debug: updating bonds: max(S) = ", len(S))
        self.psi.Ss[j] = S
        self.wt_bond += time.time() - t0
        return Ai, S, Bj


    def _sweep(self, psi, dt):
        """Perform one two-site TDVP sweep to evolve |psi> -> exp(-i H_mpo dt) |psi>.

        This can grow the bond dimension, but is *not* stricly TDVP.
        """
        t_sweep = time.time()
        logger.debug(self, "debug-yz: using 2-site sweep")
        L = self.psi.L

        # 1) sweep from left to right
        theta = self.psi.get_psi_2site(0, 1)  # FIXME: make it compatible with TTN
        for i in range(L - 2):
            j = i + 1
            k = i + 2
            Ai, S, Bj = self.evolve_split_two_site(i, 0.5*dt, theta)  # forward
            psi.As[i] = Ai  # not in right canonical form, but expect this in right-to-left sweep
            self.update_left(i, Ai)
            theta = np.tensordot(np.diag(S), Bj, axes=(1, 0))  # vL [vL'], [vL] j vC
            # here theta is one-site on site j = i + 1
            theta = self.evolve_one_site(j, -0.5*dt, theta)  # backward
            Bk = self.psi.As[k]
            theta = np.tensordot(theta, Bk, axes=(2, 0))  # vL j [vC], [vC] k vR
            # here theta is two-site on sites j, k = i + 1, i + 2
        # right boundary
        i = L - 2
        j = L - 1
        Ai, S, Bj = self.evolve_split_two_site(i, dt, theta)  # forward
        theta = np.tensordot(Ai, np.diag(S), axes=(2, 0))  # vL i [vC], [vC'] vC
        self.psi.As[j] = Bj
        self.update_right(j, Bj)

        # 2) sweep from right to left
        for i in reversed(range(L - 2)):
            j = i + 1
            # here, theta is one-site on site j = i + 1
            theta = self.evolve_one_site(j, -0.5*dt, theta)  # backward
            Ai = self.psi.As[i]  # still in left-canonical A form from the above right-sweep!
            theta = np.tensordot(Ai, theta, axes=(2, 0))  # vL i [vR], [vR'] vR
            # here, theta is two-site on sites i, j = i, i + 1
            Ai, S, Bj = self.evolve_split_two_site(i, 0.5*dt, theta)  # forward
            self.psi.As[j] = Bj
            self.update_right(j, Bj)
            theta = np.tensordot(Ai, np.diag(S), axes=(2, 0))  # vL i vC, [vC'] vC
        self.psi.As[0] = theta
        self.wt_sweep += time.time() - t_sweep


    #

    # ---- two-site evolution matvec ----
    def _two_site_matvec(self, mps: MPS, Lenv, Renv_ip2, W2, Theta):
        # Y[a,q,s,c] = sum_{x,z,b,p,r,t} L[a,x,b] W2[x,z,q,s,p,r] Theta[b,p,r,t] R[t,z,c]
        return mps.be.einsum("axb,xzqspr,bprt,tzc->aqsc", Lenv, W2, Theta, Renv_ip2)

    # ---- split theta => A_i (left-iso), C (bond), B_{i+1} (right-iso)
    def _svd_split_theta(self, mps: MPS, Theta, Dl: int, d: int, Dr: int):
        be = mps.be
        M = be.reshape(Theta, (Dl * d, d * Dr))
        U, S, Vh = be.svd(M, full_matrices=False)

        keep = int(S.shape[0]) if self.maxchi is None else min(int(self.maxchi), int(S.shape[0]))
        s0 = float(be.to_numpy(S[0])) if int(S.shape[0]) else 0.0
        if s0 > 0:
            import numpy as np
            Snp = be.to_numpy(S)
            k2 = int(np.sum(np.abs(Snp) > self.cutoff * s0))
            keep = max(1, min(keep, k2))

        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]

        A_i = be.reshape(U, (Dl, d, keep))     # left isometric
        B_j = be.reshape(Vh, (keep, d, Dr))    # right isometric
        # bond-center matrix C = diag(S)
        C = be.asarray(0.0) * be.reshape(S, (keep, 1))  # dummy init
        C = be.reshape(S, (keep,))  # keep as vector; apply as diag via scaling
        return A_i, C, B_j

    # ---- bond-center effective matvec ----
    def _bond_center_matvec(self, mps: MPS, Lenv, Renv_ip2, W0, W1, A_i, B_j, Cmat):
        """
        Cmat shape (chi, chi) (we'll pass actual matrix).
        Build projected operators:
          Lproj[u,y,v] = sum_{a,b,x,p,q} L[a,x,b] A[a,p,u] conj(A[b,q,v]) W0[x,y,q,p]
          Rproj[t,y,u2] = sum_{c,d,z,r,s} B[t,r,c] conj(B[u2,s,d]) W1[y,z,s,r] R[c,z,d]
        Then (Heff C)[u,u2] = sum_{y,v,t} Lproj[u,y,v] C[v,t] Rproj[t,y,u2]
        """
        be = mps.be
        Ac_i = be.conj(A_i)
        Bc_j = be.conj(B_j)

        # Lproj[u,y,v]
        Lproj = mps.be.einsum("axb,apu,bqv,xyqp->uyv", Lenv, A_i, Ac_i, W0)
        # Rproj[t,y,u2]  (t is left bond of B_j, u2 is conjugate-left bond)
        Rproj = mps.be.einsum("trc,usd,yzsr,czd->tyu", B_j, Bc_j, W1, Renv_ip2)

        # apply: out[u,u2] = sum_{y,v,t} Lproj[u,y,v] C[v,t] Rproj[t,y,u2]
        out = mps.be.einsum("uyv,vt,tyw->uw", Lproj, Cmat, Rproj)  #
        # The last contraction used duplicate output labels; do it in two steps safely:
        tmp = mps.be.einsum("uyv,vt->uyt", Lproj, Cmat)     # (u,y,t)
        out = mps.be.einsum("uyt,tyw->uw", tmp, Rproj)    # (u,u2)
        return out

    def _sweep_lr(self, mps: MPS, dt_local: float):
        be = mps.be
        mps.make_mixed_canonical(center=0)
        Renv = self._right_envs(mps)
        Lenv = be.asarray(1.0).reshape(1, 1, 1)

        if self._cache is None and self.use_two_site_mpo_cache:
            self._cache = TwoSiteMPOCache(self.mpo, mps.be.einsum) # mps.contract)

        for i in range(mps.L - 1):
            A = mps.As[i]
            B = mps.As[i + 1]
            Dl, d, Dm = A.shape
            Dm2, d2, Dr = B.shape
            if d != d2 or Dm != Dm2:
                raise ValueError("MPS shape mismatch")
            # two-site center tensor Theta[b,p,r,t] with b=Dl, t=Dr
            Theta = mps.be.einsum("apb,brc->aprc", A, B)  # (Dl, d, d, Dr)

            # build W2 block
            if self._cache is not None:
                W2 = self._cache.get(i)
            else:
                W0 = self.mpo.Ws[i]
                W1 = self.mpo.Ws[i + 1]
                W2 = mps.be.einsum("xyqp,yzsr->xzqspr", W0, W1)

            R_ip2 = Renv[i + 2]  # env right of site i+1

            def heff2_mv(X):
                return self._two_site_matvec(mps, Lenv, R_ip2, W2, X)

            # evolve Theta forward: exp(-i dt H_eff) Theta
            Theta_new = lanczos_expm_multiply(heff2_mv, Theta, be, dt=(-1j * dt_local), m=self.krylov_dim)

            # split
            A_i, Svec, B_j = self._svd_split_theta(mps, Theta_new, Dl, d, Dr)

            # build C matrix (diag(S))
            chi = int(Svec.shape[0])
            import numpy as np
            # create diag in backend safely
            Cmat = be.asarray(np.diag(be.to_numpy(Svec).astype(np.complex128)))

            # --- bond-center evolution (backward step) ---
            W0 = self.mpo.Ws[i]
            W1 = self.mpo.Ws[i + 1]

            def heff0_mv(X):
                return self._bond_center_matvec(mps, Lenv, R_ip2, W0, W1, A_i, B_j, X)

            # backward in time for center matrix: exp(+i dt H_eff0) C
            Cmat = lanczos_expm_multiply(heff0_mv, Cmat, be, dt=(+1j * dt_local), m=self.krylov_dim)

            # absorb center into right tensor: B <- C * B
            B_j = mps.be.einsum("ab,bpc->apc", Cmat, B_j)

            # write back
            mps.As[i] = A_i
            mps.As[i + 1] = B_j

            # update Lenv for next i (using updated A_i at site i)
            Lenv = self._left_update(mps, Lenv, i)

        # ensure canonical at end
        mps.make_mixed_canonical(center=mps.L - 1)

    """
    def step(self, mps: MPS) -> None:
        if len(self.mpo) != mps.L:
            raise ValueError("MPO length mismatch")
        if self.symmetric:
            self._sweep_lr(mps, 0.5 * self.dt)
            # symmetric version should include reverse sweep (R->L).
            # For brevity, add a second LR sweep; in practice implement RL too.
            self._sweep_lr(mps, 0.5 * self.dt)
        else:
            self._sweep_lr(mps, self.dt)
    """

class Belief(PropagatorBase):
    r"""Class implementing Blief Propagation [1]
    (will be moved into refs.bib)

    [1] xx
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
