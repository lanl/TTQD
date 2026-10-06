
r"""
Fermionic operators and second-quantized Hamiltonian helpers
------------------------------------------------------------

This module owns local fermionic matrices and lightweight containers for
second-quantized Hamiltonians.  Exact MPO construction from one- and two-body
integrals is delegated lazily to :mod:`ttqd.operator.mpo_tools`.
"""


from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray[np.complex128]

__all__ = [
    "FermionicHamiltonian",
    "local_fermion_ops",
    "spinful_fermion_ops",
    "hubbard_local_operators",
    "fcrea",
    "fanih",
    "fnumb",
    "fparity",
    "diagm",
    "unitvec",
    "unitmat",
    "unitcol",
    "unitrow",
]


def fcrea(d: int = 2) -> NDArray:
    r"""Return the single-mode creation operator :math:`c^\dagger`."""
    if d != 2:
        raise ValueError("Single-mode fermionic operators require d=2")
    return np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128)


def fanih(d: int = 2) -> NDArray:
    r"""Return the single-mode annihilation operator :math:`c`."""
    return fcrea(d).conj().T


def fnumb(d: int = 2) -> NDArray:
    r"""Return the single-mode number operator :math:`n=c^\dagger c`."""
    return fcrea(d) @ fanih(d)


def fparity(d: int = 2) -> NDArray:
    r"""Return the single-mode parity operator :math:`(-1)^n`."""
    return np.eye(d, dtype=np.complex128) - 2.0 * fnumb(d)


def local_fermion_ops(dtype=np.complex128) -> dict[str, NDArray]:
    """Return ``I, c, cd, n, F, 0`` for one spin orbital."""
    cd = fcrea().astype(dtype, copy=False)
    c = fanih().astype(dtype, copy=False)
    n = fnumb().astype(dtype, copy=False)
    I = np.eye(2, dtype=dtype)
    F = fparity().astype(dtype, copy=False)
    z = np.zeros((2, 2), dtype=dtype)
    return {"I": I, "c": c, "cd": cd, "n": n, "F": F, "0": z}


def diagm(d: int, k: int, vals) -> NDArray:
    """
    Create a dense matrix with ``vals`` placed on diagonal offset ``k``.

    ``k=0`` is the main diagonal, ``k>0`` a superdiagonal, and ``k<0`` a
    subdiagonal.
    """
    M = np.zeros((d, d), dtype=np.complex128)
    vals = np.asarray(vals, dtype=np.complex128)

    if k >= 0:
        for i in range(min(d - k, len(vals))):
            M[i, i + k] = vals[i]
    else:
        kk = -k
        for i in range(min(d - kk, len(vals))):
            M[i + kk, i] = vals[i]

    return M


def spinful_fermion_ops(dtype=np.complex128) -> dict[str, NDArray]:
    r"""
    Return local operators for one spinful fermionic site.

    The basis is

    .. math::

        |0\rangle,\quad |\uparrow\rangle,\quad |\downarrow\rangle,\quad
        |\uparrow\downarrow\rangle.

    The ``*Par`` operators include a right local parity factor and are used by
    the Hubbard finite-state MPO construction.
    """
    I = np.identity(4, dtype=dtype)
    Z = np.zeros((4, 4), dtype=dtype)

    aDagUp = np.array(
        [[0, 0, 0, 0],
         [1, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 1, 0]],
        dtype=dtype,
    )
    aDagUpPar = np.array(
        [[0, 0, 0, 0],
         [1, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, -1, 0]],
        dtype=dtype,
    )
    aDagDn = np.array(
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 0],
         [0, -1, 0, 0]],
        dtype=dtype,
    )
    aDagDnPar = np.array(
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 0],
         [0, 1, 0, 0]],
        dtype=dtype,
    )
    aUp = np.array(
        [[0, 1, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 0, 0]],
        dtype=dtype,
    )
    aUpPar = np.array(
        [[0, -1, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 0, 0]],
        dtype=dtype,
    )
    aDn = np.array(
        [[0, 0, 1, 0],
         [0, 0, 0, -1],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        dtype=dtype,
    )
    aDnPar = np.array(
        [[0, 0, -1, 0],
         [0, 0, 0, -1],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        dtype=dtype,
    )

    nUp = np.diag([0, 1, 0, 1]).astype(dtype)
    nDn = np.diag([0, 0, 1, 1]).astype(dtype)
    nTot = nUp + nDn
    nUpnDn = np.diag([0, 0, 0, 1]).astype(dtype)
    parity = np.diag([1, -1, -1, 1]).astype(dtype)

    return {
        "I": I,
        "Z": Z,
        "aDagUp": aDagUp,
        "aDagUpPar": aDagUpPar,
        "aDagDn": aDagDn,
        "aDagDnPar": aDagDnPar,
        "aUp": aUp,
        "aUpPar": aUpPar,
        "aDn": aDn,
        "aDnPar": aDnPar,
        "nUp": nUp,
        "nDn": nDn,
        "nTot": nTot,
        "nUpnDn": nUpnDn,
        "parity": parity,
    }


hubbard_local_operators = spinful_fermion_ops


@dataclass
class FermionicHamiltonian:
    r"""
    One- plus two-body spin-orbital Hamiltonian.

    .. math::

        H =
        C_0 I
        + \sum_{pq} h_{pq} a^\dagger_p a_q
        + {1 \over 2}\sum_{pqrs} h_{pqrs}
          a^\dagger_p a^\dagger_q a_r a_s.
    """

    h1: NDArray
    h2: NDArray
    constant: complex = 0.0

    def __post_init__(self):
        self.h1 = np.asarray(self.h1, dtype=np.complex128)
        self.h2 = np.asarray(self.h2, dtype=np.complex128)
        if self.h1.ndim != 2 or self.h1.shape[0] != self.h1.shape[1]:
            raise ValueError("h1 must have shape (K, K)")
        K = self.h1.shape[0]
        if self.h2.shape != (K, K, K, K):
            raise ValueError("h2 must have shape (K, K, K, K)")

    @property
    def n_spin_orbitals(self) -> int:
        return int(self.h1.shape[0])

    def to_mpo(self, compact=True, tol=1.0e-14):
        """
        Convert to an exact MPO tensor list.

        If ``compact=True`` this returns ``(Ws, bond_labels)`` from the compact
        fork/merge builder.  Otherwise it returns only ``Ws`` from the prefix
        tree builder.
        """
        if compact:
            from ttqd.operator.mpo_tools import build_compact_second_quantized_mpo

            return build_compact_second_quantized_mpo(
                self.h1,
                self.h2,
                constant=self.constant,
                tol=tol,
                dtype=np.complex128,
            )

        from ttqd.operator.mpo_tools import second_quantized_to_mpo

        if abs(self.constant) > tol:
            raise ValueError(
                "constant shifts are supported by the compact MPO builder; "
                "use compact=True or absorb the constant separately"
            )
        return second_quantized_to_mpo(
            self.h1,
            self.h2,
            tol=tol,
            dtype=np.complex128,
        )


def unitvec(n: int, d: int) -> NDArray:
    """Return a 1-indexed unit vector."""
    if not (1 <= n <= d):
        raise ValueError("n must satisfy 1 <= n <= d")
    v = np.zeros(d, dtype=np.complex128)
    v[n - 1] = 1.0
    return v


def unitmat(d1: int, d2: int | None = None) -> NDArray:
    """Return an identity/rectangular identity matrix."""
    if d2 is None:
        d2 = d1
    return np.eye(d1, d2, dtype=np.complex128)


def unitcol(n: int, d: int) -> NDArray:
    """Return a 1-indexed unit column vector."""
    return unitvec(n, d).reshape(d, 1)


def unitrow(n: int, d: int) -> NDArray:
    """Return a 1-indexed unit row vector."""
    return unitvec(n, d).reshape(1, d)
