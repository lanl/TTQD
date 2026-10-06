r"""
Spin operators and spin-chain Hamiltonian builders
--------------------------------------------------

This module owns local spin matrices and small spin-chain Hamiltonian builders.
The functions return either local dense operators or chain-local objects that
can be used directly by TEBD/TDVP examples, or wrapped into MPO/TTNO builders.
"""

from __future__ import annotations

import numpy as np

from ttqd.operator._operatorbase import BaseOperator


__all__ = [
    "SpinOp",
    "spin_matrices",
    "paulis",
    "build_h2_list_xxz",
    "build_mpo_xxz",
]


class SpinOp(BaseOperator):
    r"""
    Local spin-:math:`S` operators.

    The basis is ordered by magnetic quantum number
    :math:`m=-S,-S+1,\ldots,S`.  The attributes ``Sx``, ``Sy``, ``Sz``,
    ``Sp`` and ``Sm`` are dense matrices in this basis.
    """

    def __init__(self, S=0.5, dtype=np.complex128, **kwargs):
        super().__init__(**kwargs)
        self.label = kwargs.get("label", None)
        self.S = float(S)
        self.dtype = dtype

        two_s = 2 * self.S
        if self.S <= 0:
            raise ValueError("S must be positive")
        if not np.isclose(two_s, np.rint(two_s)):
            raise ValueError("S must be half-integer or integer")

        d = int(np.rint(two_s)) + 1
        self.d = d

        mvals = -self.S + np.arange(d)
        Sz = np.diag(mvals).astype(dtype)
        Sp = np.zeros((d, d), dtype=dtype)
        for n in range(d - 1):
            m = n - self.S
            Sp[n + 1, n] = np.sqrt(self.S * (self.S + 1) - m * (m + 1))
        Sm = Sp.conj().T

        self.I = np.eye(d, dtype=dtype)
        self.Sx = 0.5 * (Sp + Sm)
        self.Sy = -0.5j * (Sp - Sm)
        self.Sz = Sz
        self.Sp = Sp
        self.Sm = Sm

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return the local spin operators as a dictionary."""
        return {
            "I": self.I,
            "Sx": self.Sx,
            "Sy": self.Sy,
            "Sz": self.Sz,
            "Sp": self.Sp,
            "Sm": self.Sm,
        }

    def __repr__(self):
        return f"Spin-{self.S:g} operator"


def spin_matrices(S=0.5, dtype=np.complex128) -> dict[str, np.ndarray]:
    """Return ``I, Sx, Sy, Sz, Sp, Sm`` for spin ``S``."""
    return SpinOp(S=S, dtype=dtype).as_dict()


def paulis(dtype=complex):
    """Return the Pauli matrices ``I, X, Y, Z``."""
    I = np.eye(2, dtype=dtype)
    X = np.array([[0, 1], [1, 0]], dtype=dtype)
    Y = np.array([[0, -1j], [1j, 0]], dtype=dtype)
    Z = np.array([[1, 0], [0, -1]], dtype=dtype)
    return I, X, Y, Z


def build_h2_list_xxz(L, Jx=1.0, Jy=1.0, Jz=1.0, hz=0.0, use_spin_ops=False):
    r"""
    Return two-site bond Hamiltonians for the open-boundary XXZ chain.

    The bond list represents

    .. math::

        H =
        \sum_{i=0}^{L-2}
        \left(J_x X_i X_{i+1} + J_y Y_i Y_{i+1} + J_z Z_i Z_{i+1}\right)
        + h_z \sum_{i=0}^{L-1} Z_i.

    Returns a list of length ``L - 1``. Each item has shape ``(4, 4)`` and
    stores a two-site operator on sites ``i, i+1``. The onsite field is
    distributed over neighboring bonds, with boundary corrections so each site
    receives exactly one copy of ``hz * Z``. If ``use_spin_ops=True``,
    ``X, Y, Z`` are replaced by ``Sx, Sy, Sz = 0.5 * Pauli``.
    """
    if L < 1:
        raise ValueError("L must be at least 1")

    I, X, Y, Z = paulis()
    if use_spin_ops:
        X, Y, Z = 0.5 * X, 0.5 * Y, 0.5 * Z

    h2_list = []
    for i in range(L - 1):
        h2 = (
            Jx * np.kron(X, X)
            + Jy * np.kron(Y, Y)
            + Jz * np.kron(Z, Z)
        )

        if hz != 0.0:
            h2 += (hz / 2.0) * (np.kron(Z, I) + np.kron(I, Z))
            if i == 0:
                h2 += (hz / 2.0) * np.kron(Z, I)
            if i == L - 2:
                h2 += (hz / 2.0) * np.kron(I, Z)

        h2_list.append(h2)

    return h2_list


def build_mpo_xxz(L, Jx=1.0, Jy=1.0, Jz=1.0, hz=0.0, use_spin_ops=False):
    r"""
    Build an open-boundary MPO tensor list for the XXZ chain.

    .. math::

        H =
        \sum_{i=0}^{L-2}
        \left(J_x X_i X_{i+1} + J_y Y_i Y_{i+1} + J_z Z_i Z_{i+1}\right)
        + h_z \sum_{i=0}^{L-1} Z_i.

    Returns ``Ws``, a list of length ``L`` where each item has shape
    ``(w_left, w_right, 2, 2)``. If ``use_spin_ops=True``, ``X, Y, Z`` are
    replaced by ``Sx, Sy, Sz = 0.5 * Pauli``.
    """
    if L < 1:
        raise ValueError("L must be at least 1")

    I, X, Y, Z = paulis()
    if use_spin_ops:
        X, Y, Z = 0.5 * X, 0.5 * Y, 0.5 * Z

    d = 2
    if L == 1:
        W = np.zeros((1, 1, d, d), dtype=complex)
        W[0, 0] = hz * Z
        return [W]

    w = 5
    Wbulk = np.zeros((w, w, d, d), dtype=complex)

    # Row/col meaning:
    # virtual left index = row, virtual right index = col
    #
    # [ I    0    0    0    0 ]
    # [ X    0    0    0    0 ]
    # [ Y    0    0    0    0 ]
    # [ Z    0    0    0    0 ]
    # [ hzZ  JxX  JyY  JzZ   I ]
    Wbulk[0, 0] = I
    Wbulk[1, 0] = X
    Wbulk[2, 0] = Y
    Wbulk[3, 0] = Z
    Wbulk[4, 0] = hz * Z
    Wbulk[4, 1] = Jx * X
    Wbulk[4, 2] = Jy * Y
    Wbulk[4, 3] = Jz * Z
    Wbulk[4, 4] = I

    W0 = Wbulk[4:5, :, :, :].copy()

    WL = np.zeros((w, 1, d, d), dtype=complex)
    WL[0, 0] = I
    WL[1, 0] = X
    WL[2, 0] = Y
    WL[3, 0] = Z
    WL[4, 0] = hz * Z

    return [W0] + [Wbulk.copy() for _ in range(L - 2)] + [WL]


if __name__ == "__main__":
    spin = SpinOp(S=0.5)
    print("Sx = ", spin.Sx)
    print("Sy = ", spin.Sy)
    print("Sz = ", spin.Sz)
    print("Sp = ", spin.Sp)
    print("Sm = ", spin.Sm)

    S = 2.5
    Jspin = SpinOp(S=S)
    print("\n", Jspin)
    print(f"Jx(S) = \n{Jspin.Sx}")
    print(f"Jy(S) = \n{Jspin.Sy}")
    print(f"Jz(S) = \n{Jspin.Sz}")
    print(f"Jp(S) = \n{Jspin.Sp}")
