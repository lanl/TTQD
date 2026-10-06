r"""
Bosonic operators and truncated oscillator Hamiltonian builders
---------------------------------------------------------------

The local Hilbert space is a truncated Fock basis
:math:`|0\rangle,\ldots,|d-1\rangle`.  These helpers provide dense local
matrices and simple chain/inter-type two-site Hamiltonian blocks that can be
fed into TEBD-like routines or used as ingredients for later MPO/TTNO builders.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ttqd.operator._operatorbase import BaseOperator


NDArray = npt.NDArray[np.complex128]

__all__ = [
    "BosonOp",
    "boson_local_operators",
    "bcrea",
    "banih",
    "bnumb",
    "disp",
    "mome",
    "harmonic_oscillator_hamiltonian",
    "build_boson_chain_h2_list",
    "two_site_interaction",
    "build_spin_boson_h2",
    "build_fermion_boson_h2",
]


class BosonOp(BaseOperator):
    r"""
    Local bosonic operator in a truncated Fock space.

    Parameters
    ----------
    label : str
        One of ``"b"``, ``"bdag"``, ``"b+"``, ``"n"``, ``"I"``, ``"x"``,
        or ``"p"``.
    site : int
        Chain site label carried for bookkeeping.
    dim : int
        Local Fock-space truncation.
    """

    def __init__(self, label, site=0, dim: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.label = label
        self.site = site
        self.dim = int(dim)
        self.coeff = kwargs.get("coeff", 1.0)
        if self.dim < 1:
            raise ValueError("dim must be at least 1")

    @property
    def matrix(self) -> NDArray:
        """Dense matrix representation."""
        if self._mat is None:
            ops = boson_local_operators(self.dim)
            key = {
                "b": "b",
                "a": "b",
                "bdag": "bdag",
                "b+": "bdag",
                "adag": "bdag",
                "n": "n",
                "I": "I",
                "x": "x",
                "p": "p",
            }.get(self.label)
            if key is None:
                raise ValueError(f"Unknown boson operator label {self.label!r}")
            self._mat = self.coeff * ops[key]
        return self._mat

    def _matrix(self):
        """Compatibility alias for older code."""
        return self.matrix

    def __repr__(self):
        return f"{self.label}_{self.site}"


# Backward-compatible spelling used in older notes/scripts.
BosonObj = BosonOp


def bcrea(d: int) -> NDArray:
    r"""Return :math:`b^\dagger` in a ``d``-level truncated Fock basis."""
    if d < 1:
        raise ValueError("d must be at least 1")
    m = np.zeros((d, d), dtype=np.complex128)
    vals = np.sqrt(np.arange(1, d, dtype=float))
    m[1:, :-1] = np.diag(vals)
    return m


def banih(d: int) -> NDArray:
    r"""Return :math:`b` in a ``d``-level truncated Fock basis."""
    return bcrea(d).conj().T


def bnumb(d: int) -> NDArray:
    r"""Return :math:`n=b^\dagger b` in a ``d``-level truncated Fock basis."""
    return bcrea(d) @ banih(d)


def disp(d: int, wvib: float | None = None, m: float | None = None) -> NDArray:
    r"""
    Return the displacement coordinate.

    With no mass/frequency arguments this is

    .. math::

        x = {1 \over \sqrt{2}}(b^\dagger + b).

    If ``wvib`` and ``m`` are supplied, the mass/frequency-scaled coordinate is
    returned.
    """
    if (wvib is None) != (m is None):
        raise ValueError("Provide both wvib and m, or neither")

    if wvib is None:
        pref = 1.0 / np.sqrt(2.0)
    else:
        pref = 1.0 / (2.0 * np.sqrt(m * wvib / 2.0))

    return pref * (bcrea(d) + banih(d))


def mome(d: int) -> NDArray:
    r"""Return :math:`p={i\over\sqrt{2}}(b^\dagger-b)`."""
    return (1.0 / np.sqrt(2.0)) * (1j * (bcrea(d) - banih(d)))


def boson_local_operators(d: int, dtype=np.complex128) -> dict[str, NDArray]:
    """Return local dense boson operators ``I, b, bdag, n, x, p``."""
    ops = {
        "I": np.eye(d, dtype=dtype),
        "b": banih(d).astype(dtype, copy=False),
        "bdag": bcrea(d).astype(dtype, copy=False),
        "n": bnumb(d).astype(dtype, copy=False),
        "x": disp(d).astype(dtype, copy=False),
        "p": mome(d).astype(dtype, copy=False),
    }
    return ops


def harmonic_oscillator_hamiltonian(d: int, omega=1.0, zero_point=False) -> NDArray:
    r"""Return :math:`\omega n` or :math:`\omega(n + 1/2)`."""
    H = omega * bnumb(d)
    if zero_point:
        H = H + 0.5 * omega * np.eye(d, dtype=np.complex128)
    return H


def _distributed_onsite(onsite, site, L):
    """Return onsite contribution assigned to a nearest-neighbor bond."""
    if L <= 1:
        return onsite
    if site == 0 or site == L - 1:
        return onsite
    return 0.5 * onsite


def build_boson_chain_h2_list(
    L: int,
    d: int,
    omega=1.0,
    coupling=0.0,
    coupling_op: str = "x",
    zero_point=False,
) -> list[NDArray]:
    r"""
    Return nearest-neighbor two-site blocks for a truncated boson chain.

    The represented open-chain Hamiltonian is

    .. math::

        H = \sum_i \omega b_i^\dagger b_i
          + g \sum_i O_i O_{i+1},

    where ``O`` is selected by ``coupling_op`` and ``g`` is ``coupling``.
    Onsite terms are distributed over neighboring bonds with boundary
    corrections, mirroring the spin-chain helper.
    """
    if L < 1:
        raise ValueError("L must be at least 1")
    if d < 1:
        raise ValueError("d must be at least 1")
    if L == 1:
        return []

    ops = boson_local_operators(d)
    if coupling_op not in ops:
        raise ValueError(f"Unknown coupling_op {coupling_op!r}")

    I = ops["I"]
    O = ops[coupling_op]
    onsite = harmonic_oscillator_hamiltonian(d, omega=omega, zero_point=zero_point)

    h2_list = []
    for i in range(L - 1):
        h_left = _distributed_onsite(onsite, i, L)
        h_right = _distributed_onsite(onsite, i + 1, L)
        h2 = np.kron(h_left, I) + np.kron(I, h_right)
        if coupling != 0.0:
            h2 = h2 + coupling * np.kron(O, O)
        h2_list.append(h2)
    return h2_list


def two_site_interaction(op_left, op_right, coupling=1.0) -> NDArray:
    """Return ``coupling * kron(op_left, op_right)``."""
    return coupling * np.kron(
        np.asarray(op_left, dtype=np.complex128),
        np.asarray(op_right, dtype=np.complex128),
    )


def build_spin_boson_h2(
    d_boson: int,
    coupling=1.0,
    spin_op: str = "Sz",
    boson_op: str = "x",
    S=0.5,
) -> NDArray:
    """Return a two-site spin-boson coupling block."""
    from ttqd.operator.spinoperator import spin_matrices

    sops = spin_matrices(S=S)
    bops = boson_local_operators(d_boson)
    if spin_op not in sops:
        raise ValueError(f"Unknown spin_op {spin_op!r}")
    if boson_op not in bops:
        raise ValueError(f"Unknown boson_op {boson_op!r}")
    return two_site_interaction(sops[spin_op], bops[boson_op], coupling=coupling)


def build_fermion_boson_h2(
    d_boson: int,
    coupling=1.0,
    fermion_op: str = "n",
    boson_op: str = "x",
) -> NDArray:
    """Return a two-site fermion-boson coupling block."""
    from ttqd.operator.fermioperator import local_fermion_ops

    fops = local_fermion_ops()
    bops = boson_local_operators(d_boson)
    if fermion_op not in fops:
        raise ValueError(f"Unknown fermion_op {fermion_op!r}")
    if boson_op not in bops:
        raise ValueError(f"Unknown boson_op {boson_op!r}")
    return two_site_interaction(fops[fermion_op], bops[boson_op], coupling=coupling)
