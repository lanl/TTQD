r"""
General matrix-product-operator (MPO) constructor for vibronic/PCET models
==========================================================================

This module generalizes a hard-coded HBQ/HBT proton-transfer MPO into two layers:

1. A generic operator-string Hamiltonian-to-MPO compiler.
2. A convenience builder for :math:`N_e`-state vibronic/PCET Hamiltonians
   containing proton, charge-transfer/reaction-coordinate, and skeletal modes (a subset of vib modes).

Conventions
-----------
The tensor ordering is the same as in the original
``protontransfermpo.py``.  At site :math:`i`,

.. math::

   W^{[i]} \in
   \mathbb{C}^{D_{i-1}\times D_i\times d_i\times d_i},

so in NumPy

``W[i].shape == (D_left, D_right, d_i, d_i)``.

The first and last tensors are boundary-shaped.  The total Hilbert space is

.. math::

   \mathcal{H}
   = \mathcal{H}_{\mathrm{el}}
     \otimes
     \bigotimes_{\mu=1}^{N_v}\mathcal{H}_{\mu},

where the electronic site has local dimension :math:`N_e` and mode
:math:`\mu` has local dimension :math:`d_\mu`.  If the explicit bosonic
coordinates are partitioned into proton, charge-transfer, and skeletal modes,
then

.. math::

   N_v = N_H + N_{CT} + N_s,

and the full dense Hilbert-space dimension is

.. math::

   \dim\mathcal{H}
   = N_e \prod_{\mu=1}^{N_v} d_\mu.

All energies, frequencies, and coupling constants must be supplied in one
consistent energy unit.  The examples use eV and set :math:`\hbar=1`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray

CM1_PER_EV = 8065.544005

# -----------------------------------------------------------------------------
# Local operators
# -----------------------------------------------------------------------------

def boson_ops(d: int) -> Dict[str, Array]:
    r"""Return standard bosonic operators in a truncated Fock basis.

    The local basis is :math:`\{|0\rangle,\ldots,|d-1\rangle\}`.  The ladder and
    coordinate operators are

    .. math::

       \begin{aligned}
       \hat n &= \hat a^\dagger \hat a, \\
       \hat q &= \frac{\hat a+\hat a^\dagger}{\sqrt{2}}, \\
       \hat p &= \frac{i(\hat a^\dagger-\hat a)}{\sqrt{2}}, \\
       \hat x &= \hat a+\hat a^\dagger = \sqrt{2}\,\hat q.
       \end{aligned}

    For a dimensionless harmonic oscillator with :math:`\hbar=1`,

    .. math::

       \hat H_{\mathrm{ho}}
       = \frac{\omega}{2}\left(\hat p^2+\hat q^2\right)
       = \omega\left(\hat n+\frac{1}{2}\right),

    up to the usual finite-basis truncation at the highest Fock state.

    Parameters
    ----------
    d : int
        Local Fock-space dimension.

    Returns
    -------
    dict[str, numpy.ndarray]
        Identity, ladder, number, coordinate, momentum, and squared operators."""
    if d < 1:
        raise ValueError("Bosonic local dimension d must be >= 1.")

    I = np.eye(d, dtype=np.complex128)
    a = np.zeros((d, d), dtype=np.complex128)
    for n in range(1, d):
        a[n - 1, n] = np.sqrt(n)
    adag = a.conj().T
    n_op = adag @ a
    q = (a + adag) / np.sqrt(2.0)
    p = 1j * (adag - a) / np.sqrt(2.0)
    x = a + adag
    return {
        "I": I,
        "a": a,
        "adag": adag,
        "n": n_op,
        "q": q,
        "p": p,
        "x": x,
        "q2": q @ q,
        "p2": p @ p,
        "x2": x @ x,
    }


def electronic_projector(Ne: int, state: int) -> Array:
    r"""Return the electronic projector :math:`|\mathrm{state}\rangle\langle\mathrm{state}|` in an :math:`N_e`-state basis."""
    if not 0 <= state < Ne:
        raise IndexError(f"Electronic state {state} outside [0, {Ne}).")
    out = np.zeros((Ne, Ne), dtype=np.complex128)
    out[state, state] = 1.0
    return out


def electronic_transition(Ne: int, i: int, j: int) -> Array:
    r"""Return the electronic transition operator :math:`|i\rangle\langle j|`."""
    if not (0 <= i < Ne and 0 <= j < Ne):
        raise IndexError("Electronic transition index out of range.")
    out = np.zeros((Ne, Ne), dtype=np.complex128)
    out[i, j] = 1.0
    return out


def hermitian_electronic_coupling(Ne: int, i: int, j: int, value: complex = 1.0) -> Array:
    r"""Return a Hermitian pair coupling,

    .. math::

       v|i\rangle\langle j| + v^*|j\rangle\langle i|."""
    return value * electronic_transition(Ne, i, j) + np.conjugate(value) * electronic_transition(Ne, j, i)


# -----------------------------------------------------------------------------
# Generic operator-string Hamiltonian
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ProductTerm:
    r"""One operator-product contribution to the Hamiltonian.

    A term represents

    .. math::

       \hat H_r
       = c_r \bigotimes_{i=0}^{L-1}\hat O_i^{(r)},

    where ``ops`` stores only the nonidentity local operators.  Identity operators
    are inserted automatically on all unspecified sites.

    Examples
    --------
    An onsite electronic term :math:`\hat H_{\mathrm{el}}` on site 0 is

    ``ProductTerm(1.0, {0: H_el}, "H_el")``.

    An electron-proton term

    .. math::

       \hat G_{\mathrm{el}}\otimes\hat q_H

    on sites 0 and 1 is represented by

    ``ProductTerm(1.0, {0: G_el, 1: q_H}, "el-proton")``.

    A non-nearest-neighbor proton-skeletal term

    .. math::

       \kappa\,\hat q_H\hat q_s

    on sites 1 and 4 is represented by

    ``ProductTerm(kappa, {1: q_H, 4: q_s}, "H-skeletal")``."""

    coeff: complex
    ops: Mapping[int, Array]
    label: str = ""


@dataclass(frozen=True)
class BosonMode:
    r"""Definition of one bosonic coordinate/site.

    Parameters
    ----------
    name : str
        Unique mode name.
    dim : int
        Local Fock-space truncation :math:`d_\mu`.
    omega : float
        Harmonic frequency :math:`\omega_\mu` in the same energy units as all
        other Hamiltonian parameters.
    kind : str
        Informational tag such as ``"proton"``, ``"ct"``, ``"skeletal"``, or
        ``"bath"``.
    include_zpe : bool
        If ``True``, include the zero-point contribution
        :math:`\omega_\mu/2` in the local harmonic Hamiltonian.
    anharmonic_q : mapping[int, complex]
        Optional polynomial coefficients defining local terms

        .. math::

           \sum_{m} c_m\hat q_\mu^m.

        This is useful for reduced reaction-path models and cubic/quartic
        corrections.
    onsite_extra : numpy.ndarray, optional
        Arbitrary additional local matrix acting on this mode."""

    name: str
    dim: int
    omega: float
    kind: str = "vibronic"
    include_zpe: bool = True
    anharmonic_q: Mapping[int, complex] = field(default_factory=dict)
    onsite_extra: Optional[Array] = None


@dataclass(frozen=True)
class ModeModeCoupling:
    r"""Two-mode product coupling.

    The represented term is

    .. math::

       \hat H_{\mu\nu}
       = \kappa_{\mu\nu}\,
         \hat O_\mu\hat O_\nu.

    ``op_left`` and ``op_right`` name local operators returned by
    :func:`boson_ops`, for example ``"q"``, ``"p"``, ``"x"``, ``"n"``,
    ``"a"``, or ``"adag"``."""

    mode_left: str
    mode_right: str
    coeff: complex
    op_left: str = "q"
    op_right: str = "q"
    label: str = "mode-mode"


@dataclass(frozen=True)
class ElectronicModeModeCoupling:
    r"""Electronic-state-dependent two-mode vibronic coupling.

    The represented three-site term is

    .. math::

       \hat H_{\mathrm{el},\mu\nu}
       = c\,\hat G_{\mathrm{el}}\,
         \hat O_\mu\hat O_\nu.

    This supports state-dependent Hessian/Duschinsky-like couplings and other
    proton-CT-skeletal interactions whose strength depends on the electronic
    state."""

    electronic_operator: Array
    mode_left: str
    mode_right: str
    coeff: complex = 1.0
    op_left: str = "q"
    op_right: str = "q"
    label: str = "el-mode-mode"


@dataclass
class PCETModel:
    r"""General few-electronic-state vibronic/PCET Hamiltonian specification.

    The electronic Hamiltonian :math:`\hat H_{\mathrm{el}}` is an arbitrary
    Hermitian :math:`N_e\times N_e` matrix, and each bosonic coordinate is a
    separate MPO site.  With :math:`N_v` explicit modes, the Hilbert space is

    .. math::

       \mathcal{H}
       = \mathbb{C}^{N_e}
         \otimes
         \bigotimes_{\mu=1}^{N_v}\mathbb{C}^{d_\mu}.

    Displaced harmonic surfaces
    ---------------------------
    ``displacements`` is an optional array with shape :math:`(N_e,N_v)`.  For
    mode :math:`\mu` with frequency :math:`\omega_\mu` and dimensionless
    coordinate :math:`q_\mu`, a state-dependent displacement
    :math:`d_{\alpha\mu}` defines

    .. math::

       \hat H_\mu^{(\alpha)}
       = \frac{\omega_\mu}{2}
         \left[
           \hat p_\mu^2
           + \left(\hat q_\mu-d_{\alpha\mu}\right)^2
         \right].

    Separating the common harmonic oscillator gives the state-dependent part

    .. math::

       \hat H_{\mathrm{disp}}
       = \sum_{\alpha=1}^{N_e}\sum_{\mu=1}^{N_v}
         \hat P_\alpha
         \left[
           -\omega_\mu d_{\alpha\mu}\hat q_\mu
           + \frac{\omega_\mu}{2}d_{\alpha\mu}^2
         \right],

    with

    .. math::

       \hat P_\alpha = |\alpha\rangle\langle\alpha|.

    Electronic-vibrational couplings
    --------------------------------
    ``linear_el_vib[mode]`` adds an arbitrary electronic matrix
    :math:`\hat G_\mu` coupled linearly to coordinate :math:`\hat q_\mu`:

    .. math::

       \hat H_{\mathrm{lin}}
       = \sum_\mu \hat G_\mu\hat q_\mu.

    Diagonal elements of :math:`\hat G_\mu` are state-specific forces, whereas
    off-diagonal elements generate nonadiabatic/vibronic couplings.

    ``quadratic_el_vib[mode]`` analogously contributes

    .. math::

       \hat H_{\mathrm{quad}}
       = \sum_\mu \hat K_\mu\hat q_\mu^2.

    Together with ``electronic_mode_mode``, these terms can represent quadratic
    vibronic coupling expansions and electronic-state-dependent mode mixing."""

    electronic_hamiltonian: Array
    modes: Sequence[BosonMode]
    electronic_names: Optional[Sequence[str]] = None
    displacements: Optional[Array] = None
    linear_el_vib: Mapping[str, Array] = field(default_factory=dict)
    quadratic_el_vib: Mapping[str, Array] = field(default_factory=dict)
    mode_mode: Sequence[ModeModeCoupling] = field(default_factory=list)
    electronic_mode_mode: Sequence[ElectronicModeModeCoupling] = field(default_factory=list)
    extra_terms: Sequence[ProductTerm] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Generic MPO compiler
# -----------------------------------------------------------------------------

def _validate_local_operator(op: Array, d: int, where: str) -> Array:
    op = np.asarray(op, dtype=np.complex128)
    if op.shape != (d, d):
        raise ValueError(f"Operator at {where} has shape {op.shape}; expected {(d, d)}.")
    return op


def _compact_mpo_from_terms(
    dims: Sequence[int],
    onsite: Sequence[Array],
    nonlocal_terms: Sequence[Tuple[complex, Dict[int, Array], str]],
    *,
    tolerance: float,
) -> List[Array]:
    """Build an exact MPO by sharing compatible operator prefixes and suffixes.

    The generic one-channel-per-string construction is robust but wasteful for
    vibronic Hamiltonians: many terms have the same electronic operator and
    differ only in the vibrational sites they touch.  This operator graph keeps
    a left prefix up to a central switch site and a right suffix afterwards,
    sharing those pieces between product strings.  It is exact for arbitrary
    finite product strings and is particularly effective for the pyrazine
    two- and three-site terms.
    """
    dims = [int(d) for d in dims]
    nsites = len(dims)
    start = ("START",)
    stop = ("STOP",)
    bond_states: list[set[tuple]] = [set() for _ in range(nsites + 1)]
    bond_states[0].add(start)
    bond_states[-1].add(stop)
    edge_ops: list[dict[tuple[tuple, tuple], Array]] = [{} for _ in range(nsites)]
    identities = [np.eye(dim, dtype=np.complex128) for dim in dims]

    def operator_key(op: Array) -> tuple[tuple[int, int], bytes]:
        contiguous = np.ascontiguousarray(op, dtype=np.complex128)
        return tuple(contiguous.shape), contiguous.tobytes()

    def bond_label(pairs: tuple, bond: int, switch_site: int) -> tuple:
        if bond == 0:
            return start
        if bond == nsites:
            return stop
        if bond <= switch_site:
            return ("L", tuple(pair for pair in pairs if pair[0] < bond))
        return ("R", tuple(pair for pair in pairs if pair[0] >= bond))

    def add_edge(site: int, left: tuple, right: tuple, op: Array) -> None:
        edge = (left, right)
        previous = edge_ops[site].get(edge)
        if previous is None:
            edge_ops[site][edge] = np.array(op, copy=True)
        else:
            edge_ops[site][edge] = previous + op

    # Onsite terms use one identity-routing state per target site.  This avoids
    # accidentally multiplying two different onsite contributions together.
    for target, op in enumerate(onsite):
        if np.linalg.norm(op) <= tolerance:
            continue
        if target == 0:
            if nsites == 1:
                add_edge(0, start, stop, op)
            else:
                onsite_state = ("O", target)
                for bond in range(1, nsites):
                    bond_states[bond].add(onsite_state)
                add_edge(0, start, onsite_state, op)
                for site in range(1, nsites):
                    add_edge(
                        site,
                        onsite_state,
                        stop if site == nsites - 1 else onsite_state,
                        identities[site],
                    )
        else:
            onsite_state = ("O", target)
            for bond in range(1, nsites):
                bond_states[bond].add(onsite_state)
            add_edge(0, start, onsite_state, identities[0])
            for site in range(1, nsites):
                left = onsite_state
                right = stop if site == nsites - 1 else onsite_state
                add_edge(site, left, right, op if site == target else identities[site])

    # Add each nonlocal product string to the shared operator graph.
    for coeff, ops, _label in nonlocal_terms:
        support = sorted(ops)
        if len(support) < 2:
            continue
        pairs = tuple((site, operator_key(ops[site])) for site in support)
        switch_site = support[(len(support) - 1) // 2]
        for bond in range(nsites + 1):
            bond_states[bond].add(bond_label(pairs, bond, switch_site))

        for site in range(nsites):
            left = bond_label(pairs, site, switch_site)
            right = bond_label(pairs, site + 1, switch_site)
            op = ops.get(site, identities[site])
            if site == switch_site:
                op = coeff * op
            add_edge(site, left, right, op)

    labels = [sorted(states, key=repr) for states in bond_states]
    indices = [{label: index for index, label in enumerate(states)} for states in labels]
    tensors: list[Array] = []
    for site, dim in enumerate(dims):
        tensor = np.zeros(
            (len(labels[site]), len(labels[site + 1]), dim, dim),
            dtype=np.complex128,
        )
        for (left, right), op in edge_ops[site].items():
            tensor[indices[site][left], indices[site + 1][right]] += op
        tensor[np.abs(tensor) <= tolerance] = 0.0
        tensors.append(tensor)
    return tensors


def _prefix_suffix_mpo_from_terms(
    dims: Sequence[int],
    terms: Sequence[ProductTerm],
    *,
    tolerance: float,
) -> List[Array]:
    """Compile product terms with the validated pyrazine prefix/suffix graph."""
    dims = [int(d) for d in dims]
    nsites = len(dims)
    start = ("START",)
    stop = ("STOP",)
    bond_states: list[set[tuple]] = [set() for _ in range(nsites + 1)]
    bond_states[0].add(start)
    bond_states[-1].add(stop)
    edge_ops: list[dict[tuple[tuple, tuple], Array]] = [{} for _ in range(nsites)]
    identities = [np.eye(dim, dtype=np.complex128) for dim in dims]

    def operator_key(op: Array) -> tuple[tuple[int, int], bytes]:
        contiguous = np.ascontiguousarray(op, dtype=np.complex128)
        return tuple(contiguous.shape), contiguous.tobytes()

    def bond_label(pairs: tuple, bond: int, switch_site: int) -> tuple:
        if bond == 0:
            return start
        if bond == nsites:
            return stop
        if bond <= switch_site:
            return ("L", tuple(pair for pair in pairs if pair[0] < bond))
        return ("R", tuple(pair for pair in pairs if pair[0] >= bond))

    for term in terms:
        if abs(term.coeff) <= tolerance:
            continue
        support = sorted(term.ops)
        if not support:
            raise ValueError("prefix/suffix MPO terms must have nonempty support")
        pairs = tuple((site, operator_key(term.ops[site])) for site in support)
        switch_site = support[(len(support) - 1) // 2]

        for bond in range(nsites + 1):
            bond_states[bond].add(bond_label(pairs, bond, switch_site))

        for site in range(nsites):
            left = bond_label(pairs, site, switch_site)
            right = bond_label(pairs, site + 1, switch_site)
            op = term.ops.get(site, identities[site])
            if site == switch_site:
                op = term.coeff * op
            edge = (left, right)
            previous = edge_ops[site].get(edge)
            if previous is None:
                edge_ops[site][edge] = np.array(op, copy=True)
            elif site == switch_site:
                edge_ops[site][edge] = previous + op
            elif not np.allclose(previous, op, atol=tolerance, rtol=0.0):
                raise RuntimeError(
                    f"incompatible shared MPO edge at site {site} for term {term.label!r}"
                )

    labels = [sorted(states, key=repr) for states in bond_states]
    indices = [{label: index for index, label in enumerate(states)} for states in labels]
    tensors: list[Array] = []
    for site, dim in enumerate(dims):
        tensor = np.zeros(
            (len(labels[site]), len(labels[site + 1]), dim, dim),
            dtype=np.complex128,
        )
        for (left, right), op in edge_ops[site].items():
            tensor[indices[site][left], indices[site + 1][right]] += op
        tensor[np.abs(tensor) <= tolerance] = 0.0
        tensors.append(tensor)
    return tensors


def build_mpo_from_terms(
    dims: Sequence[int],
    terms: Iterable[ProductTerm],
    *,
    zero_tol: float = 0.0,
) -> List[Array]:
    r"""Compile an operator-string Hamiltonian into an exact finite-state MPO.

    Mathematical form
    -----------------
    For :math:`L` sites, write the Hamiltonian as a sum of product strings,

    .. math::

       \hat H
       = \sum_{r=1}^{R} c_r
         \bigotimes_{i=0}^{L-1}\hat O_i^{(r)},

    where :math:`\hat O_i^{(r)}=\hat I_i` whenever term :math:`r` does not act on
    site :math:`i`.

    MPO conversion
    --------------
    Product strings are compiled into a shared prefix/suffix operator graph.
    One-site terms use dedicated identity-routing paths, while compatible
    two- and three-site strings share virtual states.  Without sharing, the
    fallback finite-state construction would use one channel per nonlocal
    string and have bond dimension

    .. math::

       D = R_{\mathrm{nl}} + 2,

    with virtual state 0 denoted ``IDLE`` and virtual state :math:`D-1` denoted
    ``DONE``.

    For a product term :math:`r` whose leftmost and rightmost nonidentity sites
    are :math:`s_0` and :math:`s_1`, respectively, the corresponding MPO path is

    .. math::

       \mathrm{IDLE}
       \xrightarrow[s_0]{\;c_r\hat O_{s_0}^{(r)}\;}
       r
       \xrightarrow[s_0<i<s_1]{\;\hat O_i^{(r)}\;\text{or}\;\hat I_i\;}
       r
       \xrightarrow[s_1]{\;\hat O_{s_1}^{(r)}\;}
       \mathrm{DONE}.

    Equivalently, the nonzero MPO blocks for this channel are

    .. math::

       \begin{aligned}
       W_{0r}^{[s_0]}
       &\mathrel{+}= c_r\hat O_{s_0}^{(r)}, \\
       W_{rr}^{[i]}
       &\mathrel{+}= \hat O_i^{(r)}
          \quad\text{or}\quad \hat I_i,
          \qquad s_0<i<s_1, \\
       W_{r,D-1}^{[s_1]}
       &\mathrel{+}= \hat O_{s_1}^{(r)}.
       \end{aligned}

    At every site, identity propagation and onsite terms are encoded as

    .. math::

       \begin{aligned}
       W_{00}^{[i]} &\mathrel{+}= \hat I_i, \\
       W_{D-1,D-1}^{[i]} &\mathrel{+}= \hat I_i, \\
       W_{0,D-1}^{[i]} &\mathrel{+}= \hat h_i.
       \end{aligned}

    Contracting the boundary vectors

    .. math::

       \langle L| = \langle\mathrm{IDLE}|,
       \qquad
       |R\rangle = |\mathrm{DONE}\rangle

    therefore reproduces exactly one Hamiltonian product string per allowed
    virtual path.

    Notes
    -----
    The prefix/suffix graph is exact and substantially reduces the bond
    dimension for multimode vibronic Hamiltonians such as pyrazine.  It still
    represents the requested operator strings without numerical compression."""
    dims = [int(d) for d in dims]
    if len(dims) == 0:
        raise ValueError("At least one site is required.")
    if any(d < 1 for d in dims):
        raise ValueError("All local dimensions must be >= 1.")

    L = len(dims)
    onsite = [np.zeros((d, d), dtype=np.complex128) for d in dims]
    nonlocal_terms: List[Tuple[complex, Dict[int, Array], str]] = []
    normalized_terms: List[ProductTerm] = []

    for term in terms:
        coeff = complex(term.coeff)
        if abs(coeff) <= zero_tol:
            continue

        clean_ops: Dict[int, Array] = {}
        for site, op in term.ops.items():
            site = int(site)
            if not 0 <= site < L:
                raise IndexError(f"Term '{term.label}' uses invalid site {site} for L={L}.")
            op_arr = _validate_local_operator(op, dims[site], f"site {site} in term '{term.label}'")
            if np.linalg.norm(op_arr) > zero_tol:
                clean_ops[site] = op_arr

        support = sorted(clean_ops)
        if len(support) == 0:
            # A term with explicit operators can become identically zero after
            # truncation (for example, ``n`` and ``a+a†`` on a d=1 boson
            # site).  It must not be interpreted as a scalar identity shift;
            # only a genuinely operator-free ProductTerm is a scalar shift.
            if term.ops:
                continue
            # Scalar energy shift: place it on site 0 as coeff * I.
            onsite[0] += coeff * np.eye(dims[0], dtype=np.complex128)
            normalized_terms.append(
                ProductTerm(coeff, {0: np.eye(dims[0], dtype=np.complex128)}, term.label)
            )
        elif len(support) == 1:
            s = support[0]
            onsite[s] += coeff * clean_ops[s]
            normalized_terms.append(ProductTerm(coeff, clean_ops, term.label))
        else:
            nonlocal_terms.append((coeff, clean_ops, term.label))
            normalized_terms.append(ProductTerm(coeff, clean_ops, term.label))

    return _prefix_suffix_mpo_from_terms(
        dims,
        normalized_terms,
        tolerance=max(zero_tol, 1.0e-15),
    )

    # Use the compact operator graph for product-string sharing.  Nearest-
    # neighbor product strings can share a finite-state channel as long
    # as no two strings assigned to that channel live on the same bond.  A
    # channel with only ``IDLE -> r`` and ``r -> DONE`` transitions cannot join
    # separated bonds or create longer-range products.  This is especially
    # useful for chain Hamiltonians: all b†_j b_(j+1) terms share one channel,
    # and all b_j b†_(j+1) terms share another one.
    #
    # First combine identical strings on one bond, then greedily color the
    # remaining adjacent strings so each channel contains at most one string
    # per bond.  Non-nearest-neighbor terms retain one channel each.
    adjacent_by_key: Dict[Tuple[int, bytes, bytes], Tuple[complex, Dict[int, Array], str]] = {}
    non_adjacent_terms: List[Tuple[complex, Dict[int, Array], str]] = []
    for coeff, ops, label in nonlocal_terms:
        support = sorted(ops)
        if len(support) == 2 and support[1] == support[0] + 1:
            left, right = support
            left_key = np.ascontiguousarray(ops[left]).tobytes()
            right_key = np.ascontiguousarray(ops[right]).tobytes()
            key = (left, left_key, right_key)
            if key in adjacent_by_key:
                old_coeff, old_ops, old_label = adjacent_by_key[key]
                adjacent_by_key[key] = (old_coeff + coeff, old_ops, old_label)
            else:
                adjacent_by_key[key] = (coeff, ops, label)
        else:
            non_adjacent_terms.append((coeff, ops, label))

    adjacent_terms = [term for term in adjacent_by_key.values() if abs(term[0]) > zero_tol]
    adjacent_channels: List[List[Tuple[complex, Dict[int, Array], str]]] = []
    channel_bonds: List[set[int]] = []
    for term in adjacent_terms:
        bond = min(term[1])
        for channel, used_bonds in zip(adjacent_channels, channel_bonds):
            if bond not in used_bonds:
                channel.append(term)
                used_bonds.add(bond)
                break
        else:
            adjacent_channels.append([term])
            channel_bonds.append({bond})

    Rnl = len(adjacent_channels) + len(non_adjacent_terms)
    D = Rnl + 2
    IDLE = 0
    DONE = D - 1

    mpo_full: List[Array] = []
    for d in dims:
        W = np.zeros((D, D, d, d), dtype=np.complex128)
        I = np.eye(d, dtype=np.complex128)
        W[IDLE, IDLE] += I
        W[DONE, DONE] += I
        mpo_full.append(W)

    # Local Hamiltonians.
    for site, h in enumerate(onsite):
        mpo_full[site][IDLE, DONE] += h

    # Shared channels for nearest-neighbor product strings.
    for r, channel_terms in enumerate(adjacent_channels, start=1):
        for coeff, ops, _label in channel_terms:
            support = sorted(ops)
            first, last = support
            mpo_full[first][IDLE, r] += coeff * ops[first]
            mpo_full[last][r, DONE] += ops[last]

    # One channel per remaining nonlocal product string.
    channel_offset = 1 + len(adjacent_channels)
    for r, (coeff, ops, _label) in enumerate(non_adjacent_terms, start=channel_offset):
        support = sorted(ops)
        first, last = support[0], support[-1]
        mpo_full[first][IDLE, r] += coeff * ops[first]

        for site in range(first + 1, last):
            if site in ops:
                mpo_full[site][r, r] += ops[site]
            else:
                mpo_full[site][r, r] += np.eye(dims[site], dtype=np.complex128)

        mpo_full[last][r, DONE] += ops[last]

    return apply_boundaries(mpo_full)


def apply_boundaries(mpo_full: Sequence[Array]) -> List[Array]:
    r"""Apply the MPO boundary vectors :math:`\langle\mathrm{IDLE}|` and :math:`|\mathrm{DONE}\rangle`.

    The input tensors must all have shape
    :math:`(D,D,d_i,d_i)`.  The returned boundary-shaped tensors have dimensions

    .. math::

       (1,D,d_0,d_0),\quad
       (D,D,d_1,d_1),\quad\ldots,\quad
       (D,1,d_{L-1},d_{L-1})."""
    if len(mpo_full) == 0:
        raise ValueError("Empty MPO.")

    D = mpo_full[0].shape[0]
    DONE = D - 1
    for i, W in enumerate(mpo_full):
        if W.shape[0] != D or W.shape[1] != D:
            raise ValueError(f"Full MPO tensor {i} does not have common square bond dimension {D}.")

    vL = np.zeros(D, dtype=np.complex128)
    vR = np.zeros(D, dtype=np.complex128)
    vL[0] = 1.0
    vR[DONE] = 1.0

    if len(mpo_full) == 1:
        local = np.einsum("l,lrab,r->ab", vL, mpo_full[0], vR)
        return [local[None, None, :, :]]

    W0 = np.tensordot(vL, mpo_full[0], axes=(0, 0))[None, :, :, :]
    WN = np.tensordot(mpo_full[-1], vR, axes=(1, 0))[:, None, :, :]
    return [W0] + [np.array(W, copy=True) for W in mpo_full[1:-1]] + [WN]


# -----------------------------------------------------------------------------
# PCET/vibronic Hamiltonian -> ProductTerm list -> MPO
# -----------------------------------------------------------------------------

def _matrix_power(op: Array, power: int) -> Array:
    if power < 0:
        raise ValueError("Polynomial operator powers must be nonnegative.")
    return np.linalg.matrix_power(op, power)


def pcet_terms(model: PCETModel) -> Tuple[List[int], List[ProductTerm], Dict[str, int]]:
    r"""Expand a :class:`PCETModel` into local operator-product terms.

    General Hamiltonian
    -------------------
    For :math:`N_e` electronic diabatic states :math:`|\alpha\rangle` and
    :math:`N_v` bosonic coordinates :math:`\mu`, define

    .. math::

       \begin{aligned}
       \hat P_\alpha &= |\alpha\rangle\langle\alpha|, \\
       \hat q_\mu &= \frac{\hat a_\mu+\hat a_\mu^\dagger}{\sqrt{2}}, \\
       \hat p_\mu &= \frac{i(\hat a_\mu^\dagger-\hat a_\mu)}{\sqrt{2}}.
       \end{aligned}

    The convenience Hamiltonian implemented here is

    .. math::

       \begin{aligned}
       \hat H
       =&\; \hat H_{\mathrm{el}}
       + \sum_{\mu=1}^{N_v}
           \omega_\mu\left(\hat n_\mu+\frac{z_\mu}{2}\right)
       + \hat H_{\mathrm{disp}} \\
       &+ \sum_\mu \hat G_\mu\hat q_\mu
       + \sum_\mu \hat K_\mu\hat q_\mu^2 \\
       &+ \sum_{\mu<\nu}
           \kappa_{\mu\nu}\hat O_\mu\hat O_\nu
       + \sum_{\mu<\nu}
           \hat M_{\mu\nu}\hat O_\mu\hat O_\nu \\
       &+ \hat H_{\mathrm{anh}} + \hat H_{\mathrm{extra}}.
       \end{aligned}

    where :math:`z_\mu=1` when zero-point energy is included and
    :math:`z_\mu=0` otherwise.  The matrices :math:`\hat G_\mu`,
    :math:`\hat K_\mu`, and :math:`\hat M_{\mu\nu}` act in the electronic
    subspace.

    If ``displacements[alpha, mu]`` supplies :math:`d_{\alpha\mu}`, then

    .. math::

       \hat H_{\mathrm{disp}}
       = \sum_{\alpha=1}^{N_e}\sum_{\mu=1}^{N_v}
         \hat P_\alpha
         \left[
           -\omega_\mu d_{\alpha\mu}\hat q_\mu
           + \frac{\omega_\mu}{2}d_{\alpha\mu}^2
         \right].

    This follows by expanding the displaced oscillator

    .. math::

       \hat H_\mu^{(\alpha)}
       = \frac{\omega_\mu}{2}
         \left[
           \hat p_\mu^2
           + \left(\hat q_\mu-d_{\alpha\mu}\right)^2
         \right].

    Electronic block-matrix representation
    ---------------------------------------
    In the diabatic electronic basis, the Hamiltonian is an
    :math:`N_e\times N_e` matrix whose entries are operators on the full nuclear
    space,

    .. math::

       \hat{\mathbf H}(\mathbf Q)
       =
       \begin{pmatrix}
         \hat H_{11}(\mathbf Q) & \hat H_{12}(\mathbf Q) & \cdots \\
         \hat H_{21}(\mathbf Q) & \hat H_{22}(\mathbf Q) & \cdots \\
         \vdots & \vdots & \ddots
       \end{pmatrix}.

    A generic block has the structure

    .. math::

       \begin{aligned}
       \hat H_{\alpha\beta}(\mathbf Q)
       =&\; (H_{\mathrm{el}})_{\alpha\beta}
       + \delta_{\alpha\beta}\hat H_{\mathrm{vib}} \\
       &+ \sum_\mu (G_\mu)_{\alpha\beta}\hat q_\mu
       + \sum_\mu (K_\mu)_{\alpha\beta}\hat q_\mu^2 \\
       &+ \sum_{\mu<\nu}
          (M_{\mu\nu})_{\alpha\beta}
          \hat O_\mu\hat O_\nu + \cdots .
       \end{aligned}

    Thus diagonal electronic blocks define diabatic nuclear Hamiltonians/PESs,
    whereas off-diagonal blocks generate electronic and nonadiabatic vibronic
    couplings."""
    H_el = np.asarray(model.electronic_hamiltonian, dtype=np.complex128)
    if H_el.ndim != 2 or H_el.shape[0] != H_el.shape[1]:
        raise ValueError("electronic_hamiltonian must be a square matrix.")
    Ne = H_el.shape[0]

    modes = list(model.modes)
    if len({m.name for m in modes}) != len(modes):
        raise ValueError("Boson mode names must be unique.")
    for m in modes:
        if m.dim < 1:
            raise ValueError(f"Mode {m.name!r} has invalid dim={m.dim}.")

    Nv = len(modes)
    dims = [Ne] + [m.dim for m in modes]
    mode_site = {m.name: i + 1 for i, m in enumerate(modes)}
    mode_index = {m.name: i for i, m in enumerate(modes)}
    mode_ops = {m.name: boson_ops(m.dim) for m in modes}

    if model.electronic_names is not None and len(model.electronic_names) != Ne:
        raise ValueError("electronic_names length must equal Ne.")

    H_el_effective = np.array(H_el, copy=True)
    terms: List[ProductTerm] = []

    # State-dependent displaced harmonic surfaces.
    if model.displacements is not None:
        disp = np.asarray(model.displacements, dtype=float)
        if disp.shape != (Ne, Nv):
            raise ValueError(f"displacements must have shape {(Ne, Nv)}, got {disp.shape}.")
        for mu, mode in enumerate(modes):
            dvec = disp[:, mu]
            H_el_effective += np.diag(0.5 * mode.omega * dvec**2)
            Gdisp = np.diag(-mode.omega * dvec).astype(np.complex128)
            if np.linalg.norm(Gdisp) > 0.0:
                terms.append(ProductTerm(1.0, {0: Gdisp, mode_site[mode.name]: mode_ops[mode.name]["q"]}, f"disp:{mode.name}"))

    # Electronic Hamiltonian including displacement constants.
    terms.append(ProductTerm(1.0, {0: H_el_effective}, "H_el"))

    # Bosonic onsite Hamiltonians and local anharmonicities.
    for mode in modes:
        ops = mode_ops[mode.name]
        h = mode.omega * ops["n"]
        if mode.include_zpe:
            h = h + 0.5 * mode.omega * ops["I"]

        for power, coeff in mode.anharmonic_q.items():
            h = h + complex(coeff) * _matrix_power(ops["q"], int(power))

        if mode.onsite_extra is not None:
            h = h + _validate_local_operator(mode.onsite_extra, mode.dim, f"onsite_extra for {mode.name}")

        terms.append(ProductTerm(1.0, {mode_site[mode.name]: h}, f"H_mode:{mode.name}"))

    # Electronic-vibrational linear and quadratic couplings.
    for name, G in model.linear_el_vib.items():
        if name not in mode_site:
            raise KeyError(f"linear_el_vib refers to unknown mode {name!r}.")
        G = _validate_local_operator(G, Ne, f"linear_el_vib[{name!r}]")
        terms.append(ProductTerm(1.0, {0: G, mode_site[name]: mode_ops[name]["q"]}, f"Gq:{name}"))

    for name, K in model.quadratic_el_vib.items():
        if name not in mode_site:
            raise KeyError(f"quadratic_el_vib refers to unknown mode {name!r}.")
        K = _validate_local_operator(K, Ne, f"quadratic_el_vib[{name!r}]")
        terms.append(ProductTerm(1.0, {0: K, mode_site[name]: mode_ops[name]["q2"]}, f"Kq2:{name}"))

    # Coordinate-coordinate couplings, including non-nearest neighbors.
    for cpl in model.mode_mode:
        if cpl.mode_left not in mode_site or cpl.mode_right not in mode_site:
            raise KeyError(f"Unknown mode in coupling {cpl}.")
        if cpl.mode_left == cpl.mode_right:
            raise ValueError("ModeModeCoupling requires two distinct mode sites; use onsite_extra for one mode.")
        op_l = mode_ops[cpl.mode_left].get(cpl.op_left)
        op_r = mode_ops[cpl.mode_right].get(cpl.op_right)
        if op_l is None or op_r is None:
            raise KeyError(f"Unknown bosonic operator in coupling {cpl}.")
        terms.append(ProductTerm(cpl.coeff, {mode_site[cpl.mode_left]: op_l, mode_site[cpl.mode_right]: op_r}, cpl.label))

    # Electronic-state-dependent two-mode couplings.
    for cpl in model.electronic_mode_mode:
        if cpl.mode_left not in mode_site or cpl.mode_right not in mode_site:
            raise KeyError(f"Unknown mode in electronic coupling {cpl}.")
        Gel = _validate_local_operator(cpl.electronic_operator, Ne, cpl.label)
        op_l = mode_ops[cpl.mode_left].get(cpl.op_left)
        op_r = mode_ops[cpl.mode_right].get(cpl.op_right)
        if op_l is None or op_r is None:
            raise KeyError(f"Unknown bosonic operator in coupling {cpl}.")
        terms.append(ProductTerm(cpl.coeff, {0: Gel, mode_site[cpl.mode_left]: op_l, mode_site[cpl.mode_right]: op_r}, cpl.label))

    # Arbitrary user-supplied strings.  Sites use the final MPO numbering:
    # electronic=0, modes=1..Nv.
    terms.extend(model.extra_terms)

    return dims, terms, mode_site


def build_pcet_mpo(model: PCETModel, *, zero_tol: float = 0.0) -> List[Array]:
    r"""Construct the generalized vibronic/PCET Hamiltonian as an MPO.

    This is the main high-level constructor.  It supports

    * :math:`N_e` electronic diabatic states in one local electronic site;
    * :math:`N_v=N_H+N_{CT}+N_s` bosonic coordinates, if partitioned into proton,
      charge-transfer, and skeletal modes;
    * different local truncations :math:`d_\mu` for every coordinate;
    * displaced harmonic diabatic surfaces;
    * arbitrary electronic couplings in :math:`\hat H_{\mathrm{el}}`;
    * diagonal or off-diagonal linear vibronic couplings
      :math:`\hat G_\mu\hat q_\mu`;
    * quadratic vibronic couplings :math:`\hat K_\mu\hat q_\mu^2`;
    * proton-CT, proton-skeletal, CT-skeletal, and skeletal-skeletal couplings;
    * electronic-state-dependent mode-mode couplings; and
    * arbitrary additional operator products through :class:`ProductTerm`.

    Hamiltonian
    -----------
    With

    .. math::

       \begin{aligned}
       \hat q_\mu &= \frac{\hat a_\mu+\hat a_\mu^\dagger}{\sqrt{2}}, \\
       \hat p_\mu &= \frac{i(\hat a_\mu^\dagger-\hat a_\mu)}{\sqrt{2}}, \\
       \hat P_\alpha &= |\alpha\rangle\langle\alpha|.
       \end{aligned}

    one useful general PCET/vibronic Hamiltonian is

    .. math::

       \begin{aligned}
       \hat H
       =&\; \hat H_{\mathrm{el}}
       + \sum_\mu \omega_\mu
           \left(\hat n_\mu+\frac{1}{2}\right) \\
       &+ \sum_{\alpha,\mu}
          \hat P_\alpha
          \left[
            -\omega_\mu d_{\alpha\mu}\hat q_\mu
            + \frac{\omega_\mu}{2}d_{\alpha\mu}^2
          \right] \\
       &+ \sum_\mu \hat G_\mu\hat q_\mu
       + \sum_\mu \hat K_\mu\hat q_\mu^2 \\
       &+ \sum_{\mu<\nu}
          \kappa_{\mu\nu}\hat O_\mu\hat O_\nu
       + \sum_{\mu<\nu}
          \hat M_{\mu\nu}\hat O_\mu\hat O_\nu \\
       &+ \hat H_{\mathrm{anh}} + \hat H_{\mathrm{extra}}.
       \end{aligned}

    Here :math:`\hat G_\mu`, :math:`\hat K_\mu`, and
    :math:`\hat M_{\mu\nu}` are matrices in the electronic subspace.  Their
    diagonal elements modify diabatic potential-energy surfaces; their
    off-diagonal elements generate nonadiabatic/vibronic transitions.

    Electronic matrix representation
    ---------------------------------
    The same operator may be written as

    .. math::

       \hat{\mathbf H}(\mathbf Q)
       =
       \begin{pmatrix}
          \hat H_{11}(\mathbf Q) & \hat H_{12}(\mathbf Q) & \cdots \\
          \hat H_{21}(\mathbf Q) & \hat H_{22}(\mathbf Q) & \cdots \\
          \vdots & \vdots & \ddots
       \end{pmatrix},

    where each :math:`\hat H_{\alpha\beta}(\mathbf Q)` acts on the tensor-product
    vibrational space.

    Conversion to an MPO
    --------------------
    :func:`pcet_terms` first rewrites the Hamiltonian as

    .. math::

       \hat H
       = \sum_r c_r
         \bigotimes_{i=0}^{L-1}\hat O_i^{(r)}.

    :func:`build_mpo_from_terms` then assigns one finite-state channel to every
    nonlocal product string.  For a two-site term

    .. math::

       \hat H_r = c_r\hat A^{(i)}\hat B^{(j)},
       \qquad i<j,

    the virtual path is

    .. math::

       \mathrm{IDLE}
       \xrightarrow[i]{\;c_r\hat A\;}
       r
       \xrightarrow[i<k<j]{\;\hat I_k\;}
       r
       \xrightarrow[j]{\;\hat B\;}
       \mathrm{DONE}.

    For a many-site string, any required intermediate operator is inserted on the
    same channel rather than an identity.  Onsite Hamiltonians use the direct
    ``IDLE -> DONE`` path.  Consequently, contracting the MPO reproduces the
    operator-string Hamiltonian exactly before any optional compression.

    Returns
    -------
    list[numpy.ndarray]
        Boundary-shaped MPO tensors with shapes
        :math:`(1,D,d_0,d_0)`, :math:`(D,D,d_1,d_1)`, ..., and
        :math:`(D,1,d_{L-1},d_{L-1})`."""
    dims, terms, _ = pcet_terms(model)
    return build_mpo_from_terms(dims, terms, zero_tol=zero_tol)


# -----------------------------------------------------------------------------
# Dense-matrix utilities for small-model verification only
# -----------------------------------------------------------------------------

def terms_to_dense(dims: Sequence[int], terms: Iterable[ProductTerm]) -> Array:
    r"""Build the full dense Hamiltonian from product terms.

    This utility is intended only for small verification calculations because the
    dense matrix dimension is

    .. math::

       D_{\mathrm{dense}}
       = \prod_{i=0}^{L-1} d_i."""
    dims = list(map(int, dims))
    Dtot = int(np.prod(dims))
    H = np.zeros((Dtot, Dtot), dtype=np.complex128)

    for term in terms:
        local = []
        for site, d in enumerate(dims):
            if site in term.ops:
                op = _validate_local_operator(term.ops[site], d, f"dense term {term.label}, site {site}")
            else:
                op = np.eye(d, dtype=np.complex128)
            local.append(op)

        block = np.array([[1.0 + 0.0j]])
        for op in local:
            block = np.kron(block, op)
        H += complex(term.coeff) * block

    return H


def mpo_to_dense(mpo: Sequence[Array]) -> Array:
    r"""Contract a boundary-shaped MPO into a dense matrix for testing.

    For an MPO with tensors :math:`W^{[0]},\ldots,W^{[L-1]}`, this routine forms
    the full operator obtained by contracting all virtual indices and retaining
    the physical matrix indices.  Do not use it for production-sized vibronic
    spaces because the result scales with the full dense Hilbert-space dimension."""
    if len(mpo) == 0:
        raise ValueError("Empty MPO.")
    if mpo[0].shape[0] != 1 or mpo[-1].shape[1] != 1:
        raise ValueError("Expected boundary-shaped MPO with left/right bond dimension 1.")

    # After site 0, blocks[r] is an operator on sites processed so far.
    blocks = [np.array(mpo[0][0, r], copy=True) for r in range(mpo[0].shape[1])]

    for site in range(1, len(mpo)):
        W = mpo[site]
        if W.shape[0] != len(blocks):
            raise ValueError(f"MPO bond mismatch before site {site}.")
        new_blocks: List[Array] = []
        for r_out in range(W.shape[1]):
            acc = None
            for r_in, left_block in enumerate(blocks):
                local = W[r_in, r_out]
                if np.linalg.norm(local) == 0.0:
                    continue
                piece = np.kron(left_block, local)
                acc = piece if acc is None else acc + piece
            if acc is None:
                d_left = blocks[0].shape[0]
                d_local = W.shape[2]
                acc = np.zeros((d_left * d_local, d_left * d_local), dtype=np.complex128)
            new_blocks.append(acc)
        blocks = new_blocks

    if len(blocks) != 1:
        raise ValueError("Right boundary did not close MPO to bond dimension 1.")
    return blocks[0]


def verify_mpo(model: PCETModel, *, atol: float = 1e-10, rtol: float = 1e-10) -> Dict[str, float]:
    r"""Compare the direct product-term Hamiltonian with the dense MPO contraction.

    For a small model, this evaluates

    .. math::

       \left\|\hat H_{\mathrm{terms}}-\hat H_{\mathrm{MPO}}\right\|

    and also reports the Hermiticity error

    .. math::

       \left\|\hat H-\hat H^\dagger\right\|."""
    dims, terms, _ = pcet_terms(model)
    H_terms = terms_to_dense(dims, terms)
    H_mpo = mpo_to_dense(build_mpo_from_terms(dims, terms))
    err = np.linalg.norm(H_terms - H_mpo)
    scale = max(np.linalg.norm(H_terms), 1.0)
    ok = np.allclose(H_terms, H_mpo, atol=atol, rtol=rtol)
    herm_err = np.linalg.norm(H_terms - H_terms.conj().T)
    return {
        "absolute_error": float(err),
        "relative_error": float(err / scale),
        "hermiticity_error": float(herm_err),
        "verified": float(bool(ok)),
        "dense_dimension": float(H_terms.shape[0]),
    }


# -----------------------------------------------------------------------------
# Units and source-inspired HBQ/HBT examples
# -----------------------------------------------------------------------------


def cm1_to_ev(wavenumber_cm1: float) -> float:
    r"""Convert a spectroscopic wavenumber :math:`\tilde\nu` in cm\ :sup:`-1` to eV."""
    return float(wavenumber_cm1) / CM1_PER_EV


def example_hbq_reduced() -> PCETModel:
    r"""Return a four-state, one-proton-mode HBQ model inspired by Zhang et al. (2023).

    The electronic basis is :math:`\{|g\rangle,|e\rangle,|k\rangle,|k'\rangle\}`.
    Source-scale anchors used in this example include a proton-basis truncation of
    20 states for HBQ, an excited-state displacement difference
    :math:`\delta_{q,e}-\delta_{q,g}=0.63`, an excited enol-keto minimum gap of
    about :math:`0.49\,\mathrm{eV}`, and an electronic coupling scale of about
    :math:`0.03\,\mathrm{eV}`.

    The proton frequency is set to :math:`2696\,\mathrm{cm}^{-1}`, using the HBQ
    reaction-coordinate kinematic frequency reported by Picconi (2021), rather
    than claiming to reproduce the exact Zhang supporting-information parameter
    row.  The excited enol-to-keto displacement is an illustrative effective
    value for numerical testing."""
    names = ["g", "e", "k", "kprime"]
    Ne = len(names)

    # Pump center 390 nm is about 3.18 eV.  k is lower by about 0.49 eV.
    E = np.array([0.0, 3.18, 3.18 - 0.49, 3.18 - 0.54])
    H_el = np.diag(E).astype(np.complex128)
    H_el[1, 2] = H_el[2, 1] = 0.03

    mode = BosonMode("proton", dim=20, omega=cm1_to_ev(2696.0), kind="proton")
    disp = np.array([
        [0.00],
        [0.63],
        [3.63],
        [3.63],
    ])
    return PCETModel(H_el, [mode], electronic_names=names, displacements=disp)


def example_hbt_reduced() -> PCETModel:
    r"""Return a four-state, one-proton-mode HBT model inspired by Zhang et al. (2023).

    The electronic basis is :math:`\{|g\rangle,|e\rangle,|k\rangle,|k'\rangle\}`.
    The example uses a 25-state proton basis, the reduced-model displacement
    anchor :math:`\delta_{q,e}-\delta_{q,g}=0.63`, an excited enol-keto minimum
    gap of about :math:`0.45\,\mathrm{eV}`, and an electronic coupling scale of
    :math:`0.03\,\mathrm{eV}`.

    The proton frequency and effective excited-state displacement are
    source-scale illustrative choices, not a transcription of a complete fitted
    parameter row from the supporting information."""
    names = ["g", "e", "k", "kprime"]
    E = np.array([0.0, 3.44, 3.44 - 0.45, 3.44 - 0.50])
    H_el = np.diag(E).astype(np.complex128)
    H_el[1, 2] = H_el[2, 1] = 0.03

    mode = BosonMode("proton", dim=25, omega=cm1_to_ev(2800.0), kind="proton")
    disp = np.array([
        [0.00],
        [0.63],
        [3.8],
        [3.8],
    ])
    return PCETModel(H_el, [mode], electronic_names=names, displacements=disp)


def example_hbq_picconi_reduced() -> PCETModel:
    r"""Return a reduced three-state, multimode HBQ vibronic model.

    Picconi (2021) uses diabatic :math:`1\pi\pi^*`, :math:`2\pi\pi^*`, and
    :math:`n\pi^*` electronic states together with a 31-dimensional vibrational
    model consisting of reaction coordinate :math:`Q_1` plus 30 skeletal modes.
    The reaction coordinate is dominated by O-H stretching and has a reported
    kinematic frequency of approximately :math:`2696\,\mathrm{cm}^{-1}`.  A
    highlighted skeletal mode :math:`Q_{58}` lies near
    :math:`1710\,\mathrm{cm}^{-1}`, while coherent low-frequency skeletal motion
    appears broadly below about :math:`800\,\mathrm{cm}^{-1}`.

    This example keeps only :math:`Q_1` and three skeletal coordinates so the MPO
    is easy to inspect.  The electronic energies and vibronic couplings are a
    reduced demonstration model rather than a digitization of the full
    reaction-path surfaces."""
    names = ["1pi_pi", "2pi_pi", "n_pi"]
    E_abs = np.array([3.24, 3.69, 4.42])
    H_el = np.diag(E_abs - E_abs[0]).astype(np.complex128)

    modes = [
        BosonMode("Q1_proton", 12, cm1_to_ev(2696.0), "proton"),
        BosonMode("Q_low_390", 7, cm1_to_ev(390.0), "skeletal"),
        BosonMode("Q_low_550", 7, cm1_to_ev(550.0), "skeletal"),
        BosonMode("Q58", 6, cm1_to_ev(1710.0), "skeletal"),
    ]

    # State-dependent reaction-path shifts (illustrative reduced fit).
    disp = np.array([
        [3.0, 0.4, 0.2, 0.2],
        [2.4, 0.7, 0.3, 0.4],
        [3.4, 0.2, 0.5, 0.6],
    ])

    Ne = 3
    # pi-pi / pi-pi coupling primarily through the reaction coordinate.
    G_q1 = hermitian_electronic_coupling(Ne, 0, 1, 0.025)
    # Out-of-plane skeletal-like coupling to n-pi state.
    G_low = hermitian_electronic_coupling(Ne, 1, 2, 0.020)
    G_q58 = hermitian_electronic_coupling(Ne, 0, 2, 0.015)

    # A small q-q term demonstrates mode mixing / Duschinsky-like structure.
    mm = [ModeModeCoupling("Q1_proton", "Q58", 0.004, "q", "q", "Q1-Q58")]

    return PCETModel(
        H_el,
        modes,
        electronic_names=names,
        displacements=disp,
        linear_el_vib={"Q1_proton": G_q1, "Q_low_390": G_low, "Q58": G_q58},
        mode_mode=mm,
    )


def example_hbq_pcet_with_ct() -> PCETModel:
    r"""Return an illustrative HBQ/HBT-style PCET extension with an explicit
    CT coordinate (JPCB 114, 12319 (2010))

    This is intentionally a model extension rather than a claim that the cited
    HBQ/HBT ESIPT papers fitted an independent solvent/charge-transfer coordinate.
    The tensor-network architecture is

    .. math::

       \begin{aligned}
       N_e &= 3, \\
       N_H &= 1, \\
       N_{CT} &= 1, \\
       N_s &= 3, \\
       N_v &= N_H+N_{CT}+N_s = 5.
       \end{aligned}

    The proton and skeletal frequency scales are anchored to HBQ literature.  The
    soft CT frequency, set here to :math:`250\,\mathrm{cm}^{-1}`, and its coupling
    constants are generic illustrative PCET parameters that should be replaced by
    a fitted reorganization model or ab-initio diabatic data in quantitative work."""
    names = ["enol_star", "keto_star", "ct_star"]
    Ne = 3

    H_el = np.array(
        [
            [0.00, 0.03, 0.01],
            [0.03, -0.49, 0.02],
            [0.01, 0.02, 0.18],
        ],
        dtype=np.complex128,
    )

    modes = [
        BosonMode("H", 12, cm1_to_ev(2696.0), "proton"),
        BosonMode("CT", 8, cm1_to_ev(250.0), "ct"),
        BosonMode("S390", 6, cm1_to_ev(390.0), "skeletal"),
        BosonMode("S550", 6, cm1_to_ev(550.0), "skeletal"),
        BosonMode("S1710", 5, cm1_to_ev(1710.0), "skeletal"),
    ]

    # Rows = electronic states, columns = H, CT, S390, S550, S1710.
    disp = np.array([
        [0.0, -0.7, 0.0, 0.0, 0.0],
        [3.0, +0.5, 0.4, 0.2, 0.2],
        [1.5, +1.3, 0.2, 0.4, 0.3],
    ])

    # Off-diagonal vibronic couplings.
    G_H = hermitian_electronic_coupling(Ne, 0, 1, 0.010)
    G_CT = hermitian_electronic_coupling(Ne, 1, 2, 0.015)
    G_S = hermitian_electronic_coupling(Ne, 0, 2, 0.008)

    mode_mode = [
        ModeModeCoupling("H", "CT", 0.006, "q", "q", "proton-CT"),
        ModeModeCoupling("H", "S390", 0.004, "q", "q", "proton-skeleton"),
        ModeModeCoupling("CT", "S550", 0.003, "q", "q", "CT-skeleton"),
    ]

    # State-dependent proton-skeleton coupling: P_keto * q_H * q_S1710.
    Pk = electronic_projector(Ne, 1)
    el_mode_mode = [
        ElectronicModeModeCoupling(Pk, "H", "S1710", 0.003, "q", "q", "keto:H-S1710")
    ]

    return PCETModel(
        H_el,
        modes,
        electronic_names=names,
        displacements=disp,
        linear_el_vib={"H": G_H, "CT": G_CT, "S390": G_S},
        mode_mode=mode_mode,
        electronic_mode_mode=el_mode_mode,
    )


# from protontransfermpo.py module (which is to be deprecated)
def legacy_protontransfer_model(
    w0e: float,
    w0k: float,
    x0e: float,
    x0k: float,
    Delta: float,
    dRC: int,
    d: int,
    N: int,
    bathparams,
    RCparams,
    lambda_reorg: float,
    *,
    include_zero_point: bool = True,
    sign_g: float = -1.0,
) -> PCETModel:
    r"""Translate the original ``protontransfermpo.py`` Hamiltonian to :class:`PCETModel`.

    The legacy model is

    .. math::

       \hat H
       = \hat H_S
       + \hat H_{RC}
       + \hat H_{S-RC}
       + \hat H_{\mathrm{bath}}
       + \hat H_{RC-\mathrm{bath}}
       + \hat H_{\mathrm{counter}}.

    The two-state electronic Hamiltonian is

    .. math::

       \hat H_S
       = \omega_{0e}|e\rangle\langle e|
       + \omega_{0k}|k\rangle\langle k|
       + \Delta\left(
           |e\rangle\langle k| + |k\rangle\langle e|
         \right).

    The reaction-coordinate Hamiltonian is

    .. math::

       \hat H_{RC}
       = \omega_{RC}
         \left(\hat n_{RC}+\frac{z}{2}\right),

    where :math:`z=1` when ``include_zero_point=True`` and :math:`z=0` otherwise.
    The system-RC coupling is

    .. math::

       \hat H_{S-RC}
       = \left(
           g_e|e\rangle\langle e|
           + g_k|k\rangle\langle k|
         \right)\hat x_{RC},

    with

    .. math::

       \begin{aligned}
       g_e &= s_g\,g_{\mathrm{scale}}x_{0e}, \\
       g_k &= s_g\,g_{\mathrm{scale}}x_{0k}.
       \end{aligned}

    where :math:`s_g` is ``sign_g``.  The chain-mapped bath Hamiltonian is

    .. math::

       \hat H_{\mathrm{bath}}
       = \sum_{j=0}^{N-1}\epsilon_j\hat n_j
       + \sum_{j=0}^{N-2} t_j
         \left(
           \hat b_j^\dagger\hat b_{j+1}
           + \hat b_j\hat b_{j+1}^\dagger
         \right).

    The RC-bath interaction and counter term are

    .. math::

       \begin{aligned}
       \hat H_{RC-\mathrm{bath}}
       &= -c_0\hat x_{RC}\hat x_0, \\
       \hat H_{\mathrm{counter}}
       &= \lambda_{\mathrm{reorg}}\hat x_{RC}^2.
       \end{aligned}

    with :math:`\hat x=\hat a+\hat a^\dagger`.  The returned site order is
    ``[electronic, RC, bath_0, ..., bath_(N-1)]``.

    This adapter is useful for regression tests while migrating existing scripts
    to the generalized operator-string/MPO implementation."""
    if N < 1:
        raise ValueError("N must be >= 1.")
    eps_list, t_list, c0 = bathparams
    eps_list = list(eps_list)
    t_list = list(t_list)
    if len(eps_list) != N:
        raise ValueError(f"Expected {N} bath onsite energies, got {len(eps_list)}.")
    if len(t_list) != N - 1:
        raise ValueError(f"Expected {N-1} bath hoppings, got {len(t_list)}.")

    wRC, g_scale = RCparams
    Ne = 2
    Pee = electronic_projector(Ne, 0)
    Pkk = electronic_projector(Ne, 1)
    H_el = w0e * Pee + w0k * Pkk + Delta * hermitian_electronic_coupling(Ne, 0, 1)

    rc_ops = boson_ops(dRC)
    modes: List[BosonMode] = [
        BosonMode(
            "RC",
            dRC,
            wRC,
            "reaction_coordinate",
            include_zero_point,
            onsite_extra=lambda_reorg * rc_ops["x2"],
        )
    ]
    for j, eps in enumerate(eps_list):
        modes.append(BosonMode(f"bath{j}", d, eps, "bath", include_zpe=False))

    # linear_el_vib always multiplies q. Since x=sqrt(2)q, multiply G by sqrt(2).
    g_e = sign_g * g_scale * x0e
    g_k = sign_g * g_scale * x0k
    G_rc_q = np.sqrt(2.0) * (g_e * Pee + g_k * Pkk)

    couplings: List[ModeModeCoupling] = [
        ModeModeCoupling("RC", "bath0", -c0, "x", "x", "RC-bath")
    ]
    for j, t in enumerate(t_list):
        couplings.append(ModeModeCoupling(f"bath{j}", f"bath{j+1}", t, "adag", "a", f"hop+:{j}"))
        couplings.append(ModeModeCoupling(f"bath{j}", f"bath{j+1}", t, "a", "adag", f"hop-:{j}"))

    return PCETModel(
        electronic_hamiltonian=H_el,
        modes=modes,
        electronic_names=["e", "k"],
        linear_el_vib={"RC": G_rc_q},
        mode_mode=couplings,
    )


if __name__ == "__main__":
    # A deliberately tiny version of the generalized PCET model for an exact
    # dense-matrix sanity check of the MPO compiler.
    tiny = example_hbq_pcet_with_ct()
    tiny.modes = [
        BosonMode(m.name, 3, m.omega, m.kind, m.include_zpe, m.anharmonic_q, None)
        for m in tiny.modes[:3]
    ]
    tiny.displacements = np.asarray(tiny.displacements)[:, :3]
    tiny.linear_el_vib = {k: v for k, v in tiny.linear_el_vib.items() if k in {m.name for m in tiny.modes}}
    tiny.mode_mode = [c for c in tiny.mode_mode if c.mode_left in {m.name for m in tiny.modes} and c.mode_right in {m.name for m in tiny.modes}]
    tiny.electronic_mode_mode = []

    report = verify_mpo(tiny)
    print("MPO verification:")
    for key, value in report.items():
        print(f"  {key}: {value}")
