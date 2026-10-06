import unittest
import numpy as np
from ttqd.operator import second_quantized_to_mpo, _dense_from_mpo
from ttqd.operator import _dense_hamiltonian_from_integrals
# from ttqd.operator import build_chan_keller_mpo, close_right_boundary
from ttqd.operator import build_compact_second_quantized_mpo


def _n2_pyscf_spin_orbital_integrals(active_spatial_orbitals=2, bond_length_angstrom=1.1):
    try:
        from pyscf import gto, scf, ao2mo
    except ImportError as exc:
        raise ImportError("PySCF is required for the N2 test. Install with: pip install pyscf") from exc

    mol = gto.Mole()
    mol.atom = f"N 0.0 0.0 0.0; N 0.0 0.0 {bond_length_angstrom}"
    mol.unit = "Angstrom"
    mol.basis = "sto3g"
    mol.basis = "ccpvdz"
    mol.spin = 0
    mol.charge = 0
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF RHF did not converge for N2 test.")

    C = mf.mo_coeff
    n_mo = C.shape[1]
    if active_spatial_orbitals > n_mo:
        raise ValueError(
            f"Requested {active_spatial_orbitals} active orbitals but only {n_mo} MOs available."
        )

    hcore_ao = mf.get_hcore()
    h1_mo = C.T @ hcore_ao @ C

    eri_mo = ao2mo.restore(1, ao2mo.kernel(mol, C), n_mo)

    idx = np.arange(active_spatial_orbitals)
    h1_spatial = h1_mo[np.ix_(idx, idx)]
    eri_spatial = eri_mo[np.ix_(idx, idx, idx, idx)]
    if not np.isfinite(h1_spatial).all() or not np.isfinite(eri_spatial).all():
        raise FloatingPointError("Encountered non-finite spatial integrals from PySCF")

    n_spin = 2 * active_spatial_orbitals
    h1 = np.zeros((n_spin, n_spin), dtype=np.complex128)
    h2 = np.zeros((n_spin, n_spin, n_spin, n_spin), dtype=np.complex128)

    for p in range(active_spatial_orbitals):
        for q in range(active_spatial_orbitals):
            h1[2 * p, 2 * q] = h1_spatial[p, q]
            h1[2 * p + 1, 2 * q + 1] = h1_spatial[p, q]

    # Spin-orbital two-electron tensor for operator order a_p^† a_q^† a_r a_s
    # using spin-conserving Coulomb integrals from the RHF spatial MO basis.
    h2[0::2, 0::2, 0::2, 0::2] = eri_spatial  # sigma=0, tau=0
    h2[0::2, 1::2, 1::2, 0::2] = eri_spatial  # sigma=0, tau=1
    h2[1::2, 0::2, 0::2, 1::2] = eri_spatial  # sigma=1, tau=0
    h2[1::2, 1::2, 1::2, 1::2] = eri_spatial  # sigma=1, tau=1
    if not np.isfinite(h1).all() or not np.isfinite(h2).all():
        raise FloatingPointError("Encountered non-finite spin-orbital integrals")

    return h1, h2


class TestMpoTools(unittest.TestCase):

    def test_second_quantized_to_mpo_with_n2(self):
        tol = 1e-12
        try:
            h1, h2 = _n2_pyscf_spin_orbital_integrals(
                active_spatial_orbitals=4,
                bond_length_angstrom=1.1,
            )
        except ImportError:
            self.skipTest("PySCF is not installed")

        self.assertTrue(np.isfinite(h1).all(), msg="h1 contains non-finite values")
        self.assertTrue(np.isfinite(h2).all(), msg="h2 contains non-finite values")
        Ws1 = second_quantized_to_mpo(h1, h2, tol=tol, dtype=np.complex128)
        Ws2, bond_labels = build_compact_second_quantized_mpo(
            h1, h2, tol=tol, dtype=np.complex128
        )
        for k, W in enumerate(Ws1):
             print(f"W[{k}] shape is", W.shape, Ws2[k].shape)

        H_ref = _dense_hamiltonian_from_integrals(h1, h2, tol=tol, dtype=np.complex128)
        self.assertTrue(
            np.isfinite(H_ref).all(),
            msg="H_ref contains non-finite values",
        )
        for Ws in [Ws1, Ws2]:
            H_from_mpo = _dense_from_mpo(Ws, tol=tol, dtype=np.complex128)
            self.assertTrue(
                np.isfinite(H_from_mpo).all(),
                msg="H_from_mpo contains non-finite values",
            )
            err = np.linalg.norm(H_from_mpo - H_ref)
            max_abs = np.max(np.abs(H_from_mpo - H_ref))
            self.assertTrue(
                np.allclose(H_from_mpo, H_ref, atol=1e-9, rtol=1e-9),
                msg=(
                    "N2/PySCF second_quantized_to_mpo test failed: "
                    f"frobenius_err={err:.3e}, max_abs_err={max_abs:.3e}"
                ),
            )

if __name__ == "__main__":
    unittest.main()
