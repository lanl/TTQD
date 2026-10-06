from collections import defaultdict

import numpy as np

from ttqd.network import TreeTensorNetwork
from ttqd.operator import MPO, mpo_to_ttno_tree
from ttqd.operator import TTNO
from ttqd.solvers import DMRG, TreeTTNODMRG2Site
from ttqd.solvers import TreeTTNOTDVP1Site, TreeTTNOTDVP2Site


def _build_product_z_ttno(ttn):
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    Ws = []
    for i in ttn.nodes:
        parent = ttn.parent.get(i, None)
        deg = len(ttn.children.get(i, ())) + (0 if parent is None else 1)
        W = np.zeros((1,) * deg + (2, 2), dtype=float)
        W[(0,) * deg] = Z
        Ws.append(W)
    return TTNO.from_ttns(ttn, Ws)


def _randomize_ttn(ttn, seed=123):
    rng = np.random.default_rng(seed)
    for i in ttn.nodes:
        shp = ttn.nodes[i].tensor.data.shape
        arr = rng.normal(size=shp) + 1j * rng.normal(size=shp)
        ttn.nodes[i].tensor.data = arr


def _build_excited_ising_tree(L=6, Z=2, maxchi=20):
    from ttqd.models import ising_model

    model = ising_model.TFIModel(L=L, J=1.0, g=1.0, boundary="finite")
    ttn = TreeTensorNetwork.z_ary(N=L, Z=Z, d=2, bond_dim=1)
    for n in ttn.nodes:
        ttn.nodes[n].tensor.data[...] = 0.0
        ttn.nodes[n].tensor.data[..., 0] = 1.0

    H_ttno = mpo_to_ttno_tree(ttn, model.H_mpo)
    dmrg = TreeTTNODMRG2Site(maxiter=40, maxchi=maxchi, tol=1.0e-8, norm_reg=0.0)
    ttn.verbose = 0
    dmrg.kernel(H_ttno, ttn)

    i0 = L // 2
    A = ttn.nodes[i0].tensor.data
    SzB = np.tensordot(model.sigmaz, A, axes=(1, -1))
    ttn.nodes[i0].tensor.data = np.moveaxis(SzB, 0, -1)
    return model, H_ttno, ttn


def _excited_ising_mps_energy(L=6, maxchi=20):
    from ttqd.models import ising_model
    from ttqd.network import init_FM_MPS

    model = ising_model.TFIModel(L=L, J=1.0, g=1.0, boundary="finite")
    psi = init_FM_MPS(model.L, model.d, model.bc)
    dmrg = DMRG(tol=1.0e-10, maxchi=maxchi)
    dmrg.kernel(MPO(model.H_mpo).Ws, psi, tol=1.0e-10)

    i0 = L // 2
    SzB = np.tensordot(model.sigmaz, psi.As[i0], axes=(1, 1))
    psi.As[i0] = np.transpose(SzB, [1, 0, 2])
    return float(np.sum(psi.expval_bonds_from_psi2(model.H_bonds)))


def test_tree_tdvp_1site(L=6):
    ttn = TreeTensorNetwork.z_ary(N=L, Z=2, d=2, bond_dim=2)
    _randomize_ttn(ttn, seed=2026)
    H = _build_product_z_ttno(ttn)

    tdvp = TreeTTNOTDVP1Site(
        dt=0.01,
        tmax=0.2,
        # nsteps=3,
        maxchi=8,
        cutoff=1.0e-12,
        dense_limit=256,
        norm_reg=1.0e-12,
        symmetric=True,
    )
    tdvp.kernel(H, ttn, compute_energy=True)

    assert len(tdvp.times) == 21
    assert len(tdvp.energies) == 21
    assert np.all(np.isfinite(np.asarray(tdvp.energies)))


def test_tree_tdvp_1site_ops_move_center():
    ttn = TreeTensorNetwork.z_ary(N=5, Z=2, d=2, bond_dim=1)
    ttn.verbose = 0
    tdvp = TreeTTNOTDVP1Site(dt=0.01, tmax=0.01)
    tdvp.psi = ttn

    ops = tdvp._tdvp1_ops()
    assert [kind for kind, _, _ in ops[:3]] == ["site", "bond", "site"]

    site_weight = defaultdict(float)
    edge_weight = defaultdict(float)
    for kind, item, weight in ops:
        if kind == "site":
            site_weight[item] += weight
        elif kind == "bond":
            u, v = item
            assert v in tdvp._neighbors(u)
            edge_weight[frozenset({u, v})] += weight

    assert set(site_weight) == set(ttn.nodes)
    assert all(np.isclose(weight, 1.0) for weight in site_weight.values())
    assert set(edge_weight) == {frozenset(edge) for edge in tdvp._tree_edges()}
    assert all(np.isclose(weight, 1.0) for weight in edge_weight.values())


def test_tree_tdvp_2site(L=6):
    ttn = TreeTensorNetwork.z_ary(N=L, Z=2, d=2, bond_dim=2)
    _randomize_ttn(ttn, seed=2027)
    H = _build_product_z_ttno(ttn)

    tdvp = TreeTTNOTDVP2Site(
        dt=0.02,
        tmax=0.2,
        # nsteps=2,
        maxchi=6,
        cutoff=1.0e-10,
        dense_limit=192,
        norm_reg=1.0e-12,
        symmetric=False,
        exact_small_system=False,
    )
    tdvp.kernel(H, ttn, compute_energy=True)

    assert len(tdvp.times) == 11
    assert len(tdvp.energies) == 11
    assert np.all(np.isfinite(np.asarray(tdvp.energies)))


def test_tree_tdvp_matches_mps_excited_energy(L=6):
    ref_energy = _excited_ising_mps_energy(L=L, maxchi=20)

    for cls in (TreeTTNOTDVP1Site, TreeTTNOTDVP2Site):
        _, H_ttno, ttn = _build_excited_ising_tree(L=L, Z=2, maxchi=20)
        tdvp = cls(
            dt=0.02,
            tmax=0.1,
            maxchi=20,
            cutoff=1.0e-10,
            dense_limit=192,
            norm_reg=0.0,
            symmetric=False,
        )
        tdvp.kernel(H_ttno, ttn, compute_energy=True)

        assert np.isclose(tdvp.energies[0], ref_energy, atol=1.0e-8)
        assert np.isclose(tdvp.energies[-1], ref_energy, atol=1.0e-8)


def test_tree_tdvp_2site_true_sweep_matches_mps_excited_energy(L=8):
    ref_energy = _excited_ising_mps_energy(L=L, maxchi=50)
    _, H_ttno, ttn = _build_excited_ising_tree(L=L, Z=2, maxchi=50)

    tdvp = TreeTTNOTDVP2Site(
        dt=0.02,
        tmax=0.06,
        nsteps=3,
        maxchi=50,
        cutoff=1.0e-10,
        dense_limit=192,
        norm_reg=0.0,
        symmetric=False,
        exact_small_system=False,
    )
    tdvp.kernel(H_ttno, ttn, compute_energy=True)

    assert np.isclose(tdvp.energies[0], ref_energy, atol=1.0e-8)
    assert np.isclose(tdvp.energies[-1], ref_energy, atol=1.0e-6)


if __name__ == "__main__":
    test_tree_tdvp_1site()
    test_tree_tdvp_2site()
    test_tree_tdvp_matches_mps_excited_energy()
    test_tree_tdvp_2site_true_sweep_matches_mps_excited_energy()
