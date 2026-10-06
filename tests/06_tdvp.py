
import numpy as np
from ttqd.lib import logger
from ttqd.operator import mpo_to_ttno_tree
from ttqd.operator import MPO
import copy
import unittest

def run_tdvp_mps_demo(L, J, g, maxchi=50, n_sweep=1):
    from ttqd.models import ising_model
    from ttqd.network import init_FM_MPS, MPS
    from ttqd.solvers import DMRG
    from ttqd.solvers import TDVP1, TDVP2

    model = ising_model.TFIModel(L=L, J=J, g=g, boundary='finite')
    mpo = model.H_mpo
    Hbonds = model.H_bonds
    psi = init_FM_MPS(model.L, model.d, model.bc)

    mpo = MPO(mpo)

    dmrg = DMRG(tol=1.e-10, maxchi=maxchi)
    dmrg.kernel(mpo.Ws, psi, tol=1.e-10)

    E = dmrg.energy

    i0 = L // 2
    print("E = ", E)
    # print("psi.As[i0] before excitation: ", psi.As[i0])

    SzB = np.tensordot(model.sigmaz, psi.As[i0], axes=(1, 1))  # i [i*], vL [i] vR
    psi.As[i0] = np.transpose(SzB, [1, 0, 2])  # vL i vR
    # print("psi.As[i0] after excitation: ", psi.As[i0])
    # for i in range(L):
    #     print(f"psi.Ss[{i}] = ", psi.Ss[i])

    # Enew = np.sum(psi.expval_bonds(Hbonds))
    # print(f"New energy after excitation on {i0} site is:", Enew)

    Enew = np.sum(psi.expval_bonds_from_psi2(Hbonds))
    print(f"E after excitation on {i0} site is:", Enew)

    psi_save = copy.copy(psi)

    # 2) construct 1/2-site TDVP object
    tmax = 1.0
    dt = 0.02
    if n_sweep == 1:
        # 1-site TDVP object
        tdvp = TDVP1(mpo, psi, tmax=tmax, dt=dt, maxchi=maxchi, Hbonds=Hbonds)
    else:
        # two-site TDVP
        tdvp = TDVP2(mpo, psi, tmax=tmax, dt=dt, maxchi=maxchi, Hbonds=Hbonds)

    tdvp.dump_flags()
    S = [psi.entanglement_entropy()]
    print("initial entropy: \n", S)

    # steping
    tdvp.kernel()


def run_tdvp_tree_demo(L, J, g, Z=2, maxchi=50, n_sweep=1):
    from ttqd.models import ising_model
    from ttqd.network import TreeTensorNetwork
    from ttqd.solvers import TreeTTNODMRG1Site
    from ttqd.solvers import TreeTTNODMRG2Site
    from ttqd.solvers import TreeTTNOTDVP1Site, TreeTTNOTDVP2Site

    print(logger.task_title("Test DMRG-TTN", level=0))

    model = ising_model.TFIModel(L=L, J=J, g=g, bc='finite')
    mpo = model.H_mpo

    ttn = TreeTensorNetwork.z_ary(N=L, Z=Z, d=2, bond_dim=1)
    # fig = ttn.visualize(show=True)

    # initialize ttn
    for n in ttn.nodes:
        ttn.nodes[n].tensor.data[...] = 0.0
        ttn.nodes[n].tensor.data[...,0] = 1.0

    # print("ttn nodes:", ttn.nodes)

    H_ttno = mpo_to_ttno_tree(ttn, mpo)

    E0 = None
    dmrg = TreeTTNODMRG2Site(maxiter=50, maxchi=maxchi, tol=1e-6, norm_reg=1.e-6,
                            exact_small_syste=False) #1e-10)
    ttn.verbose = 4
    E0 = dmrg.kernel(H_ttno, ttn)
    print("Tree TTNO-DMRG energy (1-site) =", E0)


    # initial state for dynamics
    i0 = L // 2
    print(f"size of nodes[{i0}]: {ttn.nodes[i0].tensor.data.shape}")

    # SzB = np.tensordot(model.sigmaz, ttn.nodes[i0].tensor.data, axes=(1, -1))
    A = ttn.nodes[i0].tensor.data
    SzB = np.tensordot(model.sigmaz, A, axes=(1, -1))
    SzB = np.moveaxis(SzB, 0, -1)
    ttn.nodes[i0].tensor.data = SzB

    tdvp = TreeTTNOTDVP2Site(
    # tdvp = TreeTTNOTDVP1Site(
        dt=0.02,
        tmax=1.0,
        # nsteps=2,
        maxchi=maxchi,
        cutoff=1.0e-10,
        dense_limit=192,
        norm_reg=0.e0,
        symmetric=True,
        # canonicalize_projector=False,
        energy_from_updates=True,
    )

    tdvp.kernel(H_ttno, ttn, compute_energy=True)


def _mps_to_chain_ttn(mps):
    from ttqd.network import TreeTensorNetwork, TTNNode
    from ttqd.tensors import Tensor

    L = len(mps.As)
    nodes = {}
    parent = {}
    children = {}
    for i, A in enumerate(mps.As):
        pi = f"k{i}"
        if i == 0:
            data = np.transpose(np.squeeze(np.array(A, copy=True), axis=0), (1, 0))
            inds = (f"b{i}", pi)
        elif i == L - 1:
            data = np.squeeze(np.array(A, copy=True), axis=2)
            inds = (f"b{i-1}", pi)
        else:
            data = np.transpose(np.array(A, copy=True), (0, 2, 1))
            inds = (f"b{i-1}", f"b{i}", pi)
        nodes[i] = TTNNode(Tensor(data, inds), phys_inds=(pi,))
        parent[i] = i - 1 if i > 0 else None
        children[i] = (i + 1,) if i < L - 1 else ()
    return TreeTensorNetwork(nodes, parent, children, root=0)


def _ttn_expval_1body(ttn, op, i):
    rho = ttn.make_rdm1(i, normalize=True)
    return float(np.real(np.einsum("pq,qp->", op, rho)))


def _check_tree_tdvp_matches_mps_dynamics():
    from ttqd.models import ising_model
    from ttqd.network import init_FM_MPS
    from ttqd.operator import MPO, mpo_to_ttno_tree
    from ttqd.solvers import DMRG, TDVP1, TDVP2
    from ttqd.solvers import TreeTTNOTDVP1Site, TreeTTNOTDVP2Site

    L = 4
    maxchi = 16
    dt = 0.03
    nsteps = 3

    model = ising_model.TFIModel(L=L, J=1.0, g=1.0, boundary="finite")
    mpo = MPO(model.H_mpo)
    psi0 = init_FM_MPS(model.L, model.d, model.bc)
    dmrg = DMRG(tol=1.0e-10, maxchi=maxchi)
    dmrg.kernel(mpo.Ws, psi0, tol=1.0e-10)

    i0 = L // 2
    SzB = np.tensordot(model.sigmaz, psi0.As[i0], axes=(1, 1))
    psi0.As[i0] = np.transpose(SzB, (1, 0, 2))

    for mps_cls, tree_cls, atol in (
        (TDVP1, TreeTTNOTDVP1Site, 7.0e-3),
        (TDVP2, TreeTTNOTDVP2Site, 3.0e-2),
    ):
        psi_mps = psi0.copy()
        psi_mps.Ss = [np.array(s, copy=True) for s in psi0.Ss]
        ttn = _mps_to_chain_ttn(psi0)
        # fig = ttn.visualize(show=True)
        H_ttno = mpo_to_ttno_tree(ttn, mpo)

        obs_mps = [float(np.real(psi_mps.expval_1body(model.sigmaz, i0)))]
        obs_ttn = [_ttn_expval_1body(ttn, model.sigmaz, i0)]

        mps_tdvp = mps_cls(mpo, psi_mps, dt=dt, tmax=nsteps * dt, maxchi=maxchi, Hbonds=model.H_bonds)
        tree_tdvp = tree_cls(
            dt=dt,
            tmax=nsteps * dt,
            nsteps=nsteps,
            maxchi=maxchi,
            cutoff=1.0e-12,
            norm_reg=1.0e-12,
            symmetric=False,
            dense_limit=256,
            auto_exact_dim_limit=0,
        )

        def _callback(_step, _time, psi_state):
            obs_ttn.append(_ttn_expval_1body(psi_state, model.sigmaz, i0))

        for _ in range(nsteps):
            mps_tdvp.step(psi_mps)
            obs_mps.append(float(np.real(psi_mps.expval_1body(model.sigmaz, i0))))
        tree_tdvp.kernel(H_ttno, ttn, compute_energy=False, callback=_callback)

        obs_mps = np.asarray(obs_mps)
        obs_ttn = np.asarray(obs_ttn)
        assert obs_mps.shape == obs_ttn.shape
        assert np.max(np.abs(obs_mps - obs_ttn)) < atol


class TestTDVP(unittest.TestCase):
    def test_tree_tdvp_matches_mps_dynamics(self):
        _check_tree_tdvp_matches_mps_dynamics()


if __name__ == "__main__":
    unittest.main()
