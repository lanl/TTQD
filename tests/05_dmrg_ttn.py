import numpy as np
from ttqd.lib import logger
from ttqd.operator import mpo_to_ttno_tree

from ttqd.network import MPS, init_FM_MPS
from ttqd.network import TreeTensorNetwork, TTNNode
from ttqd.models.ising_model import TFIModel
from ttqd.solvers.dmrg import DMRG
from ttqd.tensors import Tensor


def _check_ttno_from_ttn(L=4):
    from ttqd.network import TreeTensorNetwork
    from ttqd.operator.ttno import TTNO
    # build a tiny balanced TTN with 5 leaves (9 nodes total)
    ttn = TreeTensorNetwork.z_ary(N=L, Z=2, d=2, bond_dim=1)
    # construct dummy Ws: one 4-index tensor per node
    Ws = [np.zeros((1, 1, 2, 2)) for _ in range(len(ttn.nodes))]
    ttno = TTNO.from_ttns(ttn, Ws)
    # topology must match
    assert ttno.parent == ttn.parent
    assert ttno.children == ttn.children
    assert ttno.root == ttn.root
    assert ttno.L == len(ttn.nodes)
    # phys_dim property should read the 2 from our zero tensors
    assert ttno.phys_dim == 2


def _check_chain_mpo_to_ttno_compact(L=5):
    model = TFIModel(L=L, J=1.5, g=1.0, bc="finite")
    mpo = model.H_mpo.Ws if hasattr(model.H_mpo, "Ws") else model.H_mpo
    ttn = TreeTensorNetwork.z_ary(N=L, Z=1, d=2, bond_dim=1)

    ttno = mpo_to_ttno_tree(ttn, model.H_mpo)
    assert ttno.parent == ttn.parent
    assert ttno.children == ttn.children
    assert ttno.Ws[0].shape == (mpo[0].shape[1], mpo[0].shape[2], mpo[0].shape[3])
    assert ttno.Ws[-1].shape == (mpo[-1].shape[0], mpo[-1].shape[2], mpo[-1].shape[3])
    for i in range(1, L - 1):
        assert ttno.Ws[i].shape == mpo[i].shape


def _check_get_As():
    # construct a simple three-node chain 0-1-2
    tensors = [np.ones((2, 2, 3)), np.ones((2, 2, 3)), np.ones((2, 2, 3))]
    # convert to TTN with explicit parent/children
    nodes = {}
    parent = {0:1, 1:None, 2:1}
    children = {1:(0,2), 0:(), 2:()}
    for idx, A in enumerate(tensors):
        nodes[idx] = ttn = None
    # easier: build z_ary and then patch
    ttn = TreeTensorNetwork.z_ary(N=3, Z=2, d=2, bond_dim=2)
    # override data shapes to create 3-index objects with two virtual legs + physical
    for n in ttn.nodes:
        x = ttn.nodes[n].tensor.data
        ttn.nodes[n].tensor.data = np.random.randn(*x.shape).astype(x.dtype, copy=False)
        # label indices artificially

    # pick nodes 1 and 0 which are connected
    A10 = ttn.get_As(1, 0)
    # shape should equal (R,d,D) with D equal to bond between 1 and 0 (size 2)
    assert A10.shape[-1] == 2
    assert A10.ndim == 3
    # repeating with reversed order should yield same last dimension
    A01 = ttn.get_As(0, 1)
    assert A01.shape[-1] == 2


    L = 8
    ttn = TreeTensorNetwork.z_ary(N=L, Z=2, d=2, bond_dim=4)
    A13 = ttn.get_As(1, 3)
    assert A13.shape == tuple([16, 2, 4])


def _check_ttn_von_neumann_entropies():
    """Bell pair on a 2-node tree: S1=1 (base-2), S2=0."""
    s2 = 1.0 / np.sqrt(2.0)
    A = np.array([[s2, 0.0], [0.0, s2]], dtype=float)  # (bond, phys0)
    B = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)  # (bond, phys1)

    nodes = {
        0: TTNNode(Tensor(A, ("b01", "k0")), phys_inds=("k0",)),
        1: TTNNode(Tensor(B, ("b01", "k1")), phys_inds=("k1",)),
    }
    parent = {0: None, 1: 0}
    children = {0: (1,), 1: ()}
    ttn = TreeTensorNetwork(nodes, parent, children, root=0)

    S0 = float(np.real(ttn.von_neumann_entropy_1body(0, base=2.0)))
    S1 = float(np.real(ttn.entropy_1body(1, base=2.0)))
    S01 = float(np.real(ttn.von_neumann_entropy_2body(0, 1, base=2.0)))

    assert abs(S0 - 1.0) < 1e-10
    assert abs(S1 - 1.0) < 1e-10
    assert abs(S01 - 0.0) < 1e-10


def _check_dmrg_ttn(L=4):

    from ttqd.network import TreeTensorNetwork
    from ttqd.operator.ttno import TTNO

    model = TFIModel(L=L, J=1.5, g=1.0, bc="finite")
    mpo = model.H_mpo
    # psi = init_FM_MPS(model.L, model.d, model.bc)

    # make a binary tree
    ttn = TreeTensorNetwork.z_ary(N=L, Z=2, d=2, bond_dim=1)
    ttn.verbose = 5

    # initialize ttn
    for n in ttn.nodes:
        ttn.nodes[n].tensor.data[...,0] = 1.0
        #print(f"tensor at node{n} has shape :", ttn.nodes[n].tensor.data.shape,
        #      " value: ", ttn.nodes[n].tensor.data)

    dmrg = DMRG(maxiter=5, tol=1e-7, maxchi=10)
    dmrg.kernel(mpo, ttn)


def _check_tree_ttno_dmrg_product_op(L=6):
    from ttqd.network import TreeTensorNetwork
    from ttqd.operator.ttno import TTNO
    from ttqd.solvers import TreeTTNODMRG2Site

    Z = np.array([[1.0, 0.0], [0.0, -1.0]])

    ttn = TreeTensorNetwork.z_ary(N=L, Z=2, d=2, bond_dim=2)
    for n in ttn.nodes:
        shp = ttn.nodes[n].tensor.data.shape
        rng = np.random.default_rng(1234 + int(n))
        ttn.nodes[n].tensor.data = rng.normal(size=shp)

    Ws = []
    for i in ttn.nodes:
        parent = ttn.parent.get(i, None)
        deg = len(ttn.children.get(i, ())) + (0 if parent is None else 1)
        W = np.zeros((1,) * deg + (2, 2), dtype=float)
        W[(0,) * deg] = Z
        Ws.append(W)
    H = TTNO.from_ttns(ttn, Ws)

    solver = TreeTTNODMRG2Site(maxiter=4, maxchi=8, tol=1e-9)
    E0 = solver.kernel(H, ttn, maxiter=4)

    # H = Z \otimes Z \otimes ... has minimum eigenvalue -1.
    assert abs(E0 + 1.0) < 1e-8


def _run_new_tree_dmrg(
    L,
    J,
    g,
    Z=2,
    maxchi=50,
    warmup_sweeps=3,
    one_site_sweeps=0,
    cleanup_sweeps=0,
):
    from ttqd.models import ising_model
    from ttqd.network import TreeTensorNetwork
    from ttqd.solvers import TreeTTNODMRG1Site
    from ttqd.solvers import TreeTTNODMRG2Site

    print(logger.task_title("Test DMRG-TTN", level=0))

    model = ising_model.TFIModel(L=L, J=J, g=g, bc='finite')
    mpo = model.H_mpo

    ttn = TreeTensorNetwork.z_ary(N=L, Z=Z, d=2, bond_dim=1)
    ttn.verbose = 3
    # initialize ttn
    for n in ttn.nodes:
        ttn.nodes[n].tensor.data[...] = 0.0
        ttn.nodes[n].tensor.data[...,0] = 1.0

    # print("ttn nodes:", ttn.nodes)

    H_ttno = mpo_to_ttno_tree(ttn, mpo)

    E0 = None
    if warmup_sweeps > 0:
        dmrg_warm = TreeTTNODMRG2Site(maxiter=warmup_sweeps, maxchi=maxchi, tol=1e-8, norm_reg=0.0)
        E_warm = dmrg_warm.kernel(H_ttno, ttn)
        print("Tree TTNO-DMRG warmup (2-site) energy =", E_warm)
        E0 = E_warm

    if one_site_sweeps > 0:
        dmrg = TreeTTNODMRG2Site(maxiter=one_site_sweeps, maxchi=maxchi, tol=1e-8, norm_reg=0) #1e-10)
        # dmrg = TreeTTNODMRG1Site(maxiter=one_site_sweeps, maxchi=maxchi, tol=1e-8, norm_reg=1e-12)
        E0 = dmrg.kernel(H_ttno, ttn)
        print("Tree TTNO-DMRG energy (1-site) =", E0)

    if cleanup_sweeps > 0:
        dmrg_cleanup = TreeTTNODMRG2Site(
            maxiter=cleanup_sweeps,
            maxchi=maxchi,
            tol=1e-6,
            norm_reg=0.0,
        )
        E0 = dmrg_cleanup.kernel(H_ttno, ttn)
        print("Tree TTNO-DMRG cleanup (2-site) energy =", E0)

    """
    for i in range(L-1):
        for j in range(i+1, L):
            si = ttn.entropy_1body(i, normalize=False)
            sj = ttn.entropy_1body(j, normalize=False)
            sij = ttn.entropy_2body(i, j)
            # print("entropy is ", si)
            # print("entropy-twobody (1,3) is ", sij)
            minfo = sij - si - sj
            print(f"mutual information between {i} and {j}: {minfo:12.7f}")
    """

    return E0



import unittest


class TestDMRGTtn(unittest.TestCase):
    # @unittest.skip("skipping failing dmrg_ttn test until upstream fix")

    def test_ttno_from_ttn(self):
        _check_ttno_from_ttn(L=4)

    def test_chain_mpo_to_ttno_compact(self):
        _check_chain_mpo_to_ttno_compact(L=5)

    def test_dmrg_ttn(self):
        # test_dmrg_ttn(L=14)
        ref_energies = {
            4: -5.683019948657,
            6: -8.953484812753,
            8: -12.26205025761,
            10: -15.5892233228,
        }

        J = 1.5
        g = 1.0

        # 2) TTN
        _check_get_As()

        for L in [4, 6, 8]: #, 10]:
            # test_dmrg_tree(L, J, g, maxchi=50)
            E = _run_new_tree_dmrg(
                L,
                J,
                g,
                Z=2,
                maxchi=20,
                warmup_sweeps=0,
                one_site_sweeps=50,
                cleanup_sweeps=0,
            )
            assert abs(E - ref_energies[L]) < 1.e-7

    def test_ttno_tree_product(self):
        _check_tree_ttno_dmrg_product_op()

    def test_tree_entropy(self):
        _check_ttn_von_neumann_entropies()

if __name__ == "__main__":
    # test_dmrg_ttn(L=8)
    unittest.main()
