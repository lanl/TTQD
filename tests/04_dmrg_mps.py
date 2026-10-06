

import numpy as np

from ttqd.network import MPS, init_FM_MPS
from ttqd.models.ising_model import TFIModel
from ttqd.solvers.dmrg import DMRG


def _check_traversal_orders(L=4):
    mps = MPS.rand(L=L, d=2, chi=3, boundary="infinite")
    path = mps._downward_order()
    print("mps.boundary = ", mps.bc)
    print("contraction path is", path)

    # continuous path should begin at the root and end at a leaf
    assert path[0] == mps.root
    assert path[-1] != mps.root
    # every adjacent pair in the path must be connected by an edge
    for a, b in zip(path, path[1:]):
        assert b in mps.children.get(a, ()) or a in mps.children.get(b, ())
    # upward order is simply the reverse of downward continuous path
    assert mps._upward_order() == path[::-1]


def _check_dmrg_mps(L=4):
    # small transverse-field Ising chain where exact energy is known
    model = TFIModel(L=L, J=1.5, g=1.0, bc="finite")
    mpo = model.H_mpo
    psi = init_FM_MPS(model.L, model.d, model.bc)

    dmrg = DMRG(maxiter=5, tol=1e-7, maxchi=10)
    dmrg.kernel(mpo, psi)
    # energy should be finite and reasonable (ferromagnet with J=1,g=1)
    refs = {
       4:  -5.6830199487,
       6:  -8.9534848128,
       8: -12.2620502576,
      10: -15.5892233228,
      12: -18.9253763982,
      14: -22.2657349254,
    }
    assert isinstance(dmrg.energy, float)
    assert dmrg.energy < 0.0
    if L in refs:
        assert abs(dmrg.energy - refs[L]) < 1.e-6

import unittest


class TestDMRGMPS(unittest.TestCase):
    def test_traversal_orders(self):
        _check_traversal_orders(L=4)


    def test_dmrg_mps(self):
        for n in [4, 6, 10]:
            _check_dmrg_mps(L=n)


if __name__ == "__main__":
    unittest.main()
