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

"""

import sys
from ttqd.lib import logger
# base class
class BaseSolver(object):
    r"""
    """
    def __init__(
        self,
        H=None,
        tol=1.e-8,
        **kwargs,
    ):
        self.H = H
        self.tol = tol
        self.verbose = kwargs.get("verbose", 3) # may overwrite it with ttn.verbose
        self.stdout = sys.stdout
        self.wt_kernel = 0.0 # total wall time
        self.wt_sweep = 0.0 # wall time of sweep method
        self.wt_Heff = 0.0  # wall time of constructing effective Hamiltonian
        self.wt_envs = 0.0  # wall time of constructing environments
        self.wt_prediag = 0.0  # wall time of preparing local operators (for diaognalization)
        self.wt_diag = 0.0  # wall time of diaognalization
        self.wt_bond = 0.0  # wall time of splitting/truncating bonds


    def update_local(self):
        r"""Form local Hamiltonian and update the bonds

        """
        raise NotImplementedError('Should be implemented in inherited class')

    def kernel(self, mps): # TBA
        raise NotImplementedError("Shoudl be implemented in inherited class")

    def post_kernel(self):
        raise NotImplementedError('Should be implemented in inherited class')
