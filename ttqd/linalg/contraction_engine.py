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


from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union


# -----------------------------
# Contraction Engine
# -----------------------------
class ContractionEngine:
    """
    Routes einsum contractions through:
      - opt_einsum (fast + path optimized)
      - optionally cotengra (better path optimizer) if installed
    Includes optional expression caching (contract_expression).
    """

    def __init__(
        self,
        backend: Backend,
        prefer: str = "auto",          # "auto"|"opt_einsum"|"backend"
        use_cotengra: bool = True,
        cache_expressions: bool = True,
        cotengra_kwargs: Optional[dict] = None,
    ):
        self.be = backend
        self.cache_expressions = cache_expressions
        self._expr_cache: Dict[Tuple[str, Tuple[Tuple[int, ...], ...], str], Any] = {}

        self.oe = None
        self.optimizer = None

        prefer = prefer.lower()
        if prefer == "backend":
            return

        try:
            import opt_einsum as oe
            self.oe = oe
        except Exception:
            # fallback to backend.einsum
            return

        # Choose optimizer: cotengra HyperOptimizer if available
        self.optimizer = "auto"
        if use_cotengra:
            try:
                import cotengra as ct
                ck = cotengra_kwargs or {}
                self.optimizer = ct.HyperOptimizer(**ck)
            except Exception:
                self.optimizer = "auto"

    def contract(self, equation: str, *operands: Any, optimize: Optional[Any] = None) -> Any:
        """
        Perform einsum contraction. If opt_einsum available use it (with cached expression),
        otherwise fall back to backend.einsum.
        """
        if self.oe is None:
            return self.be.einsum(equation, *operands)

        opt = self.optimizer if optimize is None else optimize

        if not self.cache_expressions:
            return self.oe.contract(equation, *operands, optimize=opt)

        shapes = tuple(tuple(int(s) for s in op.shape) for op in operands)
        dtypestr = str(getattr(operands[0], "dtype", ""))
        key = (equation, shapes, dtypestr)

        expr = self._expr_cache.get(key)
        if expr is None:
            expr = self.oe.contract_expression(equation, *shapes, optimize=opt)
            self._expr_cache[key] = expr
        return expr(*operands)
