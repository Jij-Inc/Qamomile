from __future__ import annotations
import jijmodeling as jm


class Hamiltonian:
    def __init__(self, expr: jm.Expression, name: str = "") -> Hamiltonian:
        self.hamiltonian = expr
        self.name = name

    def _repr_latex_(self):
        return self.hamiltonian._repr_latex_()
