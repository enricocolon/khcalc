#!/usr/bin/env python3

from fractions import Fraction
from poly import LaurentPoly
from rat import LaurentFrac

class FreeModule:
    def __init__(self, ring, dim, basis={f"r_{i}" for i in range(dim)}):
        self.ring = ring
        self.basis = basis
        self.dim = len(self.basis)

class FreeModuleElem:
    def __init__(self):
        raise NotImplementedError

class Matrix:
    def __init__(self):
        raise NotImplementedError
