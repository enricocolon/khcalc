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

class SparseMatrix:
    def __init__(self, ring, data):
        '''
        input:
            ring: (int, fraction, LaurentPoly, LaurentFrac)
            data: A dictionary whose keys are pairs (i,j) of nonnegative ints and values are elements of ring in
            the entry (i,j).
        output: a SparseMatrix object
        '''
        self.ring = ring
        self.data = dict()

        for key, val in data.items():
            if not (isinstance(key, tuple) and len(key) == 2 and all(isinstance(k, int) and k >= 0 for k in key)):
                raise ValueError(f"{key} is not a valid key. Keys must be pairs of nonnegative integers.")
            if not isinstance(val, ring):
                raise ValueError(f"{val} is not an element of {ring}.")
            if val != ring(0):
                self.data[key] = val

        self._prune()

    def _prune(self):
        self.data = {key: val for key, val in self.data.items() if val != self.ring(0)}
