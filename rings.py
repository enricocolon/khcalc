#!/usr/bin/env python3

from fractions import Fraction
from poly import LaurentPoly
from rat import LaurentFrac

class CoeffRing:
    def __init__(self):
        raise NotImplementedError

    def zero(self):
        raise NotImplementedError

    def one(self):
        raise NotImplementedError

    def coerce(self, value):
        raise NotImplementedError

    def __call__(self, value):
        return self.coerce(value)

    def is_zero(self, value):
        return value == self.zero()

    def equal(self, a, b):
        return a == b

    @property
    def is_field(self):
        raise NotImplementedError

class RingElem(CoeffRing):
    def __init__(self, parent: CoeffRing):
        self.parent = parent

class RationalField(CoeffRing):
    def zero(self):
        return Fraction(0)

    def one(self):
        return Fraction(1)

    def coerce(self, value):
        return Fraction(value)

    @property
    def is_field(self):
        return True

    def __repr__(self):
        return "QQ"

class PolynomialRing(CoeffRing):
    def __init__(self, base_ring=CoeffRing, varset=("q",)):
        self.base_ring = base_ring
        self.varset = varset
        self.nvars = len(self.varset)
        self.var_index = {v: i for i, v in enumerate(self.varset)}
        self.gens = None

    def zero(self):
        raise NotImplementedError

class PolynomialElem(RingElem):
    NotImplemented
