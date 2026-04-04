#!/usr/bin/env python3

from fractions import Fraction
from poly import LaurentPoly
from rat import LaurentFrac

class CoeffRing:
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

class RingElem:
    def __init__(self, parent: CoeffRing):
        self.parent = parent

class RationalField(CoeffRing):
    coeff_type = Fraction

    def zero(self):
        return Fraction(0)

    def one(self):
        return Fraction(1)

    def coerce(self, value):
        return Fraction(value)

    def is_zero(self, value):
        return value == 0

    @property
    def is_field(self):
        return True

    def __repr__(self):
        return "QQ"

QQ = RationalField()

class LaurentRing(CoeffRing): #check for completion?
    def __init__(self, base_ring=None, varset=("q",)):
        if base_ring is None:
            base_ring = QQ
        self.base_ring = base_ring
        self.varset = varset
        self.nvars = len(self.varset)
        self.var_index = {v: i for i, v in enumerate(self.varset)}
        self.gens = tuple(self._make_generator(i) for i in range(self.nvars))

    def _make_generator(self, i):
        exp = tuple(1 if j == i else 0 for j in range(self.nvars))
        return LaurentPoly(
            {exp: self.base_ring.one()},
            self.varset,
            Fraction #unsure if necessary/will generalize later
        )

    def zero(self):
        return LaurentPoly.zero(self.varset, Fraction)

    def one(self):
        return LaurentPoly.one(self.varset, Fraction)

    def _validate_poly(self, value):
        if isinstance(value, LaurentPoly):
            if value.varset != self.varset:
                raise ValueError(f"varsets {value.varset} and {self.varset} do not agree.")
            if value.coeff_type != Fraction:
                raise ValueError(f"coeff types {value.coeff_type} and {Fraction} do not agree.")

    def coerce(self, value):
        if isinstance(value, LaurentPoly):
            self._validate_poly(value)
            return value
        c = self.base_ring.coerce(value)
        return LaurentPoly.const(c, self.varset, Fraction)

    def __repr__(self):
        gens = ", ".join(f"{v}^±1" for v in self.varset)
        return f"{self.base_ring}[{gens}]"


class PolynomialRing(LaurentRing):
    def _validate_poly(self, value):
        super()._validate_poly(value)
        for exp in value.terms:
            if any(i < 0 for i in exp):
                raise ValueError("polynomial cannot have negative exponents")

    def __repr__(self):
        gens = ", ".join(f"{v}" for v in self.varset)
        return f"{self.base_ring}[{gens}]"
