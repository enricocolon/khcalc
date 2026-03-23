#!/usr/bin/env python3

from poly import LaurentPoly

class LaurentFrac:
    def __init__(self, num, den):
        if not (isinstance(num, LaurentPoly) and isinstance(den,LaurentPoly)):
            raise TypeError("num and den must be of type LaurentPoly.")
        num._check_compatible(den)
        if den.is_zero():
            raise ZeroDivisionError(f"denominator {den} cannot be zero")
        self.num = num
        self.den = den
        self.varset = self.num.varset
        self.coeff_type = self.num.coeff_type #when over not Q, this will matter


    @classmethod
    def zero(cls, varset):
        return cls(LaurentPoly.zero(varset), LaurentPoly.one(varset))

    @classmethod
    def one(cls, varset):
        return cls(LaurentPoly.one(varset), LaurentPoly.one(varset))

    @classmethod
    def from_poly(cls, poly):
        if not isinstance(poly, LaurentPoly):
            raise TypeError(f"{poly} must be of type LaurentPoly")
        return cls(poly, LaurentPoly.one(poly.varset))

    def is_zero(self):
        return self.num.is_zero()

    def __repr__(self):
        return f"({self.num})/({self.den})"

    def __str__(self):
        if self.den == LaurentPoly.one(self.varset):
            return str(self.num)
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, LaurentFrac):
            return False
        return self.varset == other.varset and self.num*other.den == self.den*other.num

    def copy(self):
        return LaurentFrac(self.num.copy(), self.den.copy())

    def __neg__(self):
        return LaurentFrac(-self.num, self.den)

    def __add__(self, other):
        if not isinstance(other, LaurentFrac):
            other = LaurentFrac.from_poly(other)
            #check if this works for both const and poly
        num = self.num * other.den + other.num * self.den
        den = self.den * other.den
        return LaurentFrac(num, den)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if not isinstance(other, LaurentFrac):
            other = LaurentFrac.from_poly(other)
            #check if this works for both const and poly
        num = self.num * other.num
        den = self.den * other.den
        return LaurentFrac(num, den)

    def __rmul__(self, other):
        if not isinstance(other, LaurentFrac):
            other = LaurentFrac.from_poly(other)
            #check if this works for both const and poly
        return self * other

    def __truediv__(self, other):
        if not isinstance(other, LaurentFrac):
            other = LaurentFrac.from_poly(other)
            #check if this works for both const and poly
        self._check_compatible(other)
        if other.num.is_zero():
            raise ZeroDivisionError(f"Cannot divide by zero {other}")
        num = self.num * other.den
        den = self.den * other.num
        return LaurentFrac(num, den)
