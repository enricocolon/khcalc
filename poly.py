#!/usr/bin/env python3

'''
poly.py contains all scaffolding for supporting (multivariate) polynomial and Laurent rings over Q (or Z kinda).
'''

from fractions import Fraction

class LaurentPoly:
    def __init__(self, terms=None, varset=("q",), coeff_type=Fraction):
        self.coeff_type = coeff_type
        self.terms = {}
        self.varset = tuple(varset)
        self.nvars = len(self.varset)

        if terms:
            for key,val in terms.items():
                keytup = tuple(key)
                if len(keytup) != self.nvars:
                    raise ValueError(f"len({keytup}) != {self.nvars}")
                coeff = self.coeff_type(val) #hah
                if coeff != 0:
                    self.terms[keytup] =  self.terms.get(keytup, self.coeff_type(0)) + coeff

        self._prune()


    def _prune(self):
        self.terms = {key: val for key, val in self.terms.items() if val != 0}


    def _check_compatible(self, other):
        if not isinstance(other, LaurentPoly):
            raise TypeError(f"{other} must be of type LaurentPoly.")
        if self.varset != other.varset:
            raise ValueError(f"varsets {self.varset} and {other.varset} are not compatible")
        if self.coeff_type != other.coeff_type:
            raise ValueError(f"coeff types {self.coeff_type} and {other.coeff_type} are not compatible")

    @classmethod
    def zero(cls, varset=("q",), coeff_type=Fraction):
        return cls({}, varset, coeff_type)

    @classmethod
    def const(cls, c, varset=("q",), coeff_type=Fraction):
        c = coeff_type(c)
        if c == 0:
            return cls.zero(varset, coeff_type)
        return cls({(0,)*len(varset): coeff_type(c)}, varset, coeff_type)

    @classmethod
    def one(cls, varset=("q",), coeff_type=Fraction):
        return cls.const(1, varset, coeff_type)

    def copy(self):
        return LaurentPoly(dict(self.terms), self.varset, self.coeff_type)

    def is_zero(self):
        return not self.terms

    def __add__(self, other):
        if isinstance(other, (int, Fraction)): #coerce scalars
            other = LaurentPoly.const(other, self.varset, self.coeff_type)
        self._check_compatible(other)
        output = dict(self.terms)
        for key, val in other.terms.items():
            output[key] = output.get(key, self.coeff_type(0)) + val
        return LaurentPoly(output, self.varset, self.coeff_type)

    def __neg__(self):
        return LaurentPoly({key: self.coeff_type(-1)*val for key, val in self.terms.items()}, self.varset, self.coeff_type)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, (int, Fraction)): #coerce scalars
            other = LaurentPoly.const(other, self.varset, self.coeff_type)
        self._check_compatible(other)
        output = {}
        for key, val in self.terms.items():
            for key2, val2 in other.terms.items():
                newkey = tuple(a+b for a,b in zip(key, key2))
                output[newkey] = output.get(newkey, self.coeff_type(0)) + val*val2
        return LaurentPoly(output, self.varset, self.coeff_type)

    def __rmul__(self, other):
        return self * other

    def __eq__(self, other):
        if not isinstance(other, LaurentPoly):
            return False
        if self.varset != other.varset or self.coeff_type != other.coeff_type:
            return False
        return self.terms == other.terms

    def coeff(self, key):
        return self.terms.get(tuple(key), self.coeff_type(0))

    def monomials(self):
        return self.terms.keys()

    @classmethod
    def monomial(cls, exp, c=1, varset=("q",), coeff_type=Fraction):
        exp = tuple(exp)
        varset = tuple(varset)

        if len(exp) != len(varset):
            raise ValueError(f"len({exp}) != {len(varset)}")

        c = coeff_type(c)
        if c == 0:
            return cls.zero(varset, coeff_type)

        return cls({exp: c}, varset, coeff_type)


    @classmethod
    def variable(cls, var, varset=("q",), power=1, coeff_type=Fraction):
        varset = tuple(varset)
        if var not in varset:
            raise ValueError(f"variable {var} not in varset {varset}")
        exp = [0] * len(varset)
        exp[varset.index(var)] = power
        return cls.monomial(tuple(exp), 1, varset, coeff_type)

    def __str__(self):
        if not self.terms:
            return "0"

        terms = []
        for key, val in sorted(self.terms.items()):
            term = []
            for var, exp in zip(self.varset, key):
                if exp == 0:
                    continue
                elif exp == 1:
                    term.append(var)
                else:
                    term.append(f"{var}^{exp}")
            term_str = "*".join(term) if term else "1"

            if term_str == "1":
                terms.append(f"{val}")
            else:
                if val == 1:
                    terms.append(term_str)
                elif val == -1:
                    terms.append(f"-{term_str}")
                else:
                    terms.append(f"{val}*{term_str}")

        return " + ".join(terms).replace("+ -", "- ")

    def __pow__(self, n):
        if not isinstance(n, int):
            raise TypeError("power must be of type int")
        if n < 0:
            raise ValueError("power must be positive for general Laurent polynomial")

        result = LaurentPoly.one(self.varset, self.coeff_type)
        base = self

        while n:
            if n & 1:
                result = result * base
            base = base * base
            n = n // 2
        return result
