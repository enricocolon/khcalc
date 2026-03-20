#!/usr/bin/env python3

from poly import LaurentPoly
from coxeter import Permutation

class HeckeElem:
    def __init__(self, n, terms=None, varset=("q",)):
        self.n = n
        self.varset = tuple(varset)
        self.terms = {} #{perm:laurent}
        if terms: #check for accuracy?
            for key,val in terms.items():
                if isinstance(key, Permutation):
                    perm = key
                else:
                    perm = Permutation(key)
                if len(perm) != n:
                    raise ValueError(f"len({perm}) != {n}")
                if isinstance(val, LaurentPoly):
                    coeff = val
                else:
                    coeff = LaurentPoly.const(val, self.varset)
                if not coeff.is_zero():
                    self.terms[perm] = self.terms.get(perm, LaurentPoly.zero(self.varset)) + coeff
        self._prune()

    def _prune(self): #gotta be a better way to do this
        self.terms = {key: val for key, val in self.terms.items() if not val.is_zero()}

    def __str__(self):
        if not self.terms:
            return "0"

        terms = []
        idperm = Permutation.identity(self.n)
        one = LaurentPoly.one(self.varset)

        for perm, coeff in sorted(self.terms.items(), key=lambda item: item[0].array):
            if perm == idperm:
                basis = "T_id"
            else:
                basis = f"T_{perm.array}"

            if coeff == one:
                piece = basis
            elif coeff == -one:
                piece = "-" + basis
            else:
                piece = f"{coeff}*{basis}"

            terms.append(piece)

        return " + ".join(terms).replace("+ -", "- ")

    def __repr__(self):
       return str(self)

    def copy(self):
       return HeckeElem(self.n, dict(self.terms), self.varset)

    def is_zero(self):
       return not self.terms

    def _check_compatible(self, other):
       if not isinstance(other, HeckeElem):
           raise TypeError(f"{other} must be of type HeckeElem.")
       if self.n != other.n:
           raise ValueError(f"Hecke algebras H_{self.n} and H_{other.n} do not match.")
       if self.varset != other.varset:
           raise ValueError(f"Variable sets {self.varset} and {other.varset} do not match.")

    def __add__(self, other):
        self._check_compatible(other)
        out = dict(self.terms)
        for perm, coeff in other.terms.items():
            out[perm] = out.get(perm, LaurentPoly.zero(self.varset)) + coeff
        return HeckeElem(self.n, out, self.varset)

    def __neg__(self):
        return HeckeElem(
            self.n,
            {perm: -coeff for perm, coeff in self.terms.items()},
            self.varset,
        )

    def __sub__(self, other):
        return self + (-other)

    def scale(self, scalar):
        if not isinstance(scalar, LaurentPoly):
            scalar = LaurentPoly.const(scalar, self.varset)
        return HeckeElem(
            self.n,
            {perm: scalar * coeff for perm, coeff in self.terms.items()},
            self.varset,
        )


    @classmethod
    def zero(cls, n, varset=("q",)):
        return cls(n, {}, varset)

    @classmethod
    def one(cls, n, varset=("q",)):
        return cls(n, {Permutation.identity(n): LaurentPoly.one(varset)}, varset)

    @classmethod
    def tw_basis(cls, n, w, varset=("q",)):
        if not isinstance(w, Permutation):
            w = Permutation(w)
        return cls(n, {w: LaurentPoly.one(varset)}, varset)

    @classmethod
    def simple(cls, n, i, varset=("q",)):
        return cls.tw_basis(n, Permutation.s(n, i), varset)

    def __eq__(self, other):
        if not isinstance(other, HeckeElem):
            return False
        return (
            self.n == other.n
            and self.varset == other.varset
            and self.terms == other.terms
        )

    def right_swap(self, i):
        '''
        input: HeckeElem, index i
        output: HeckeElem(T_s_i) where s_i is the transposition i <-> i+1
        '''
        q = LaurentPoly.variable("q", self.varset)
        qinv = LaurentPoly.variable("q", self.varset, power=-1)

        out = HeckeElem.zero(self.n, self.varset)

        for w, coeff in self.terms.items():
            ws = w.right_swap(i)
            if ws.coxeter() == w.coxeter() + 1:
                out = out + HeckeElem(self.n, {ws: coeff}, self.varset)
            else:
                out = out + HeckeElem(
                    self.n,
                    {ws: coeff, w: (q-qinv) * coeff},
                    self.varset,
                )
        return out

    def multiply_basis_right(self, w, u):
        """
        input: permutations w,u
        output: T_w(T_u) as a HeckeElem.
        """
        out = HeckeElem.tw_basis(self.n, w, self.varset)
        for i in u.reduced_word():
            out = out.right_swap(i)
        return out

    def __mul__(self, other):
        if isinstance(other, LaurentPoly) or isinstance(other, int):
            return self.scale(other)

        if not isinstance(other, HeckeElem):
            return NotImplemented

        self._check_compatible(other)
        out = HeckeElem.zero(self.n, self.varset)

        for w, coeff_w in self.terms.items():
            for u, coeff_u in other.terms.items():
                prod = self.multiply_basis_right(w, u)
                out = out + prod.scale(coeff_w * coeff_u)

        return out

    def __rmul__(self, other):
        if isinstance(other, LaurentPoly) or isinstance(other, int):
            return self.scale(other)
        return NotImplemented
