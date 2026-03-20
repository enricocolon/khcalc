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
                    coeff = LaurentPoly.const(val, varset)
                if not coeff.is_zero():
                    self.terms[perm] = self.terms.get(perm, LaurentPoly.zero(varset)) + coeff
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

    #tk: methods!
