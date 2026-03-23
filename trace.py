#!/usr/bin/env python3

from hecke import HeckeElem
from poly import LaurentPoly
from rat import LaurentFrac
from fractions import Fraction

def trace_recursion(varset=("q", "a"), coeff_type=Fraction):
    ct = coeff_type
    q = LaurentPoly.variable("q", varset, coeff_type=ct)
    qinv = LaurentPoly.variable("q", varset, power=-1, coeff_type=ct)
    a = LaurentPoly.variable("a", varset, coeff_type=ct)
    ainv = LaurentPoly.variable("a", varset, power=-1, coeff_type=ct)
    return LaurentFrac(a - ainv, q - qinv)

def trace_tw(n,w, varset=("q","a"), coeff_type=Fraction):
    '''
    recursively obtain the Ocneanu trace on tw_basis elements
    '''
    if n == 1:
        return LaurentFrac.one(varset)

    u, pos = parabolic_factor(w)

    if pos == n-1:
        term = trace_recursion(varset)
        new_hecke = HeckeElem.tw_basis(n-1,u,varset)
    else:
        a = LaurentPoly.variable("a",varset)
        term = LaurentFrac.from_poly(a)
        new_hecke = HeckeElem.tw_basis(n-1,u,varset)
        for i in range(n-3, pos-1, -1):
            new_hecke *= HeckeElem.simple(n-1, i, varset)
    return term*trace_tw(n-1, new_hecke, varset, coeff_type)

def ocneanu_trace(n, hecke_elem):
    '''
    input: n, a hecke element with terms in S_n and varset ("q", "a")
    output: the trace, a Laurent fraction in q and a
    '''
    if n < 1:
        raise ValueError(f"H_{n} not a valid Hecke algebra for trace")
    if n == 1:
        i = Permutation.identity(1)
        coeff = hecke_elem.terms.get(id1, LaurentPoly.zero(("q","a")))
        return LaurentFrac.from_poly(coeff)
    output = LaurentFrac.zero(("q","a"))
    for w, coeff in hecke_elem.terms.items():
        output += LaurentFrac.from_poly(coeff)*trace_tw(n,w, ("q","a"))
    return output

def homfly_from_braid(braid):
    pass
