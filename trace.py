#!/usr/bin/env python3

from coxeter import Permutation
from braid import BraidElem
from hecke import HeckeElem
from poly import LaurentPoly

def ocneanu_trace(n, hecke_elem):
    '''
    input: n, a hecke element with terms in S_n and varset ("q", "a")
    output: the trace, a Laurent polynomial in q and a
    '''
    if n < 0:
        raise ValueError(f"H_{n} not a valid Hecke algebra")
    if n == 0:
        return LaurentPoly.one(("q", "a"))
    else:
        term = 0
        new_hecke = 0
        return term*ocneanu_trace(n-1, new_hecke)



#future implementation of the ocneanu trace and renormalization so that we can recover homfly
