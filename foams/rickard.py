#!/usr/bin/env python3

from .boundary import WebBoundary
from .uq import DividedPower, UWord, UTwoMorphism, RickardU2
from .complex import ShiftedUWord, UDirectSum, MorphismMatrix, ChainComplex

def make_rickard_term(a, i, lam_i, s):
    if lam_i <= 0:
        F = DividedPower("F", i, s, a)

        if F.is_zero:
            return None

        E = DividedPower("E", i, -lam_i + s, F.target)

        if E.is_zero:
            return None

        word = UWord(a, (F, E))

    else:
        E = DividedPower("E", i, s, a)

        if E.is_zero:
            return None

        F = DividedPower("F", i, lam_i + s, E.target)

        if F.is_zero:
            return None

        word = UWord(a, (E, F))

    return ShiftedUWord(word, s)

def positive_rickard(n,a,i):
    '''
    Input: a tuple (n,a,i), n determines the sl_n theory, a a boundary color sequence, i the index
    of the first strand in the positive crossing.
    Output: the UNNORMALIZED sl_n Rickard complex associated to the input data, in Kom(Hom_{Kar U_Q}(a,s_i(a))).
    c.f. arXiv:1405.5920v1, (2.42-3).
    '''
    if not isinstance(a, WebBoundary):
        raise TypeError("a must be a WebBoundary")


    a.require_admissible(n)

    if type(i) is not int:
        raise TypeError("i must be an integer")

    if not 0 <= i < a.m - 1:
        raise IndexError(f"{i} is not a valid crossing index for {a}")


    lam_i = a.lam(i)
    source = a
    target = a.s(i)
    C = dict()
    d = dict()
    s = 0
    prev_word = None

    while True:
        word = make_rickard_term(a, i, lam_i, s)

        if word is None or word.word.is_zero_at_rank(n):
            break

        #add_term(word)
        C[s] = UDirectSum([word])
        if prev_word is not None:
            if lam_i <= 0:
                expression = RickardU2("EF",i=i,step=s)
            else:
                expression = RickardU2("FE",i=i,step=s)
            entries = {(0,0): UTwoMorphism(prev_word.word, word.word, expression, q_degree=1)} #ck q-degree??
            d[s-1] = MorphismMatrix(C[s-1], C[s], entries)
            #d[k]:C[k]->C[k+1], disagreeing with the index of d_i in QR (2.42-3).
        s += 1
        prev_word = word

    return ChainComplex(terms=C, differentials=d)
