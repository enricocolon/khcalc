#!/usr/bin/env python3

from .boundary import WebBoundary

def positive_rickard(n,a,i):
    '''
    Input: a tuple (n,a,i), n determines the sl_n theory, a a boundary color sequence, i the index
    of the first strand in the positive crossing.
    Output: the sl_n Rickard complex associated to the input data, in Kom(Hom_{Kar U_Q}(a,s_i(a))).
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

    #then build the terms
    pass
