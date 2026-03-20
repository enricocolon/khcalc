#!/usr/bin/env python3

from coxeter import Permutation
from poly import LaurentPoly
from hecke import HeckeElem #which of these do i actually need?

class BraidElem:
    def __init__(self, n, word=()):
        self.n = n
        self.word = tuple(word)

        for letter in self.word:
            if not isinstance(letter, int) or letter < 1 or letter >= n:
                raise ValueError(f"Invalid letter {letter} in braid word for B_{n}")

    def __repr__(self):
        return f"BraidWord({self.n},{self.word})"

    def __str__(self):
        pass

    def __eq__(self):
        pass

    def copy(self):
        pass

    def __mul__(self, other):
        pass

    def inverse(self):
        pass

    def free_reduce(self):
        '''
        Reduce the easy things to reduce (eg adjacent inverses)
        '''
        pass

    def writhe(self):
        pass

    def is_identity(self):
        pass

    def to_hecke(self, varset=("q",)):
        pass
