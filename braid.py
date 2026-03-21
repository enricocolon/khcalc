#!/usr/bin/env python3

from coxeter import Permutation
from poly import LaurentPoly
from hecke import HeckeElem #which of these do i actually need?

class BraidElem:
    def __init__(self, n, word=()):
        self.n = n
        self.word = tuple(word)
        self.alphabet = set{range(-self.n-1,self.n)}-{0} #alphabet for braid generators/their inverses.
        #i represents (in zero-indexing), a (signed) crossing between strand i-1 and i.

        for letter in self.word:
            if letter not in self.alphabet:
                raise ValueError(f"Invalid letter {letter} in braid word for B_{n}")

    def __repr__(self):
        return f"BraidWord({self.n},{self.word})"

    def __str__(self): #replace with better word printing
        return f"Br_{self.n}({self.word})"

    def __eq__(self, other):
        return isinstance(other, BraidElem) and self.n == other.n and self.word == other.word

    def copy(self):
        return BraidElem(self.n, self.word)

    def __mul__(self, other):
        '''
        input: self, other: braid elements
        output: their concatenation other(self).
        '''
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
