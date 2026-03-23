#!/usr/bin/env python3

from poly import LaurentPoly
from hecke import HeckeElem

class BraidElem:
    def __init__(self, n, word=()):
        self.n = n
        self.word = tuple(word)
        self.alphabet = set(range(-self.n+1,self.n))-{0} #alphabet for braid generators/their inverses.
        #i represents (in zero-indexing), a (signed) crossing between strand i-1 and i.

        for letter in self.word:
            if letter not in self.alphabet:
                raise ValueError(f"Invalid letter {letter} in braid word for Br_{n}")

    def __repr__(self):
        return f"BraidElem({self.n}, {self.word})"

    def __str__(self):
        if not self.word:
            return "id"

        terms = []
        for letter in reversed(self.word): #composition convention: (g*f)=g(f)
            if letter > 0:
                terms.append(f"σ_{letter}")
            else:
                terms.append(f"σ_{-letter}^-1")
        return " ".join(terms)


    def _check_compatible(self, other):
        if not isinstance(other, BraidElem):
            raise TypeError(f"{other} must be of type BraidElem.")
        if self.n != other.n:
            raise ValueError(f"Braid elements of different sizes {self.n} and {other.n}")


    def __eq__(self, other): #same as words, not as elements (i.e. rel's not checked)
        return isinstance(other, BraidElem) and self.n == other.n and self.word == other.word

    def copy(self):
        return BraidElem(self.n, self.word)

    def __mul__(self, other):
        '''
        input: self, other: braid elements
        output: their concatenation other(self).
        '''
        self._check_compatible(other)
        word = BraidElem(self.n, self.word+other.word)
        #add word simplification here
        return word

    def inverse(self):
        return BraidElem(self.n, tuple(-x for x in reversed(self.word)))

    def free_reduce(self):
        '''
        Cancel adjacent σ σ^-1's.
        '''
        redword = []
        for letter in self.word:
            if redword and redword[-1] == -letter:
                redword.pop()
            else:
                redword.append(letter)
        return BraidElem(self.n, tuple(redword))

    def writhe(self):
        count = 0
        word = self.word
        for i in word:
            if i > 0:
                count += 1
            else:
                count -= 1
        return count

    def is_freely_trivial(self):
        return len(self.free_reduce().word) == 0

    def to_hecke(self, varset=("q",)):
        """
        input: braid element
        output: its image in the Hecke algebra.
        """
        q = LaurentPoly.variable("q", varset)
        qinv = LaurentPoly.variable("q", varset, power=-1)

        output = HeckeElem.one(self.n, varset)
        one = HeckeElem.one(self.n, varset)

        for letter in self.word:
            i = abs(letter) - 1
            Ti = HeckeElem.simple(self.n, i, varset)

            if letter > 0:
                output = output*Ti
            else:
                output = output*(Ti - one.scale(q-qinv))

        return output
