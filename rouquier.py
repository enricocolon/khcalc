#!/usr/bin/env python3

from complexes import ChainComplex

class FormalMorphism:
    def __init__(self, source, target, name, zero=False):
        self.source = source
        self.target = target
        self.name = name
        self._zero = zero

    @classmethod
    def zero(cls, source, target):
        return cls(source, target, "0", zero=True)

    def is_zero(self):
        return self._zero

    def __matmul__(self, other):
        if not isinstance(other, FormalMorphism):
            return NotImplemented
        if other.target != self.source:
            raise ValueError("Morphisms are not composable")
        if self.is_zero() or other.is_zero():
            return FormalMorphism.zero(other.source, self.target)
        return FormalMorphism(other.source, self.target, f"{self.name}∘{other.name}")

    def __repr__(self):
        return self.name

class BottSamelson:
    def __init__(self, n, sequence):
        '''
        given a natural n and sequence in {1,...,n-1},
        yields the corresponding Bott-Samelson of type
        A_{n-1} (S_n).
        '''
        if not isinstance(n, int):
            raise TypeError(f"{n} not of type int")
        if n <= 1:
            raise ValueError(f"{n}<=1 out of range.")
        self.n = n
        for i in sequence:
            if not isinstance(i, int):
                raise TypeError(f"{i} not of type int")
            if i not in range(1,self.n):
                raise ValueError(f"{i} not in range {{1,...,n-1}}")

        self.sequence = tuple(sequence)

    def __repr__(self):
        return f"BottSamelson(n={self.n}, sequence={self.sequence})"

    def __str__(self):
        if not self.sequence:
            return "BS[]"
        return "BS[" + ",".join(str(i) for i in self.sequence) + "]"

    @classmethod
    def identity(cls, n):
        return cls(n, ())

    @classmethod
    def simple(cls, n, i):
        return cls(n, (i,))

    def __eq__(self, other):
        return isinstance(other, BottSamelson) and self.n==other.n and self.sequence==other.sequence

    def tensor(self, other):
        if not isinstance(other, BottSamelson):
            raise TypeError(f"{other} must be of type BottSamelson")
        if self.n != other.n:
            raise ValueError("Bott-Samelson objects have incompatible n.")
        return BottSamelson(self.n, self.sequence + other.sequence)

    def __matmul__(self, other):
        return self.tensor(other)

    def __len__(self):
        return len(self.sequence)


def crossing_complex(i,n,sign):
    '''
    returns the rouquier complex associated to a
    positive crossing sigma_i for type A_n-1.
    sign is in {1,-1}, and corresponds to pos/neg
    crossings
    '''
    if i not in range(1,n):
        raise ValueError(f"{i} not in range {{1,...,n-1}}")
    if sign not in {1,-1}:
        raise ValueError(f"sign {sign} is invalid")
    Bi = BottSamelson.simple(n, i)
    Id = BottSamelson.identity(n)
    d = FormalMorphism(Bi, Id, f"u_{i}")
    dd = FormalMorphism(Id, Bi, f"v_{i}")

    if sign == -1:
        return ChainComplex(objects={0: Bi, 1: Id},differentials={0: d})
    if sign == 1:
        return ChainComplex(objects={-1: Id, 0: Bi},differentials={-1: dd})

def positive_crossing_complex(i,n):
    return crossing_complex(i,n,1)

def negative_crossing_complex(i,n):
    return crossing_complex(i,n,-1)
