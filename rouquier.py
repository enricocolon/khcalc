#!/usr/bin/env python3

from complexes import ChainComplex
from braid import BraidElem

class FormalMorphism: #tk: extend to matrix category
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

    def __hash__(self):
        return hash((self.n, self.sequence))

class FormalDirectSum:
    def __init__(self, data, n=None):
        '''
        given a dictionary with nonnegative integer
        coefficients as values and Bott-Samelsons as
        keys, provides a formal direct sum object
        '''
        if n is None:
            n = next(iter(data)).n if data else 0
        self.data = dict()
        self.n = n

        for key, val in data.items():
            if not isinstance(key, BottSamelson):
                raise TypeError(f'key {key} not of type BottSamelson')
            if not isinstance(val, int) or val < 0:
                raise ValueError(f'value {val} not a nonnegative integer')
            if key.n != self.n:
                raise ValueError(f"all Bott-Samelson objects must have the same n, but {key} has n={key.n} and expected n={self.n}")
            self.data[key] = val

        self._prune()

    def _prune(self):
        self.data = {key: val for key, val in self.data.items() if val != 0}

    @classmethod
    def zero(cls, n):
        return cls({}, n)

    def is_zero(self):
        return not self.data

    def __repr__(self):
        return f"FormalDirectSum(n={self.n}, data={self.data})"

    def __str__(self):
        if self.is_zero():
            return "0"
        summands = []
        for key, val in self.data.items():
            if val == 1:
                summands.append(str(key))
            else:
                summands.append(f"{val}⬝{key}")
        return " ⊕ ".join(summands)

    def __add__(self, other):
        if not isinstance(other, FormalDirectSum):
            return NotImplemented
        if self.n != other.n:
            raise ValueError("Direct sums have incompatible n")
        out = dict(self.data)
        for key, val in other.data.items():
            out[key] = out.get(key, 0) + val
        return FormalDirectSum(out, self.n)

    def tensor(self, other):
        if isinstance(other, BottSamelson):
            other = FormalDirectSum.from_bs(other)
        if not isinstance(other, FormalDirectSum):
            raise TypeError(f"{other} must be FormalDirectSum or BottSamelson")
        if self.n != other.n:
            raise ValueError("Direct sums have incompatible n")

        out = {}
        for left, a in self.data.items():
            for right, b in other.data.items():
                prod = left @ right
                out[prod] = out.get(prod, 0) + a * b
        return FormalDirectSum(out, self.n)

    def __matmul__(self, other):
        return self.tensor(other)

    @classmethod
    def from_bs(cls, bs):
        return cls({bs: 1}, bs.n)

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
    Bi = FormalDirectSum.from_bs(BottSamelson.simple(n, i))
    Id = FormalDirectSum.from_bs(BottSamelson.identity(n))
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

def rouquier_complex(braid, n): #tk: do so after implementing ⊗, after ⊕/matrix morphisms
    NotImplemented
