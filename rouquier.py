#!/usr/bin/env python3

from complexes import ChainComplex
from braid import BraidElem

class FormalMorphism: #tk: extend to matrix category
    def __init__(self, source, target, terms=None):
        self.source = source
        self.target = target
        self.terms = tuple(terms) if terms is not None else tuple()

    @classmethod
    def zero(cls, source, target):
        return cls(source, target, ())

    @classmethod
    def named(cls, source, target, name):
        return cls(source, target, (name,))

    def is_zero(self):
        return len(self.terms) == 0

    def __add__(self, other):
            if not isinstance(other, FormalMorphism):
                return NotImplemented
            if self.source != other.source or self.target != other.target:
                raise ValueError("Can only add morphisms with same source and target")
            return FormalMorphism(self.source, self.target, self.terms + other.terms)

    def __matmul__(self, other):
        if not isinstance(other, FormalMorphism):
            return NotImplemented
        if other.target != self.source:
            raise ValueError("Morphisms are not composable")
        if self.is_zero() or other.is_zero():
            return FormalMorphism.zero(other.source, self.target)
        out = []
        for a in self.terms:
            for b in other.terms:
                out.append(f"{a}∘{b}")
        return FormalMorphism(other.source, self.target, tuple(out))

    #TK: tensor product. mixing that up with composition.


    def __repr__(self):
        if self.is_zero():
            return "0"
        return " + ".join(self.terms)

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

    def expanded_summands(self):
        out = []
        for key, val in self.data.items():
            out.extend([key]*val)
        return tuple(out)

    def __eq__(self, other):
        return isinstance(other,FormalDirectSum) and self.n==other.n and self.data==other.data


class FormalMatrixMorphism:
    def __init__(self, source, target, entries=None):
        '''
        A matrix of FormalMorphisms from source to target.
        source, target: FormalMorphism
        entries: dict[(row,col)]= FormalMorphism
        '''
        if not isinstance(source, FormalDirectSum):
            raise TypeError(f"{source} must be of type FormalDirectSum")
        if not isinstance(target, FormalDirectSum):
            raise TypeError(f"{target} must be of type FormalDirectSum")
        if source.n != target.n:
            raise ValueError(f"source and target have incompatible n: {self.source.n} vs {self.target.n}")
        self.source = source
        self.target = target
        self.n = self.source.n
        self.rows = self.target.expanded_summands()
        self.cols = self.source.expanded_summands()
        if entries is None:
            entries = dict()
        self.entries = dict()
        for key, val in entries.items():
            if not (isinstance(key,tuple) or len(key) == 2):
                raise ValueError(f"key {key} must be a tuple of (colIndex, rowIndex)")
            i, j = key
            if not isinstance(i,int) or not isinstance(j, int):
                raise ValueError(f"key {key} must be a tuple of integers (colIndex, rowIndex)")
            if i < 0 or i>=len(self.rows):
                raise ValueError(f"row index {i} out of range for source with {len(self.rows)} summands")
            if j < 0 or j>=len(self.cols):
                raise ValueError(f"column index {j} out of range for target with {len(self.cols)} summands")
            if not isinstance(val, FormalMorphism):
                raise TypeError(f"value {val} must be of type FormalMorphism")
            if val.source != self.cols[j]:
                raise ValueError(f"entry {key} has source {val.source}, expected {self.cols[j]}")
            if val.target != self.rows[i]:
                raise ValueError(f"entry {key} has target {val.target}, expected {self.rows[i]}")
            if not val.is_zero():
                curr = self.entries.get((i,j),FormalMorphism.zero(self.cols[j], self.rows[i]))
                self.entries[(i,j)] = curr + val

        self._prune()

    def _prune(self):
        self.entries = {key: val for key, val in self.entries.items() if not val.is_zero()}

    @classmethod
    def zero(cls, source, target):
        return cls(source, target, dict())

    @classmethod
    def identity(cls, obj):
        if not isinstance(obj, FormalDirectSum):
            raise TypeError(f'{obj} must be of type FormalDirectSum')

        summands = obj.expanded_summands()
        entries = dict()

        for i, summand in enumerate(summands):
            entries[(i,i)] = FormalMorphism(summand, summand, f"id_{summand}")

        return cls(obj, obj, entries)

    def is_zero(self):
        return not self.entries

    def copy(self):
        return FormalMatrixMorphism(self.source, self.target, dict(self.entries))

    def __add__(self, other):
        if not isinstance(other, FormalMatrixMorphism):
            return NotImplemented
        if self.source != other.source or self.target != other.target:
            raise ValueError("Source and target of matrix morphisms must match for addition")

        out = dict(self.entries)

        rows = self.source.expanded_summands()
        cols = self.target.expanded_summands()

        for key, val in other.entries.items():
            i,j = key
            existing = out.get(key, FormalMorphism.zero(rows[j],cols[i]))
            out[key] = existing + val

        return FormalMatrixMorphism(self.source, self.target, out)

    def __neg__(self):
        neg_entries = dict()
        for key, val in self.entries.items():
            if val.is_zero():
                continue
            neg_terms = tuple("-" + t for t in val.terms)
            neg_entries[key] = FormalMorphism(val.source, val.target, neg_terms)
        return FormalMatrixMorphism(self.source, self.target, neg_entries)

    def __sub__(self, other):
        return self + (-other)

    def __matmul__(self, other):
        if not isinstance(other, FormalMatrixMorphism):
            return NotImplemented
        if other.target != self.source:
            raise ValueError("Incompatible source/target for composition")
        out = dict()
        rows = other.source.expanded_summands()
        cols = self.target.expanded_summands()

        for (i,k), val1 in self.entries.items():
            for (j,i2), val2 in other.entries.items():
                if i != i2:
                    continue
                existing = out.get((j,k), FormalMorphism.zero(rows[j], cols[k]))
                out[(j,k)] = existing + (val1 @ val2)

        return FormalMatrixMorphism(other.source, self.target, out)

    def __repr__(self):
        return f"FormalMatrixMorphism(source={self.source}, target={self.target}, entries={self.entries})"

    def __str__(self):
        if self.is_zero():
            return "0"
        pieces = []
        for (i,j), val in self.entries.items():
            pieces.append(f"({val})[{i},{j}]")
        return "\n".join(pieces)


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

    bs_i = Bi.expanded_summands()[0]
    bs_id = Id.expanded_summands()[0]
    if sign == -1:
        d = FormalMatrixMorphism(Bi,Id,{(0,0):FormalMorphism.named(bs_i,bs_id, f"u_{i}")})
        return ChainComplex(objects={0: Bi, 1: Id},differentials={0: d})
    if sign == 1:
        d = FormalMatrixMorphism(Id,Bi,{(0,0):FormalMorphism.named(bs_id,bs_i, f"v_{i}")})
        return ChainComplex(objects={-1: Id, 0: Bi},differentials={-1: d})

def tensor_complexes(c1, c2):
    '''
    tensor product of complexes of ⊕BS's.
    '''
    degs1 = c1.degrees()
    degs2 = c2.degrees()

    obs = dict()
    diffls = dict()

    for a in degs1:
        for b in degs2:
            k = a+b
            summand = c1.objects[a]@c2.objects[b]

            if k in obs:
                obs[k] += summand
            else:
                obs[k] = summand

    for k in sorted(obs.keys()):
        if k+1 not in obs:
            continue
        source = obs[k]
        target = obs[k+1]
        entries = {}
        for a in degs1:
            for b in degs2:
                if a+b!=k:
                    continue
                source_summand = c1.objects[a]@c2.objects[b]
                source_length = len(source_summand.expanded_summands())

                #d_c1
                if a in c1.differentials:
                    d1 = c1.differentials[a]

                    for (i,j), f in d1.entries.items():
                        key = (i,j)
                        if key in entries:
                            entries[key] += f @ FormalMorphism.identity(c2.objects[b])
                        else:
                            entries[key] = f @ FormalMorphism.identity(c2.objects[b])
                #d_c2
                if b in c2.differentials:
                    d2 = c2.differentials[b]

                    for (i,j), f in d2.entries.items():
                        if a % 2:
                            signed = FormalMorphism.identity(c1.objects[a])@FormalMorphism(d2.source, d2.target, tuple("-"+t for t in f.terms))
                        else:
                            signed = FormalMorphism.identity(c1.objects[a])@f

                        key = (i+source_length, j+source_length)
                        if key in entries:
                            entries[key] += signed
                        else:
                            entries[key] = signed

        diffls[k] = FormalMatrixMorphism(source, target, entries)
    return ChainComplex(objects=obs, differentials=diffls)


def positive_crossing_complex(i,n):
    return crossing_complex(i,n,1)

def negative_crossing_complex(i,n):
    return crossing_complex(i,n,-1)

def rouquier_complex(braid, n): #tk: do so after implementing ⊗, after ⊕/matrix morphisms
    NotImplemented
