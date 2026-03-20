#!/usr/bin/env python3

'''
Provides support for the algebra of type A Coxeter systems
'''

class Permutation():
    def __init__(self, arr):
        '''
        given a list arr of n items {0,...,n-1}, returns an object representing the element of S_n sending
        i to lst[i].
        '''
        self.array = tuple(arr)
        if set(self.array) != set(range(len(self.array))):
            raise ValueError(f"Invalid permutation {self.array}")

    def __len__(self):
        return len(self.array)

    def __mul__(self, other):
        '''
        given self and other, returns the symmetric group composition other(self).
        '''
        if not isinstance(other, Permutation):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError(f"Permutations of different sizes {len(self)} and {len(other)}")
        return Permutation([self.array[i] for i in other.array])

    @classmethod
    def identity(cls, n):
        return cls(tuple(range(n)))

    @classmethod
    def s(cls,n,i):
        '''
        returns the simple transposition s_i in S_n, which sends i to i+1 and i+1 to i.
        '''
        if i < 0 or i >= n-1:
            raise ValueError(f"Invalid index {i} for s_i in S_{n}")
        arr = list(range(n))
        arr[i], arr[i+1] = arr[i+1], arr[i]
        return cls(arr)

    def __repr__(self):
        return f"Permutation({self.array})"

    def __str__(self):
        return f"s_[{self.array}]"

    def __hash__(self):
        return hash(self.array)

    def __eq__(self, other):
        return isinstance(other, Permutation) and self.array == other.array

    def inverse(self):
        '''
        input: permutation
        output: its inverse in S_|len(permutation)|
        '''
        arr = [0]*len(self)
        for i,j in enumerate(self.array):
            arr[j] = i
        return Permutation(tuple(arr))

    def coxeter(self):
        '''
        input: permutation
        output: the number of inversions/Coxeter length of a permutation.
        '''
        count = 0
        for i in range(len(self)):
            for j in range(i+1, len(self)):
                if self.array[i] > self.array[j]:
                    count += 1
        return count

    def descents(self):
        '''
        input: permutation
        output: a set containing all i such that self(i) > self(i+1).
        '''
        return {i for i in range(len(self)-1) if self.array[i] > self.array[i+1]}

    def right_swap(self, i):
        '''
        input: permutation
        output: permutation(s_i)
        '''
        return self * self.s(len(self), i)

    def left_swap(self, i):
        '''
        input: permutation
        output s_i(permutation)
        '''
        return self.s(len(self), i) * self

    def reduced_word(self):
        """
        input: permutation
        output: a (0-indexed) reduced word [i1, i2, ..., ik] with
        self = s_{i1} s_{i2} ... s_{ik}
        Warning: there is nothing canonical about this reduced word
        """
        arr = list(self.array)
        curr = list(range(len(self)))
        word = []

        for pos in range(len(self)):
            i = arr[pos]
            j = curr.index(i)
            while j > pos:
                curr[j - 1], curr[j] = curr[j], curr[j - 1]
                word.append(j - 1)
                j -= 1

        return word
