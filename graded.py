#!/usr/bin/env python3

class Generator:
    def __init__(self, name, qdeg=0, hdeg=0):
        self.name = name
        self.q = qdeg
        self.h = hdeg

    def qshift(self, q):
        if not isinstance(q, int):
            raise TypeError(f"{q} must be of type int")
        return Generator(self.name, self.q+q, self.h)

    def hshift(self, h):
        if not isinstance(h, int):
            raise TypeError(f"{h} must be of type int")
        return Generator(self.name, self.q, self.h+h)

    def __eq__(self, other):
        if not isinstance(other, Generator):
            return False
        return self.name == other.name and self.q == other.q and self.h == other.h

    def __hash__(self):
        return hash((self.name,self.q,self.h))

class GradedFreeModule:
    def __init__(self, ring, generators):
        self.ring = ring
        for g in generators:
            if not isinstance(g, Generator):
                raise TypeError(f"{g} must be of type Generator")
        self.generators = tuple(generators)
        self.generator_set = set(self.generators)

class FreeModuleElem:
    def __init__(self, module, terms=None):
        self.module = module
        self.ring = module.ring
        self.terms = {}

        if terms:
            for gen, coeff in terms.items():
                if gen not in self.module.generator_set:
                    raise ValueError(f"{gen} is not a generator of this module")
                c = self.ring(coeff)
                if c != 0:
                    self.terms[gen] = self.terms.get(gen, self.ring(0)) + c

        self._prune()

    def _prune(self):
        self.terms = {g: c for g, c in self.terms.items() if c != 0}

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def copy(self):
        return FreeModuleElem(self.module, dict(self.terms))

    def is_zero(self):
        return not self.terms

    def __add__(self, other):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError
