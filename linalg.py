#!/usr/bin/env python3

class FreeModule:
    def __init__(self, ring, basis=None, dim=None):
        self.ring = ring
        if basis is not None and dim is not None:
            if len(basis) != dim:
                 raise ValueError("Length of basis must match dim.")
        if basis is not None:
            self.basis = tuple(basis)
            self.dim = len(self.basis)
        elif dim is not None:
            self.basis = tuple(f"e_{i}" for i in range(dim))
            self.dim = dim
        else:
            raise ValueError("Must specify either basis or dim.")

        self.index = {e: i for i, e in enumerate(self.basis)}

    def zero(self):
        return FreeModuleElem(self, {}) #Not Implemented

    def elem(self, data):
        return FreeModuleElem(self, data) #Not Implemented

    def basis_vector(self, e):
        if e not in self.basis:
            raise ValueError(f"{e} is not a basis element.")
        return FreeModuleElem(self, {e: self.ring.one()}) #make sure this works

    def __repr__(self):
        return f"FreeModule({self.ring}, basis={self.basis})"


class FreeModuleElem:
    def __init__(self, parent, data=None):
        self.parent = parent
        self.data = dict()

        if data is None:
            data = dict()

        for e, coeff in data.items():
            if e not in self.parent.index:
                raise ValueError(f"{e} is not a basis element of the parent module.")
            ccoeff = self.parent.ring.coerce(coeff)
            if not self.parent.ring.is_zero(ccoeff):
                self.data[e] = self.data.get(e, self.parent.ring.zero()) + ccoeff

        self._prune()

    def _prune(self):
        self.data = {e: coeff for e, coeff in self.data.items() if not self.parent.ring.is_zero(coeff)}

    def copy(self):
        return FreeModuleElem(self.parent, dict(self.data))

    def is_zero(self):
        return not self.data

    def __add__(self, other):
        if not isinstance(other, FreeModuleElem):
            return NotImplemented
        if self.parent is not other.parent:
            raise ValueError("Cannot add elements of different free modules")

        out = dict(self.data)
        for e, c in other.data.items():
            if e in out:
                out[e] = out[e] + c
            else:
                out[e] = c
        return FreeModuleElem(self.parent, out)

    def __radd__(self, other):
        if other == 0:
            return self
        return NotImplemented

    def __neg__(self):
        return FreeModuleElem(self.parent, {e: -c for e, c in self.data.items()})

    def __sub__(self, other):
        return self + (-other)

    def __rmul__(self, scalar):
        s = self.parent.ring.coerce(scalar)
        return FreeModuleElem(
            self.parent,
            {e: s * c for e, c in self.data.items()}
        )

    def __mul__(self, scalar):
        return self.__rmul__(scalar)

    def __eq__(self, other):
        if not isinstance(other, FreeModuleElem):
            return False
        return self.parent is other.parent and self.data == other.data

    def __repr__(self):
        if not self.data:
            return "0"

        pieces = []
        one = self.parent.ring.one()
        minus_one = -one

        for e in self.parent.basis:
            if e not in self.data:
                continue

            c = self.data[e]

            if c == one:
                pieces.append(f"{e}")
            elif c == minus_one:
                pieces.append(f"-{e}")
            else:
                coeff_str = str(c)

                # handle parentheses
                if (" + " in coeff_str) or (" - " in coeff_str):
                    coeff_str = f"({coeff_str})"

                pieces.append(f"{coeff_str}*{e}")

        return " + ".join(pieces).replace("+ -", "- ")

class LinearMap:
    def __init__(self, source, target, data=None):
        if data is None:
            data = dict()
        if not isinstance(source, FreeModule) or not isinstance(target, FreeModule):
            raise TypeError("source and target must be FreeModule objects")
        self.source = source
        self.target = target

        if self.source.ring is not self.target.ring:
            raise ValueError(f'source and target over incompatible rings {self.source.ring}!={self.target.ring}')
        self.ring = self.source.ring
        self.data = dict()

        for key, val in data.items():
            '''
            scaffolding: check that key of form
            (targetBasisElem, sourceBasisElem)
            check that value is in the base ring
            then assign
            '''
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError(f"Key {key} must be a tuple of (sourceBasisElem, targetBasisElem)")
            if not isinstance(key[0], str) or key[0] not in self.target.basis:
                raise ValueError(f"First element of key {key} must be a basis element of the target module")
            if not isinstance(key[1], str) or key[1] not in self.source.basis:
                raise ValueError(f"Second element of key {key} must be a basis element of the source module")

            c = self.ring.coerce(val)
            if not self.ring.is_zero(c):
                self.data[key] = self.data.get(key, self.ring.zero()) + c

        self._prune()
        '''
        implement composition, addition, scalar
        multiplication, and application to elements
        of the source module
        '''

    def _prune(self):
        self.data = {key: val for key, val in self.data.items() if not self.ring.is_zero(val)}


    def __call__(self, elem):
        if not isinstance(elem, FreeModuleElem):
            raise TypeError("LinearMap can only be applied to FreeModuleElems")
        if elem.parent is not self.source:
            raise ValueError("Element does not belong to the source module")

        image = {}

        for s, coeff in elem.data.items():
            for t in self.target.basis:
                a = self.data.get((t,s), self.ring.zero())
                if not self.ring.is_zero(a):
                    image[t] = image.get(t, self.ring.zero())+a*coeff
        return FreeModuleElem(self.target, image)


    def __repr__(self):
        return f"LinearMap({self.source} -> {self.target}, num_nonzero={len(self.data)})"

