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
            self.basis = {f"e_{i}" for i in range(dim)}
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
    NotImplemented

class LinearMap:
    NotImplemented
