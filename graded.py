#!/usr/bin/env python3


from linalg import FreeModule, FreeModuleElem, LinearMap

class GradedFreeModule():
    def __init__(self, ring, basis, degrees):
        self.module = FreeModule(ring, basis=basis)
        self.ring = self.module.ring
        self.basis = self.module.basis
        self.index = self.module.index
        self.dim = self.module.dim

        if isinstance(degrees, dict):
            if set(degrees.keys()) != set(self.basis):
                raise ValueError("Degree dictionary keys must match basis")
            self.degrees = {e: int(degrees[e]) for e in self.basis}
        else:
            degrees = tuple(degrees)
            if len(self.basis) != len(degrees):
                raise ValueError("Length of degrees must match length of basis")
            self.degrees = {e: int(d) for e, d in zip(self.basis, degrees)}

    def zero(self):
        return self.module.zero()

    def elem(self, data):
        return self.module.elem(data)

    def basis_vector(self, e):
        return self.module.basis_vector(e)

    def degree_of_basis(self, e):
        if e not in self.index:
            raise ValueError(f"{e} is not a basis element")
        return self.degrees[e]

    def homogeneous_degrees(self, elem):
        if not isinstance(elem, FreeModuleElem):
            raise TypeError("Expected a FreeModuleElem")
        if elem.parent is not self.module:
            raise ValueError("Element does not belong to this graded module.")
        return {self.degrees[e] for e in elem.data.keys()}

    def degree(self, elem):
        degs = self.homogeneous_degrees(elem)
        if len(degs) == 0:
            return None
        elif len(degs) == 1:
            return degs.pop()
        else:
            raise ValueError("Element is not homogeneous")

    def is_homogeneous(self, elem):
        try:
            self.degree(elem)
            return True
        except ValueError:
            return False

    def shift(self, k):
        return GradedFreeModule(self.ring, self.basis, {e: d+k for e, d in self.degrees.items()})

    def __repr__(self):
        return f"GradedFreeModule({self.ring}, basis={self.basis}, degrees={self.degrees})"
