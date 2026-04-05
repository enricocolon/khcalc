#!/usr/bin/env python3

class ChainComplex:
    def __init__(self, objects=None, differentials=None):
        if objects is None:
            objects = dict()
        if differentials is None:
            differentials = dict()

        self.objects = dict(objects)
        self.differentials = dict(differentials)

        self._validate_objects()
        self._validate_differentials()
        self._validate_complex()

    def _validate_objects(self):
        for key, val in self.objects.items():
            if not isinstance(key, int):
                raise TypeError(f"Object keys must be integers, got {key}")

    def _validate_differentials(self):
        for key, val in self.differentials.items():
            if not isinstance(key, int):
                raise TypeError(f"Differential keys must be integers, got {key}")

            if not hasattr(val, "source") or not hasattr(val, "target"):
                raise TypeError(f"Differential {key}: {val} must have a source and target")

            if key not in self.objects:
                raise ValueError(f"Differential {key} has no source object")

            if key + 1 not in self.objects:
                raise ValueError(f"Differential {key} has no target object")


            if val.source is not self.objects[key]:
                raise ValueError(f"Differential {key} has source {val.source}, expected {self.objects[key]}")

            if val.target is not self.objects[key+1]:
                raise ValueError(f"Differential {key} has target {val.target}, expected {self.objects[key+1]}")

    def _validate_complex(self):
        for i in self.differentials:
            if i + 1 in self.differentials:
                comp = self.differentials[i+1]@self.differentials[i]
                if not comp.is_zero():
                    raise ValueError(f"d^{i+1}∘d^{i}≠0")

    def degrees(self):
        return sorted(self.objects.keys())

    def object(self, i):
        return self.objects[i]

    def differential(self, i):
        return self.differentials[i]

    def __repr__(self):
        return f"ChainComplex(degrees={self.degrees()})"
