#!/usr/bin/env python3


class WebBoundary:
    def __init__(self, array):
        colors = tuple(array) #(a_1,...,a_m)

        if any(type(color) is not int for color in colors):
            raise TypeError("All colors must be integers.")

        if any(color < 0 for color in colors):
            raise ValueError("All colors must be nonnegative.")

        self.colors = colors


    @property
    def m(self):
        return len(self.colors)

    @property
    def N(self):
        return sum(self.colors)

    def lam(self, i):
        return self.colors[i] - self.colors[i+1]

    def s(self, i):
        arr = list(self.colors)
        tmp = arr[i]
        arr[i] = arr[i+1]
        arr[i+1] = tmp
        return WebBoundary(arr)

    def __str__(self):
        return f"WebBoundary{self.colors}"

    def __repr__(self):
        return f"WebBoundary({self.colors!r})"

    def __eq__(self, other):
        if not isinstance(other, WebBoundary):
            return NotImplemented
        return self.colors == other.colors

    def __hash__(self):
        return hash(self.colors)

    def require_admissible(self, n: int):
        if type(n) is not int or n < 1:
            raise ValueError(f"{n!r} must be a positive integer.")
        if any(color > n for color in self.colors):
            raise ValueError(f"All colors must be less than or equal to {n}.")
