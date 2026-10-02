#!/usr/bin/env python3

from .boundary import WebBoundary

class DividedPower:
    def __init__(self, direction, i, r, source):
        '''
        Input: a direction 'E' or 'F', an index i,
        a nonnegative integer r, and a source WebBoundary.
        Output: Object representing boundary-labelled divided power E/F^(r)_i 1_source.
        '''
        if direction not in {"E","F"}:
            raise ValueError("direction must be 'E' or 'F'")

        if not isinstance(source, WebBoundary):
            raise TypeError("source must be a WebBoundary")

        if type(i) is not int:
            raise TypeError("i must be an integer")

        if not 0 <= i < source.m - 1:
            raise IndexError(f"{i} is not a valid index for {source}")

        if type(r) is not int:
            raise TypeError("r must be an integer")

        if r < 0:
            raise ValueError("r must be nonnegative")

        self.direction = direction
        self.i = i
        self.r = r
        self.source = source
        self.target = self._compute_target()

    def _compute_target(self):
        colors = list(self.source.colors)

        if self.direction == "E":
            colors[self.i] += self.r
            colors[self.i+1] -= self.r
        else:
            colors[self.i] -= self.r
            colors[self.i+1] += self.r

        if any(color < 0 for color in colors):
            return None #c.f. is_zero property

        return WebBoundary(colors)

    @property
    def is_zero(self):
        return self.target is None

    def __str__(self):
        return (
            f"{self.direction}_{self.i}^({self.r}) "
            f"1_{self.source}"
        )

    def __repr__(self):
        return (
            f"DividedPower({self.direction!r}, {self.i!r}, "
            f"{self.r!r}, {self.source!r})"
        )

    def __eq__(self, other):
        if not isinstance(other, DividedPower):
            return NotImplemented
        return (
            self.direction,
            self.i,
            self.r,
            self.source,
        ) == (
            other.direction,
            other.i,
            other.r,
            other.source,
        )

    def __hash__(self):
        return hash((
            self.direction,
            self.i,
            self.r,
            self.source,
        ))

class UWord:
    def __init__(self, source, factors=()):
        if not isinstance(source, WebBoundary):
            raise TypeError("source must be a WebBoundary")

        factors = tuple(factors)

        if any(not isinstance(factor, DividedPower) for factor in factors):
            raise TypeError("all factors must be DividedPower objects")

        curr = source

        for factor in factors:
            if factor.source != curr:
                raise ValueError(f"Composition error, expected source {curr}, got {factor.source}")

            if factor.is_zero:
                curr = None
                break

            curr = factor.target

        self.source = source
        self.factors = tuple(factor for factor in factors if factor.r != 0)
        self.target = curr


    @property
    def is_zero(self):
        return self.target is None

    @property
    def is_identity(self):
        return not self.is_zero and len(self.factors) == 0

    def then_factor(self, factor):
        '''
        Append a factor to the end of the word.
        '''
        if not isinstance(factor, DividedPower):
            raise TypeError("factor must be a DividedPower")

        if self.is_zero:
            raise ValueError("cannot append factor to a zero word")

        if self.target != factor.source:
            raise ValueError("Self target and factor source do not match")

        return UWord(source = self.source,
                     factors = self.factors + (factor,),)

    def then(self, other):
        if not isinstance(other, UWord):
            raise TypeError("other must be a UWord")

        if self.is_zero or other.is_zero:
            raise NotImplementedError

        if self.target != other.source:
            raise ValueError("UWords not compatible")

        return UWord(self.source, (self.factors+other.factors),)

    def __len__(self):
        return len(self.factors)

    def __str__(self):
        if self.is_identity:
            return f"1_{self.source}"

        #Reverse to reflect usual composition order
        return " ".join(
            str(factor).split(" 1_")[0]
            for factor in reversed(self.factors)
        ) + f" 1_{self.source}"

    def __repr__(self):
        return (
            f"UWord(source={self.source!r}, "
            f"factors={self.factors!r})"
        )

    def __eq__(self, other):
        if not isinstance(other, UWord):
            return NotImplemented

        return (
            self.source,
            self.factors,
        ) == (
            other.source,
            other.factors,
        )

    def __hash__(self):
        return hash((self.source, self.factors))

    def is_zero_at_rank(self, n):
        if self.is_zero:
            return True

        try:
            self.source.require_admissible(n)
        except ValueError:
            return True

        for factor in self.factors:
            if factor.is_zero:
                return True

            try:
                factor.target.require_admissible(n)
            except ValueError:
                return True

        return False
