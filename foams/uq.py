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

        self.source = source
        self.factors = factors
        self.target = self._compute_target()

    def _compute_target(self):
        curr = self.source

        for factor in self.factors:
            if factor.source != curr:
                raise ValueError(
                        f"Composition error: expected source {curr}, "
                        f"got {factor.source}"
                    )

            if factor.is_zero:
                return None

            curr = factor.target

        return curr

    @property
    def is_zero(self):
        return self.target is None

    @property
    def is_identity(self):
        return not self.is_zero and len(self.factors) == 0

    def then(self, factor):
        '''
        Append a factor to the end of the word.
        '''
        if not isinstance(factor, DividedPower):
            raise TypeError("factor must be a DividedPower")

        return UWord(source = self.source,
                     factors = self.factors + (factor,),)

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


class IdentityU2:
    def __repr__(self):
        return "IdentityU2()"

    def __eq__(self, other):
        return isinstance(other, IdentityU2)

    def __hash__(self):
        return hash(IdentityU2)


class ZeroU2:
    def __repr__(self):
        return "ZeroU2()"

    def __eq__(self, other):
        return isinstance(other, ZeroU2)

    def __hash__(self):
        return hash(ZeroU2)


class RickardU2:
    def __init__(self, i, step):
        if type(i) is not int:
            raise TypeError("i must be an integer")

        if i < 0:
            raise ValueError("i must be nonnegative")

        if type(step) is not int:
            raise TypeError("step must be an integer")

        if step < 0:
            raise ValueError("step must be nonnegative")

        self.i = i
        self.step = step

    def __repr__(self):
        return (
            f"RickardU2(i={self.i!r}, "
            f"step={self.step!r})"
        )

    def __eq__(self, other):
        if not isinstance(other, RickardU2):
            return NotImplemented

        return (
            self.i,
            self.step,
        ) == (
            other.i,
            other.step,
        )

    def __hash__(self):
        return hash((self.i, self.step))


class UTwoMorphism:
    def __init__(
        self,
        source,
        target,
        expression,
        q_degree=0,
    ):
        if not isinstance(source, UWord):
            raise TypeError("source must be a UWord")

        if not isinstance(target, UWord):
            raise TypeError("target must be a UWord")

        if source.source != target.source:
            raise ValueError(
                "source and target words must have "
                "the same source boundary"
            )

        if source.target != target.target:
            raise ValueError(
                "source and target words must have "
                "the same target boundary"
            )

        if not isinstance(
            expression,
            (IdentityU2, ZeroU2, RickardU2),
        ):
            raise TypeError(
                "unsupported UTwoMorphism expression"
            )

        if type(q_degree) is not int:
            raise TypeError("q_degree must be an integer")

        if isinstance(expression, IdentityU2):
            if source != target:
                raise ValueError(
                    "an identity 2-morphism must have "
                    "equal source and target"
                )

            if q_degree != 0:
                raise ValueError(
                    "an identity 2-morphism must have "
                    "quantum degree zero"
                )

        self.source = source
        self.target = target
        self.expression = expression
        self.q_degree = q_degree

    @property
    def is_zero(self):
        return isinstance(self.expression, ZeroU2)

    @property
    def is_identity(self):
        return isinstance(self.expression, IdentityU2)

    def __repr__(self):
        return (
            f"UTwoMorphism("
            f"source={self.source!r}, "
            f"target={self.target!r}, "
            f"expression={self.expression!r}, "
            f"q_degree={self.q_degree!r})"
        )

    def __eq__(self, other):
        if not isinstance(other, UTwoMorphism):
            return NotImplemented

        return (
            self.source,
            self.target,
            self.expression,
            self.q_degree,
        ) == (
            other.source,
            other.target,
            other.expression,
            other.q_degree,
        )

    def __hash__(self):
        return hash((
            self.source,
            self.target,
            self.expression,
            self.q_degree,
        ))
