#!/usr/bin/env python3

from .boundary import WebBoundary

class LadderRung:
    def __init__(self, direction, i, r, source):
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
            f"LadderRung({self.direction!r}, {self.i!r}, "
            f"{self.r!r}, {self.source!r})"
        )

    def __eq__(self, other):
        if not isinstance(other, LadderRung):
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




class Web:
    def __init__(self, source, rungs=()):
        if not isinstance(source, WebBoundary):
            raise TypeError("source must be a WebBoundary")

        rungs = tuple(rungs)

        if any(not isinstance(rung, LadderRung) for rung in rungs):
            raise TypeError("all rungs must be LadderRung objects")

        curr = source

        for rung in rungs:
            if rung.source != curr:
                raise ValueError(f"Composition error, expected source {curr}, got {rung.source}")

            if rung.is_zero:
                curr = None
                break

            curr = rung.target

        self.source = source
        self.rungs = tuple(rung for rung in rungs if rung.r != 0)
        self.target = curr


    @property
    def is_zero(self):
        return self.target is None

    @property
    def is_identity(self):
        return not self.is_zero and len(self.rungs) == 0

    def then_rung(self, rung):
        '''
        Append a rung to the top of the web.
        '''
        if not isinstance(rung, LadderRung):
            raise TypeError("rung must be a LadderRung")

        if self.is_zero:
            raise ValueError("cannot append rung to a zero word")

        if self.target != rung.source:
            raise ValueError("Self target and rung source do not match")

        return Web(self.source, self.rungs + (rung,),)

    def then(self, other):
        if not isinstance(other, Web):
            raise TypeError("other must be a Web")

        if self.is_zero or other.is_zero:
            raise NotImplementedError

        if self.target != other.source:
            raise ValueError("Webs not compatible")

        return Web(self.source, (self.rungs+other.rungs),)

    def __len__(self):
        return len(self.rungs)

    def __str__(self):
        if self.is_identity:
            return f"1_{self.source}"

        #Reverse to reflect usual composition order
        return " ".join(
            str(rung).split(" 1_")[0]
            for rung in reversed(self.rungs)
        ) + f" 1_{self.source}"

    def __repr__(self):
        return (
            f"Web(source={self.source!r}, "
            f"rungs={self.rungs!r})"
        )

    def __eq__(self, other):
        if not isinstance(other, Web):
            return NotImplemented

        return (
            self.source,
            self.rungs,
        ) == (
            other.source,
            other.rungs,
        )

    def __hash__(self):
        return hash((self.source, self.rungs))

    def is_zero_at_rank(self, n):
        if self.is_zero:
            return True

        try:
            self.source.require_admissible(n)
        except ValueError:
            return True

        for rung in self.rungs:
            if rung.is_zero:
                return True

            try:
                rung.target.require_admissible(n)
            except ValueError:
                return True

        return False

class IdentityFoam:
    def __repr__(self):
        return "IdentityFoam()"

    def __eq__(self, other):
        return isinstance(other, IdentityFoam)

    def __hash__(self):
        return hash(IdentityFoam)

class IdentityFoam:
    def __repr__(self):
        return "IdentityFoam()"

    def __eq__(self, other):
        return isinstance(other, IdentityFoam)

    def __hash__(self):
        return hash(IdentityFoam)


class ZeroFoam:
    def __repr__(self):
        return "ZeroFoam()"

    def __eq__(self, other):
        return isinstance(other, ZeroFoam)

    def __hash__(self):
        return hash(ZeroFoam)

class RickardFoam:
    pass
