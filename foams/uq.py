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

    def then_word(self, other):
        if not isinstance(other, UWord):
            raise TypeError("other must be a UWord")

        if self.target != other.source:
            raise ValueError("UWords not compatible")

        return UWord(self.source, self.factors+other.factors,)

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
    """
    Symbolic Rickard differential d_step.

    family="EF":
        E^(-lambda+s) F^s -> E^(-lambda+s+1) F^(s+1) (c.f., arXiv:1405.5920v1, (2.42))

    family="FE":
        F^(lambda+s) E^s -> F^(lambda+s+1) E^(s+1) (c.f., arXiv:1405.5920v1, (2.43))
    """

    def __init__(self, family, i, step):
        if family not in {"EF", "FE"}:
            raise ValueError(
                "family must be 'EF' or 'FE'"
            )

        if type(i) is not int:
            raise TypeError("i must be an integer")

        if i < 0:
            raise ValueError("i must be nonnegative")

        if type(step) is not int:
            raise TypeError("step must be an integer")

        if step < 1:
            raise ValueError(
                "a Rickard differential step must be positive"
            )

        self.family = family
        self.i = i
        self.step = step

    def __repr__(self):
        return (
            f"RickardU2("
            f"family={self.family!r}, "
            f"i={self.i!r}, "
            f"step={self.step!r})"
        )

    def __eq__(self, other):
        if not isinstance(other, RickardU2):
            return NotImplemented

        return (
            self.family,
            self.i,
            self.step,
        ) == (
            other.family,
            other.i,
            other.step,
        )

    def __hash__(self):
        return hash((
            self.family,
            self.i,
            self.step,
        ))


def _is_u2_expression(expression):
    return isinstance(
        expression,
        (
            IdentityU2,
            ZeroU2,
            RickardU2,
            CompositeU2,
            HorizontalU2,
            SumU2,
            NegU2,
        ),
    )


def _require_u2_expression(expression):
    if not _is_u2_expression(expression):
        raise TypeError(
            "expected a U2 expression"
        )


class CompositeU2:
    """
    Formal vertical composition.

    CompositeU2(first, second) represents

        second ∘ first.
    """

    def __init__(self, first, second):
        _require_u2_expression(first)
        _require_u2_expression(second)

        self.first = first
        self.second = second

    def __repr__(self):
        return (
            f"CompositeU2("
            f"first={self.first!r}, "
            f"second={self.second!r})"
        )

    def __eq__(self, other):
        if not isinstance(other, CompositeU2):
            return NotImplemented

        return (
            self.first,
            self.second,
        ) == (
            other.first,
            other.second,
        )

    def __hash__(self):
        return hash((
            self.first,
            self.second,
        ))


class HorizontalU2:
    """
    Formal horizontal composition of 2-morphisms.
    """

    def __init__(self, first, second):
        _require_u2_expression(first)
        _require_u2_expression(second)

        self.first = first
        self.second = second

    def __repr__(self):
        return (
            f"HorizontalU2("
            f"first={self.first!r}, "
            f"second={self.second!r})"
        )

    def __eq__(self, other):
        if not isinstance(other, HorizontalU2):
            return NotImplemented

        return (
            self.first,
            self.second,
        ) == (
            other.first,
            other.second,
        )

    def __hash__(self):
        return hash((
            self.first,
            self.second,
        ))


class SumU2:
    """
    Formal sum of parallel 2-morphism expressions.
    """

    def __init__(self, terms):
        terms = tuple(terms)

        if len(terms) < 2:
            raise ValueError(
                "SumU2 must contain at least two terms"
            )

        for term in terms:
            _require_u2_expression(term)

        self.terms = terms

    def __repr__(self):
        return f"SumU2(terms={self.terms!r})"

    def __eq__(self, other):
        if not isinstance(other, SumU2):
            return NotImplemented

        return self.terms == other.terms

    def __hash__(self):
        return hash(self.terms)


class NegU2:
    """
    Formal additive inverse of a 2-morphism expression.
    """

    def __init__(self, expression):
        _require_u2_expression(expression)

        self.expression = expression

    def __repr__(self):
        return (
            f"NegU2("
            f"expression={self.expression!r})"
        )

    def __eq__(self, other):
        if not isinstance(other, NegU2):
            return NotImplemented

        return self.expression == other.expression

    def __hash__(self):
        return hash(self.expression)


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
                "source and target words must have the same source boundary"
            )

        if source.target != target.target:
            raise ValueError(
                "source and target words must have the same target boundary"
            )

        if not isinstance(
            expression,
            (IdentityU2,
             ZeroU2,
             RickardU2,
             CompositeU2,
             HorizontalU2,
             SumU2,
             NegU2),
        ):
            raise TypeError(
                "unsupported UTwoMorphism expression"
            )

        if type(q_degree) is not int:
            raise TypeError("q_degree must be an integer")

        if isinstance(expression, IdentityU2):
            if source != target:
                raise ValueError(
                    "an identity 2-morphism must have equal source and target"
                )

            if q_degree != 0:
                raise ValueError(
                    "an identity 2-morphism must have q-degree zero"
                )

        #TK:validate source/target for Rickard morphisms

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

    def __neg__(self):
        if self.is_zero:
            return self

        if isinstance(self.expression, NegU2):
            return UTwoMorphism(
                source=self.source,
                target=self.target,
                expression=self.expression.expression,
                q_degree=self.q_degree,
            )

        return UTwoMorphism(
            source=self.source,
            target=self.target,
            expression=NegU2(self.expression),
            q_degree=self.q_degree,
        )

    def __add__(self, other):
        if not isinstance(other, UTwoMorphism):
            return NotImplemented

        if self.source != other.source:
            raise ValueError(
                "2-morphisms must have the same source"
            )

        if self.target != other.target:
            raise ValueError(
                "2-morphisms must have the same target"
            )

        if self.q_degree != other.q_degree:
            raise ValueError(
                "2-morphisms must have the same quantum degree"
            )

        if self.is_zero:
            return other

        if other.is_zero:
            return self

        terms = []

        if isinstance(self.expression, SumU2):
            terms.extend(self.expression.terms)
        else:
            terms.append(self.expression)

        if isinstance(other.expression, SumU2):
            terms.extend(other.expression.terms)
        else:
            terms.append(other.expression)

        return UTwoMorphism(
            source=self.source,
            target=self.target,
            expression=SumU2(terms),
            q_degree=self.q_degree,
        )

    def then(self, other):
        """
        Vertical composition.

        self : F => G
        other : G => H

        returns other o self : F => H.
        """
        if not isinstance(other, UTwoMorphism):
            raise TypeError(
                "other must be a UTwoMorphism"
            )

        if self.target != other.source:
            raise ValueError(
                "2-morphisms are not vertically composable"
            )

        q_degree = (
            self.q_degree
            + other.q_degree
        )

        if self.is_zero or other.is_zero:
            return UTwoMorphism(
                source=self.source,
                target=other.target,
                expression=ZeroU2(),
                q_degree=q_degree,
            )

        if self.is_identity:
            return UTwoMorphism(
                source=self.source,
                target=other.target,
                expression=other.expression,
                q_degree=q_degree,
            )

        if other.is_identity:
            return UTwoMorphism(
                source=self.source,
                target=other.target,
                expression=self.expression,
                q_degree=q_degree,
            )

        return UTwoMorphism(
            source=self.source,
            target=other.target,
            expression=CompositeU2(
                self.expression,
                other.expression,
            ),
            q_degree=q_degree,
        )

    def horizontal(self, other):
        """
        Horizontal composition.

        If
            self  : F => F'
            other : G => G'

        with F,F' : a -> b and G,G' : b -> c,
        returns

            G o F => G' o F'.
        """
        if not isinstance(other, UTwoMorphism):
            raise TypeError(
                "other must be a UTwoMorphism"
            )

        source = self.source.then_word(
            other.source
        )

        target = self.target.then_word(
            other.target
        )

        q_degree = (
            self.q_degree
            + other.q_degree
        )

        if self.is_zero or other.is_zero:
            expression = ZeroU2()

        elif self.is_identity and other.is_identity:
            expression = IdentityU2()

        else:
            expression = HorizontalU2(
                self.expression,
                other.expression,
            )

        return UTwoMorphism(
            source=source,
            target=target,
            expression=expression,
            q_degree=q_degree,
        )

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
