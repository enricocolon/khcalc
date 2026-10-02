#!/usr/bin/env python3

from .uq import ShiftedUWord

class U2Expression:
    pass

class IdentityU2(U2Expression):
    def __repr__(self):
        return "IdentityU2()"

    def __eq__(self, other):
        return isinstance(other, IdentityU2)

    def __hash__(self):
        return hash(IdentityU2)

class ZeroU2(U2Expression):
    def __repr__(self):
        return "ZeroU2()"

    def __eq__(self, other):
        return isinstance(other, ZeroU2)

    def __hash__(self):
        return hash(ZeroU2)

class RickardU2(U2Expression):
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
    return isinstance(expression, U2Expression)


def _require_u2_expression(expression):
    if not _is_u2_expression(expression):
        raise TypeError(
            "expected a U2 expression"
        )


class CompositeU2(U2Expression):
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


class HorizontalU2(U2Expression):
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


class SumU2(U2Expression):
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

class NegU2(U2Expression):
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


#####
# 2-morphism generators (thin), c.f. Queffelec-Rose Def 2.1 2-morphisms
#####

class DotU2(U2Expression):
    def __init__(self, strand_index, morphism_type):
        #strands are indexed from zero
        if type(strand_index) is not int:
            raise TypeError("strand_index must be an integer")

        if strand_index < 0:
            raise ValueError("strand_index must be a nonnegative integer")

        if morphism_type not in {"E", "F"}:
            raise ValueError(
                "morphism_type must be 'E' or 'F'"
            )

        self.strand_index = strand_index
        self.morphism_type = morphism_type #E==up, F==down., cf Queffelec-Rose Def 2.1

    def __repr__(self):
        return f"DotU2(strand_index={self.strand_index!r}, morphism_type={self.morphism_type!r})"

    def __eq__(self, other):
        if not isinstance(other, DotU2):
            return NotImplemented

        return (self.strand_index == other.strand_index) and (self.morphism_type == other.morphism_type)

    def __hash__(self):
        return hash((DotU2, self.strand_index, self.morphism_type))


class CrossingU2(U2Expression):
    def __init__(self, left_strand_index, morphism_type):
        if type(left_strand_index) is not int:
            raise TypeError("left_strand_index must be integer")

        if left_strand_index < 0:
            raise ValueError("left_strand_index must be a nonnegative integer")

        if morphism_type not in {"E", "F"}:
            raise ValueError(
                "morphism_type must be 'E' or 'F'"
            )

        self.left_strand_index = left_strand_index
        self.right_strand_index = left_strand_index + 1
        self.morphism_type = morphism_type #E==up, F==down., cf Queffelec-Rose Def 2.1

    def __repr__(self):
        return f"CrossingU2(left_strand_index={self.left_strand_index}, morphism_type={self.morphism_type!r})"

    def __eq__(self, other):
        if not isinstance(other, CrossingU2):
            return NotImplemented

        return (self.left_strand_index == other.left_strand_index) and (self.morphism_type == other.morphism_type)

    def __hash__(self):
        return hash((CrossingU2, self.left_strand_index, self.morphism_type))

class CupU2(U2Expression):
    def __init__(self, gap_index, morphism_type):
        #gap_index = i inserts a cup between strands i and i+1. indexed from 0.
        if type(gap_index) is not int:
            raise TypeError("gap_index must be an integer")

        if gap_index < 0:
            raise ValueError("gap_index must be nonnegative")

        if morphism_type not in {"EF", "FE"}:
            raise ValueError(
                "morphism_type must be 'EF' or 'FE'"
            )

        self.gap_index = gap_index
        self.morphism_type = morphism_type #EF==leftward, FE==rightward, cf Queffelec-Rose Def 2.1

    def __repr__(self):
        return f"CupU2(gap_index={self.gap_index!r}, morphism_type={self.morphism_type!r})"

    def __eq__(self, other):
        if not isinstance(other, CupU2):
            return NotImplemented

        return (self.gap_index == other.gap_index) and (self.morphism_type == other.morphism_type)

    def __hash__(self):
        return hash((CupU2, self.gap_index, self.morphism_type))


class CapU2(U2Expression):
    def __init__(self, gap_index, morphism_type):
        if type(gap_index) is not int:
            raise TypeError("gap_index must be an integer")

        if gap_index < 0:
            raise ValueError("gap_index must be nonnegative")

        if morphism_type not in {"EF", "FE"}:
            raise ValueError(
                "morphism_type must be 'EF' or 'FE'"
            )

        self.gap_index = gap_index
        self.morphism_type = morphism_type #EF==rightward, FE==leftward, cf Queffelec-Rose Def 2.1

    def __repr__(self):
        return f"CapU2(gap_index={self.gap_index!r}, morphism_type={self.morphism_type!r})"

    def __eq__(self, other):
        if not isinstance(other, CapU2):
            return NotImplemented

        return (self.gap_index == other.gap_index) and (self.morphism_type == other.morphism_type)

    def __hash__(self):
        return hash((CapU2, self.gap_index, self.morphism_type))


def _validate_dot_u2(source, target, expression, q_degree):
    # source: ShiftedUWord
    # target: ShiftedUWord
    # expression: U2Expression
    # q_degree: integer
    if not isinstance(expression, DotU2):
        raise TypeError("expression must be a DotU2")

    if expression.strand_index < 0:
        raise ValueError("strand_index must be nonnegative")

    if expression.morphism_type not in {"E", "F"}:
        raise ValueError(
            "morphism_type must be 'E' or 'F'"
        )


    if expression.strand_index >= len(source.factors):
        raise ValueError(
            "strand_index out of range for source ShiftedUWord"
        )

    if expression.morphism_type == "E":
        if source.factors[expression.strand_index].direction != "E": #.direction? or bare factor. direction for now.
            raise ValueError(
                "strand_index does not correspond to an E strand in source ShiftedUWord"
            )
        if target.factors[expression.strand_index].direction != "E":
            raise ValueError(
                "strand_index does not correspond to an E strand in target ShiftedUWord"
            )


    if expression.morphism_type == "F":
        if source.factors[expression.strand_index].direction != "F":
            raise ValueError(
                "strand_index does not correspond to an F strand in source ShiftedUWord"
            )
        if target.factors[expression.strand_index].direction != "F":
            raise ValueError(
                "strand_index does not correspond to an F strand in target ShiftedUWord"
            )


    if target.q_shift - source.q_shift != 2:
        raise ValueError(
            "q-degree shift for DotU2 on E strand must be +2"
        )


def _validate_crossing_u2(source, target, expression, q_degree):
    if not isinstance(expression, CrossingU2):
        raise TypeError("expression must be a CrossingU2")

    if expression.left_strand_index < 0:
        raise ValueError("left_strand_index must be nonnegative")

    if expression.morphism_type not in {"E", "F"}:
        raise ValueError(
            "morphism_type must be 'E' or 'F'"
        )

    if expression.left_strand_index >= len(source.factors) - 1:
        raise ValueError(
            "left_strand_index out of range for source ShiftedUWord"
        )

    if expression.morphism_type == "E":
        pass

    if expression.morphism_type == "F":
        pass

    #implement a degree checker in terms of color
    degree = target.q_shift - source.q_shift

    if degree == -2:
        pass
    elif degree == 1:
        pass
    elif degree == 0:
        pass
    else:
        raise ValueError("q-degree shift for CrossingU2 must be -2, 0, or +1")

def _validate_cup_u2(source, target, expression, q_degree):
    if not isinstance(expression, CupU2):
        raise TypeError("expression must be a CupU2")

    if expression.gap_index < 0:
        raise ValueError("gap_index must be nonnegative")

    if expression.morphism_type not in {"EF", "FE"}:
        raise ValueError(
            "morphism_type must be 'EF' or 'FE'"
        )

    #MAKE SURE TO DISAMBIGUATE COLOR, THICKNESS, STRAND INDEX.
    lam_i = source.source.lam(expression.gap_index) #CHECK THAT THE INDICES MATCH

    if expression.morphism_type == "EF":
        #check source/target compatibility before q shift

        if target.q_shift - source.q_shift != 1 - lam_i:
            raise ValueError(
                "q-degree shift for CupU2 of type EF must be 1 - lam_i"
            )
        pass

    if expression.morphism_type == "FE":
        #check source/target compatibility before q shift

        if target.q_shift - source.q_shift != 1 + lam_i:
            raise ValueError(
                "q-degree shift for CupU2 of type FE must be 1 + lam_i"
            )
        pass


def _validate_cap_u2(source, target, expression, q_degree):
    if not isinstance(expression, CapU2):
        raise TypeError("expression must be a CapU2")

    if expression.gap_index < 0:
        raise ValueError("gap_index must be nonnegative")

    if expression.morphism_type not in {"EF", "FE"}:
        raise ValueError(
            "morphism_type must be 'EF' or 'FE'"
        )

    lam_i = source.source.lam(expression.gap_index) #CHECK THAT THE INDICES MATCH

    if expression.morphism_type == "EF":
        #check source/target compatibility before q shift

        if target.q_shift - source.q_shift != 1 - lam_i:
            raise ValueError(
                "q-degree shift for CapU2 of type EF must be 1 - lam_i"
            )
        pass

    if expression.morphism_type == "FE":
        #check source/target compatibility before q shift

        if target.q_shift - source.q_shift != 1 + lam_i:
            raise ValueError(
                "q-degree shift for CapU2 of type FE must be 1 + lam_i"
            )
        pass




def _validate_u2_morphism(source, target, expression, q_degree):
        if not isinstance(source, ShiftedUWord):
            raise TypeError("source must be a ShiftedUWord")

        if not isinstance(target, ShiftedUWord):
            raise TypeError("target must be a ShiftedUWord")

        if source.source != target.source:
            raise ValueError(
                "source and target words must have the same source boundary"
            )

        if source.target != target.target:
            raise ValueError(
                "source and target words must have the same target boundary"
            )

        if not isinstance(expression, U2Expression):
            raise TypeError(
                "expression must be a U2Expression"
            )

        if isinstance(expression, IdentityU2):
            if source != target:
                raise ValueError(
                    "an identity 2-morphism must have equal source and target"
                )

        #source/target checking for elementary 2-morphisms
        if isinstance(expression, DotU2):
            _validate_dot_u2(source, target, expression, q_degree)

        if isinstance(expression, CrossingU2):
            _validate_crossing_u2(source, target, expression, q_degree)

        if isinstance(expression, CupU2):
            _validate_cup_u2(source, target, expression, q_degree)

        if isinstance(expression, CapU2):
            _validate_cap_u2(source, target, expression, q_degree)

        pass


class UTwoMorphism:
    def __init__(
        self,
        source,
        target,
        expression,
        q_degree=0,
    ):
        if not isinstance(source, ShiftedUWord):
            raise TypeError("source must be a ShiftedUWord")

        if not isinstance(target, ShiftedUWord):
            raise TypeError("target must be a ShiftedUWord")

        if source.source != target.source:
            raise ValueError(
                "source and target words must have the same source boundary"
            )

        if source.target != target.target:
            raise ValueError(
                "source and target words must have the same target boundary"
            )

        if not isinstance(expression, U2Expression):
            raise TypeError(
                "expression must be a U2Expression"
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
        _validate_u2_morphism(source, target, expression, q_degree)

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
            expression=SumU2(tuple(terms)),
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

        source = self.source.then(
            other.source
        )

        target = self.target.then(
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
