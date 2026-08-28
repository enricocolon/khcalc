#!/usr/bin/env python3

from .uq import UWord

class Shifted:
    """
    A formal quantum grading shift q^k(value).

    The value must provide:
        source
        target
        is_zero

    and may optionally provide:
        is_identity
    """

    def __init__(self, value, q_shift=0):
        if type(q_shift) is not int:
            raise TypeError("q_shift must be an integer")

        for attribute in (
            "source",
            "target",
            "is_zero",
        ):
            if not hasattr(value, attribute):
                raise TypeError(
                    f"value must provide {attribute!r}"
                )

        self.value = value
        self.q_shift = q_shift

    @property
    def source(self):
        return self.value.source

    @property
    def target(self):
        return self.value.target

    @property
    def is_zero(self):
        return self.value.is_zero

    @property
    def is_identity(self):
        return (
            self.q_shift == 0
            and getattr(self.value, "is_identity", False)
        )

    def __str__(self):
        if self.q_shift == 0:
            return str(self.value)

        return f"q^{self.q_shift}({self.value})"

    def __repr__(self):
        return (
            f"Shifted({self.value!r}, "
            f"q_shift={self.q_shift!r})"
        )

    def __eq__(self, other):
        if not isinstance(other, Shifted):
            return NotImplemented

        return (
            self.value,
            self.q_shift,
        ) == (
            other.value,
            other.q_shift,
        )

    def __hash__(self):
        return hash((self.value, self.q_shift))


class DirectSum:
    """
    An ordered formal direct sum of shifted 1-morphisms.

    Repeated and zero summands are preserved so that future
    differential-matrix indices remain stable.
    """

    def __init__(self, summands=()):
        summands = tuple(summands)

        if any(
            not isinstance(summand, Shifted)
            for summand in summands
        ):
            raise TypeError(
                "all summands must be Shifted objects"
            )

        self.summands = summands

        nonzero = tuple(
            summand
            for summand in summands
            if not summand.is_zero
        )

        if nonzero:
            self.source = nonzero[0].source
            self.target = nonzero[0].target

            for summand in nonzero[1:]:
                if summand.source != self.source:
                    raise ValueError(
                        "all nonzero summands must have "
                        "the same source"
                    )

                if summand.target != self.target:
                    raise ValueError(
                        "all nonzero summands must have "
                        "the same target"
                    )
        else:
            self.source = None
            self.target = None

    @property
    def is_zero(self):
        return all(
            summand.is_zero
            for summand in self.summands
        )

    @property
    def is_identity(self):
        return (
            len(self.summands) == 1
            and self.summands[0].is_identity
        )

    def __len__(self):
        return len(self.summands)

    def __iter__(self):
        return iter(self.summands)

    def __getitem__(self, index):
        return self.summands[index]

    def __str__(self):
        if not self.summands:
            return "0"

        return " ⊕ ".join(
            str(summand)
            for summand in self.summands
        )

    def __repr__(self):
        return f"DirectSum({self.summands!r})"

    def __eq__(self, other):
        if not isinstance(other, DirectSum):
            return NotImplemented

        return self.summands == other.summands

    def __hash__(self):
        return hash(self.summands)



class ShiftedUWord(Shifted):
    def __init__(self, word, q_shift=0):
        if not isinstance(word, UWord):
            raise TypeError("word must be a UWord")

        super().__init__(word, q_shift=q_shift)

    @property
    def word(self):
        return self.value

    def __repr__(self):
        return (
            f"ShiftedUWord({self.word!r}, "
            f"q_shift={self.q_shift!r})"
        )


class UDirectSum(DirectSum):
    def __init__(self, summands=()):
        summands = tuple(summands)

        if any(
            not isinstance(summand, ShiftedUWord)
            for summand in summands
        ):
            raise TypeError(
                "all summands must be ShiftedUWord objects"
            )

        super().__init__(summands)


class MorphismMatrix:
    '''
    A sparse matrix of morphisms between two DirectSums.

    Entry (row, column) is a morphism source[column].value -> target[row].value
    '''

    def __init__(self, source, target, entries=None):
        pass
