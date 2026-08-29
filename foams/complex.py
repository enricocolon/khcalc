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
        if not isinstance(source, DirectSum):
            raise TypeError("source must be a DirectSum")

        if not isinstance(target, DirectSum):
            raise TypeError("target must be a DirectSum")

        if entries is None:
            entries = {}

        if not isinstance(entries, dict):
            raise TypeError("entries must be a dictionary")

        self.source = source
        self.target = target
        self.entries = dict(entries)

        self._validate_entries()

    def _validate_entries(self):
        for position, morphism in self.entries.items():
            if (
                not isinstance(position, tuple)
                or len(position) != 2
            ):
                raise TypeError(
                    "matrix positions must be "
                    "(row, column) tuples"
                )

            row, column = position

            if type(row) is not int:
                raise TypeError(
                    "row indices must be integers"
                )

            if type(column) is not int:
                raise TypeError(
                    "column indices must be integers"
                )

            if not 0 <= row < len(self.target):
                raise IndexError(
                    f"row index {row} is out of range"
                )

            if not 0 <= column < len(self.source):
                raise IndexError(
                    f"column index {column} is out of range"
                )

            for attribute in (
                "source",
                "target",
                "q_degree",
                "is_zero",
            ):
                if not hasattr(morphism, attribute):
                    raise TypeError(
                        "matrix entries must provide "
                        f"{attribute!r}"
                    )

            source_summand = self.source[column]
            target_summand = self.target[row]

            if morphism.source != source_summand.value:
                raise ValueError(
                    f"entry {(row, column)} has "
                    "the wrong source"
                )

            if morphism.target != target_summand.value:
                raise ValueError(
                    f"entry {(row, column)} has "
                    "the wrong target"
                )

            expected_degree = (
                target_summand.q_shift
                - source_summand.q_shift
            )

            if morphism.q_degree != expected_degree:
                raise ValueError(
                    f"entry {(row, column)} has "
                    f"quantum degree {morphism.q_degree}; "
                    f"expected {expected_degree}"
                )

    @property
    def shape(self):
        return (len(self.target), len(self.source))

    @property
    def is_zero(self):
        return all(
            morphism.is_zero
            for morphism in self.entries.values()
        )

    def __getitem__(self, position):
        if (
            not isinstance(position, tuple)
            or len(position) != 2
        ):
            raise TypeError(
                "matrix position must be (row, column)"
            )

        row, column = position

        if type(row) is not int or type(column) is not int:
            raise TypeError(
                "matrix indices must be integers"
            )

        if not 0 <= row < len(self.target):
            raise IndexError(
                f"row index {row} is out of range"
            )

        if not 0 <= column < len(self.source):
            raise IndexError(
                f"column index {column} is out of range"
            )

        return self.entries.get((row, column))

    def __repr__(self):
        return (
            f"MorphismMatrix("
            f"source={self.source!r}, "
            f"target={self.target!r}, "
            f"entries={self.entries!r})"
        )


    def __eq__(self, other):
        if not isinstance(other, MorphismMatrix):
            return NotImplemented

        return (
            self.source,
            self.target,
            self.entries,
        ) == (
            other.source,
            other.target,
            other.entries,
        )


class ChainComplex:
    '''
    WARNING: DIFFERENTIAL IS NOT VALIDATED
    '''
    def __init__(self, terms=None, differentials=None):
        if terms is None:
            terms = dict()

        if differentials is None:
            differentials = dict()

        if not isinstance(terms, dict):
            raise TypeError("terms must be a dictionary")

        if not isinstance(differentials, dict):
            raise TypeError("differentials must be a dictionary")

        for deg, term in terms.items():
            if type(deg) is not int:
                raise TypeError("homological degrees must be integers")

            if not isinstance(term, DirectSum):
                raise TypeError("chain terms must be DirectSum objects")

        for deg, differential in differentials.items():
            if type(deg) is not int:
                raise TypeError("differential degrees must be integers")

            if not isinstance(differential, MorphismMatrix):
                raise TypeError("differentials must be MorphismMatrix objects")

            if deg not in terms:
                raise ValueError(f"d^{deg} missing source term")

            if deg + 1 not in terms:
                raise ValueError(f"d^{deg} missing target term")

            if differential.source != terms[deg]:
                raise ValueError(f"d^{deg} has wrong source, expected {terms[deg]!r}, got {differential.source!r}")

            if differential.target != terms[deg + 1]:
                raise ValueError(f"d^{deg} has wrong target, expected {terms[deg+1]!r}, got {differential.target!r}")

        self.terms = dict(terms)
        self.differentials = dict(differentials)

    @property
    def degrees(self):
        return tuple(sorted(self.terms))

    @property
    def min_degree(self):
        return min(self.terms) if self.terms else None

    @property
    def max_degree(self):
        return max(self.terms) if self.terms else None

    @property
    def is_zero(self):
        return all(term.is_zero for term in self.terms.values())

    def term(self, degree):
        return self.terms.get(degree)

    def differential(self, degree):
        return self.differentials.get(degree)

    def __eq__(self, other):
        if not isinstance(other, ChainComplex):
            return NotImplemented

        return (
            self.terms,
            self.differentials
        ) == (
            other.terms,
            other.differentials
        )

    def __repr__(self):
        return (f"ChainComplex(terms={self.terms!r}," +
                f"differentials={self.differentials!r})")
