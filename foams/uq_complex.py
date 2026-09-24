#!/usr/bin/env python3

from .boundary import WebBoundary
from .complex import Shifted, DirectSum, MorphismMatrix, ChainComplex
from .uq import UWord, UTwoMorphism, IdentityU2, ZeroU2

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

def u_identity_matrix(obj):
    if not isinstance(obj, UDirectSum):
        raise TypeError("obj must be a UDirectSum")

    entries = {}

    for i, summand in enumerate(obj):
        if not summand.is_zero:
            entries[(i,i)] = UTwoMorphism(
                source=summand.word,
                target=summand.word,
                expression=IdentityU2(),
                q_degree=0,
            )

    return MorphismMatrix(obj, obj, entries)
#TK: make non-hardcoded identity constructor

def u_identity_complex(boundary):
    if not isinstance(boundary, WebBoundary):
        raise TypeError("boundary must be a WebBoundary")

    identity_word = UWord(
        source=boundary,
        factors=(),
    )

    term = UDirectSum((
        ShiftedUWord(
            identity_word,
            q_shift=0,
        ),
    ))

    return ChainComplex(
        terms={0: term},
        differentials={},
    )
