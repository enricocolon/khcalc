#!/usr/bin/env python3

from .foam import LadderRung, Web
from .uq import DividedPower, UWord

def foamate_divided_power(divided_power):
    if not isinstance(divided_power, DividedPower):
        raise TypeError("divided_power must be a DividedPower object")

    direction = divided_power.direction
    i = divided_power.i
    r = divided_power.r
    source = divided_power.source

    return LadderRung(direction, i, r, source)

def foamate_uword(uword):
    if not isinstance(uword, UWord):
        raise TypeError("uword must be a UWord object")

    source = uword.source
    factors = uword.factors

    rungs = [foamate_divided_power(factor) for factor in factors]

    return Web(source, tuple(rungs))
