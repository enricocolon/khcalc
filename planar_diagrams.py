#!/usr/bin/env python3


class BoundaryPoint():
    def __init__(self, name, sign):
        self.name = name
        self.arc = None
        self.sign = sign

    def setArc(self, arc):
        self.arc = arc
        if self.sign == 1:
            arc.setSink(self)
        else:
            arc.setSource(self)

    def __repr__(self):
        return self.name


class Arc():
    def __init__(self, name):
        self.name = name
        self.source = None
        self.sink = None

    def setSource(self,source):
        if source.sign != -1:
            raise Exception(f'source {source} has incorrect sign')
        self.source = source
        source.arc = self

    def setSink(self,sink):
        if sink.sign != 1:
            raise Exception(f'source {source} has incorrect sign')
        self.sink = sink
        sink.arc = self

    def setEnds(self, bp1,bp2):
        if bp1.sign == 1:
            self.setSink(bp1)
            self.setSource(bp2)
        else:
            self.setSource(bp1)
            self.setSink(bp2)

    def __repr__(self):
        return self.name

class PlanarDiagram():
    def __init__(self, output_word, input_words, arcs, unknot_components):
        # output_word: a word in {+,-} specifying orientations of boundary points of the outward-boundary.
        # input_words: a list of words in {+,-} "" inward-boundaries. the ordering of the components is
        # implicit in the order of the list. marked points on each disk are natural once a first-element is
        # picked.
        # arcs: a matching of points - -> +. [how to represent...]
        # unknot_components: an integer. additive under operad sum.
        self.output = output_word
        self.inputs = input_words
        self.unknots = unknot_components
        self.arcs = None

    def compose(self, other, site):
        #check that output of other is opposite of site (input on self) word
        #identify arcs: ???
        #add unknot components
        pass
