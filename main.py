from sage.all import *

#sage.knots.link.Link.new_method = new_method


def resolution(self, vertex):
    '''
    Input: A link (given as a Link object's PD code) and a vertex of the cube [0,1]^#crossings(L),
    represented by a string of A's and B's (representing 0 and 1 respectively) of
    length #crossings(L).
    Output: A module V^\otimes{#pi_0(resolution(L))} corresponding to the resolution given by a disjoint
    union of #pi_0(resolution(L)) circles induced by the vertex. (see: def. of 1+1-dim TQFT)
    '''
    resolution_queue = vertex
    while len(resolution_queue) > 0:
        resolution_now = vertex.pop(0)
        if resolution_now == 'A':
            pass
        elif resolution_now == 'B':
            pass
        else:
            raise Exception(f'{vertex} is not a valid vertex.')
