from sage.knots.knot import Link
from sage.knots.knot import Knot
import sage.knots.knotinfo as KnotInfo #this might not be standard

#sage.knots.link.Link.new_method = new_method


def resolution(knot, vertex):
    '''
    Input: A link (given as a Link object's PD code) and a vertex of the cube [0,1]^#crossings(L),
    represented by a string of A's and B's (representing 0 and 1 respectively) of
    length #crossings(L).
    Output: A module V^otimes{#pi_0(resolution(L))} corresponding to the resolution given by a disjoint
    union of #pi_0(resolution(L)) circles induced by the vertex. (see: def. of 1+1-dim TQFT)
    '''
    crossings = knot.pd_code() #input is a KNOT object--not a knotinfo object
    maxes = [max(j) for j in crossings]
    maxx = max(maxes)

    resolutions = [i for i in vertex]
    components = set(range(1,maxx+1))
    while len(resolutions) > 0:
        curr_vertex = crossings.pop(0)
        resolution_now = resolutions.pop(0)
        [a,b,c,d] = curr_vertex
        components.remove(b)
        components.remove(d)
        for crossing in crossings:
            if b in crossing:
                if resolution_now == 'A':
                    crossing[crossing.index(b)] = a
                    components.remove(b)
                elif resolution_now == 'B':
                    crossing[crossing.index(b)] = c
                    components.remove(b)
                else:
                    raise Exception(f"{resolution_now} is not a valid resolution.")
            elif d in crossing:
                if resolution_now == 'A':
                    crossing[crossing.index(d)] = c
                    components.remove(d)
                elif resolution_now == 'B':
                    crossing[crossing.index(d)] = a
                    components.remove(d)
                else:
                    raise Exception(f"{resolution_now} is not a valid resolution.")
            else:
                continue
    return len(components)



K = KnotInfo.KnotInfo.K3_1
tref = K.pd_notation()
vertex = 'AAB'
trefoil = Knot(tref)
print(resolution(trefoil, vertex))

 #another idea you should write down somewhere. can you just make a graph from the crossings and then take faces in some way?
        #A type: [a,b,c,d]->a=b,c=d
        #B type: [a,b,c,d]->a=d,c=b
