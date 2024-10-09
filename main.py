from sage.all import *

#sage.knots.link.Link.new_method = new_method


def resolution(self, vertex):
    '''
    Input: A link (given as a Link object's PD code) and a vertex of the cube [0,1]^#crossings(L),
    represented by a string of A's and B's (representing 0 and 1 respectively) of
    length #crossings(L).
    Output: A module V^otimes{#pi_0(resolution(L))} corresponding to the resolution given by a disjoint
    union of #pi_0(resolution(L)) circles induced by the vertex. (see: def. of 1+1-dim TQFT)
    '''
    crossings = self.pd_code()
    maxes = [max(j) for j in crossings]
    maxx = max(maxes)

    resolutions = vertex
    components = range(1,maxx)
    while len(resolution_queue) > 0:
        curr_vertex = crossings.pop(0)
        curr_res = resolutions.pop(0)
        [a,b,c,d] = curr_vertex
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
                pass
    return len(components)


        
        #A type: [a,b,c,d]->a=b,c=d
        #B type: [a,b,c,d]->a=d,c=b
