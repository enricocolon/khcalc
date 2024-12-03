from sage.knots.knot import Link
from sage.knots.knot import Knot
import sage.knots.knotinfo
from sage.knots.knotinfo import KnotInfo as KnotInfo #this might not be standard

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



K = KnotInfo.K3_1
tref = K.pd_notation()
vertex = 'AAB'
trefoil = Knot(tref)
#print(resolution(trefoil, vertex))

 #another idea you should write down somewhere. can you just make a graph from the crossings and then take faces in some way?
        #A type: [a,b,c,d]->a=b,c=d
        #B type: [a,b,c,d]->a=d,c=b

def dict_print(dicti):
    if type(dicti) != dict:
        raise Exception('Not a dictionary!')
    else:
        for i in dicti.keys():
            print(f'{i}: {dicti[i]}')

def pd_code(knotlike):
    '''
    Input: KnotInfo, Knot, or PD Code object
    Output: PD Code
    '''
    if isinstance(knotlike, Link):
        return knotlike.pd_code()
    elif type(knotlike) == KnotInfo:
        return knotlike.pd_notation()
    elif type(knotlike) == list:
        for i in knotlike:
            if len(i) != 4:
                raise Exception(f'Not a valid PD code, vertex {i} has {len(i)} components.')
    else:
        return knotlike


def pd_graph_better(knotlike):
    code = pd_code(knotlike)
    graph = dict()
    for i in range(1,len(code)):
        graph[f'v_{i}'] = dict()
    for vertex in code:
        #first incoming vertex is definitely sent to itself.
        #second incoming vertex is also sent to itself, but
        #its location depends on the sign of the crossing.
        #implement the modular sign-checker.
        pass

    
def pd_graph(knotlike):
    '''
    Input: PD code
    Output: Adjacency list with keys the vertices, labelled,
    and with values ((edge order), outgoing edges)

    OBSERVE: if a PD code has the edges ordered in the correct way,
    then a vertex will always contain some {i,i+1}. The edge
    goes to the vertex with edges {i+1,i+2}. Is the edge with
    {i+1,i+2} unique?
    '''
    code = pd_code(knotlike)
    graph = dict()
    for vertex in code:
        [a,b,c,d] = sorted(vertex)
        print([a,b,c,d])
        if a == 1 and b != 2:
            #YOU INHERENTLY MADE THE VERTICES NUMBER-DECORATED
            print('??')
            graph[tuple(vertex)] = {1, c}
        else:
            graph[tuple(vertex)] = {b, d}
    #note, once we convert to a pd_graph, the ordering
    #of the edges no longer matters. to convert back, we
    #must consider this.
    return graph



def resolution_1(pd_graph, vertex):
    '''
    Input: a pd_graph and a distinguished vertex
    [(a_1,a_2,a_3,a_4),{a_m,a_n}].
    Output: a pd_graph intermediary object... with the vertex
    resolved (a_1=a_4, a_2=a_3)
    '''
    [vert, edges] = vertex
    (a,b,c,d) = vert
    if edges in [{a,d},{b,c}]:
        sign = 1
    elif edges in [{a,b},{c,d}]:
        sign = 0
    else:
        raise Exception(f'Undefined crossing sign at {vert}')
    new_code = pd_graph
    del pd_graph[vertex]
    for new_vert in new_code.keys():
        new_edges = new_code[key]
        if sign == 0: #move for loop for efficiency
            if d in new_edges:
                new_edges = (new_edges-{d}).union({a})
            if c in new_edges:
                new_edges = (new_edges-{c}).union({b})
            for i in range(4):
                if new_vert[i] == d:
                     new_vert[i] = a
                if new_vert[i] == c:
                     new_vert[i] = b
        elif sign == 1:
            if d in new_edges:
                new_edges = (new_edges-{d}).union({c})
            if b in new_edges:
                new_edges = (new_edges-{b}).union({a})
            for i in range(4):
                if new_vert[i] == d:
                     new_vert[i] = c
                if new_vert[i] == b:
                     new_vert[i] = a

    #i dont think you took handedness into account
    return new_code

def resolution_2(pd_graph, vertex):
    '''
    Input: as before
    Output: vertex instead resolved to a thick edge.
    '''
    [vert, edges] = vertex
    [a,b,c,d] = vert
    new_code = pd_graph
    if edges in [{a,d},{b,c}]:
        sign = 1
    elif edges in [{a,b},{c,d}]:
        sign = 0
    else:
        raise Exception(f'Undefined crossing sign at {vert}')
    del new_code[vertex]
    if sign == 0:
        new_code[((a,b),a,b)] = {(a,b)}
        new_code[((a,b),c,d)] = {c,d}
    elif sign == 1:
        new_code[((d,a),d,a)] = {(d,a)}
        new_code[((d,a),b,c)] = {b,c}
    return new_code

#the roles of 1 and 0 ar eflipped, check with local tangle rotation
#bridge number and sl_n stabilization? conjecture
#KH: x^N, Gornik: x^N-1 (distinct roots), Lobb: gen. polynom
# potential w distinct roots: what foes at the vertices
# now has a canonical MOY-state basis
# nonatural basis? natural filtration.
# DISTINCT ROOTS GIVES:
# f -> (f(lam1),...f(lamn)), in C[x] mod p_N(x), get value at each root.

def get_resolution(pd_graph, word):
    '''
    Word is a list of len(pd_graph.keys()) 0's or 1's, corresponding
    to a resolution at each vertex.
    '''
    queue = pd_graph.keys()
    while queue:
        curr = queue.pop(0)
        curr_word = word.pop(0)
        if len(curr) != 4:
            continue
        else:
            pass
