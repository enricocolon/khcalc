from sage.knots.knot import Link
from sage.knots.knot import Knot
import sage.knots.knotinfo
from sage.knots.knotinfo import KnotInfo as KnotInfo #this might not be standard

#sage.knots.link.Link.new_method = new_method


def resolution_depricated(knot, vertex):
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
#print(resolution_depricated(trefoil, vertex))

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

def sign(crossing):
    '''
    Input: a crossing [n, m, n+1, m+/-1] and the number of
    edges in the
    Output: sign of the crossing
    '''
    #this function does not check for valid crossings. proceed
    #with caution.
    [a,b,c,d] = crossing

    if b - d in {1,-1}:
        return b - d
    else:
        return (d-b)/abs(d-b) #cyclic returning
#FIX NAMING CONVENTION: its KNOT to X, not PD to X.
def pd_to_adj_list(knotlike):
    code = pd_code(knotlike)
    pre_graph = dict()
    graph = dict()
    edges = []
    for i in range(1,len(code)+1):
        [a,b,c,d] = code[i-1]
        signe = sign([a,b,c,d])
        graph[f'v_{i}'] = dict()
        if signe == 1:
            pre_graph[f'v_{i}'] = {'inputs': [a,d], 'outputs': [c,b]}
        elif signe == -1:
            pre_graph[f'v_{i}'] = {'inputs': [a,b], 'outputs': [c,d]}
    termini = dict()
    for vertex in pre_graph.keys():
        for edge in pre_graph[vertex]['inputs']:
            termini[edge] = vertex
    for vertex in pre_graph.keys():
        for edge in pre_graph[vertex]['outputs']:
            if termini[edge] not in graph[vertex].keys():
                graph[vertex][termini[edge]] = [edge]
            else:
                graph[vertex][termini[edge]].append(edge)
    return graph


def pd_to_graph(knotlike):
    code = pd_code(knotlike)
    dicti = pd_to_adj_list(code)
    graph = DiGraph(dicti, format='dict_of_dicts')
    return graph

#CURRENT ISSUE: how to handle GRAPH EMBEDDINGS when the
#DIAGRAM IS NOT SIMPLE?????
    
def pd_graph_depricated(knotlike):
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


def sign(crossing, edge_pairs):
    '''
    Input: Crossing [a,b,c,d]
    edge_pairs: Set containing pairs b->d if b and d are not numerically sequential but close a loop.
    Return: +/-1 the sign of the crossing.

    Warning: this only works assuming the labels in your PD code are cyclic in each component.
    THIS ALSO DOES NO CHECKS FOR VALIDITY.
    '''

    #cant do these until i check for hopf links
    [a,b,c,d] = crossing
    if (d,b) in edge_pairs:
        return 1
    elif (b,d) in edge_pairs:
        return -1
    elif d == b+1:
        return -1
    elif b == d+1:
        return 1
    else:
        raise Exception(f'Error computing sign at crossing f{crossing}')


def knot_to_ktg(knotlike):
    '''
    Input: Knot
    Output: Knotted trivalent graph, where each +/- crossing is replaced by a +/-1/2-framed
    edge, whose respective neighborhoods are the ingoing and outgoing crossings.
    TK: elaborate
    '''
    code = pd_code(knotlike)
    ktg = dict()
    edge_pairs = set()
    signs = []
    edge_orientations = {i:(None,None) for i in range(1,2*len(code)+1)}
    for vert in code: #here I get the exceptional loops and edge termini.
        [a,b,c,d] = vert
        edge_pairs.add((a,c))
        if (b,d) not in edge_pairs and (d,b) not in edge_pairs:
            if b == d+1:
                edge_pairs.add((d,b))
            if d == b+1:
                edge_pairs.add((b,d))
            if abs(b-d) != 1:
                [m,n] = sorted([b,d])
                edge_pairs.add((n,m))
        if (b,d) in edge_pairs: #i think this is how you fix hopf
            edge_pairs.add((d,b))
        if (d,b) in edge_pairs:
            edge_pairs.add((b,d))

        ktg[f't_{vert}_out'] = dict()
        ktg[f't_{vert}_in'] = dict()
        ktg[f't_{vert}_in'][f't_{vert}_out'] = [f't_{vert}']

    for vert in code:
        [a,b,c,d] = vert
        edge_orientations[a][1] = vert
        edge_orientations[c][0] = vert
        if (d,b) in edge_pairs and (b,d) in edge_pairs: #continue fixing hopf
            raise Exception('Meridional components are not yet supported.')
        if (d,b) in edge_pairs:
            signs.append(1)
            edge_orientations[d][1] = vert
            edge_orientations[b][0] = vert
        if (b,d) in edge_pairs: #elif? do i get hopf issues?
            signs.append(-1)
            edge_orientations[b][1] = vert
            edge_orientations[d][0] = vert

    for edge in range(1,2*len(code)+1):
        pass
        #connect the graph. this should be it! the code below is probably depricated.

    print(edge_pairs)
    for vert in code:
        [a,b,c,d] = vert
        signe = sign([a,b,c,d], edge_pairs)
        ktg[f't_{vert}_out'][f't_{termini[c]}_in'] = [c]
        if signe == 1:
            ktg[f't_{vert}_out'][f't_{termini[b]}_in'] = [b]
        elif signe == -1:
            ktg[f't_{vert}_out'][f't_{termini[d]}_in'] = [d]
    return ktg
#main takeaway: you probably need to code something that keeps track of which cycles
#are knot components/ this will fix the missing half part. Something like "if they have"
#intersection that would be like a 2-cycle, add it to the partition, otherwise record "gap" pairs

def knot_to_ktg_new(knotlike):
    code = pd_code(knotlike)
