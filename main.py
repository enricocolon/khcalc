from sage.knots.knot import Link
from sage.knots.knot import Knot
import sage.knots.knotinfo
from sage.knots.knotinfo import KnotInfo as KnotInfo #this might not be standard
import copy

#sage.knots.link.Link.new_method = new_method

K = KnotInfo.K3_1

def dict_print(dicti):
    if type(dicti) != dict:
        raise Exception('Not a dictionary!')
    else:
        for i in dicti.keys():
            print(f'{i}: {dicti[i]}')

def list_print(lst):
    if type(lst) != list:
        raise Exception('Not a list!')
    else:
        for i in lst:
            print(i)

def printt(var):
    if type(var) == dict:
        dict_print(var)
    elif type(var) == list:
        list_print(var)
    else:
        print(var)

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


def knot_to_init_resolution(knotlike):
    '''
    INPUT: A knot or knotinfo object, or a PD code.
    OUTPUT: An oriented knot resolution, orientation induced by
    the order of edge labels in the PD code,
    '''
    code = pd_code(knotlike)
    resolution = DiGraph()
    edge_history = set()
    for crossing in code:
        [a,b,c,d] = crossing
        for e in crossing:
            resolution.add_vertex(e)
        if (b,d) in edge_history or (d,b) in edge_history: #do for AC too or no?
            raise Exception('Meridional components are not yet supported.')
        elif d == b + 1:
            #negative crossing
            resolution.add_edges([(a,f'{crossing}_in'),
                                  (b,f'{crossing}_in'),
                                  (f'{crossing}_in',f'{crossing}_out'),
                                  (f'{crossing}_out',c),
                                  (f'{crossing}_out',d),#add this down the way
                                  ])
            resolution.set_vertex(f'{crossing}_in',{'in_orientation':(a,b)})
            resolution.set_vertex(f'{crossing}_out',{'out_orientation':(c,d)})
            #out (resp in)_orientation is the clockwise orientation of the
            #out (resp in) neighbors of the vertex.
            edge_history.add(tuple((b,d)))
        elif b == d + 1:
            #postive crossing
            resolution.add_edges([(d,f'{crossing}_in'),
                                  (a,f'{crossing}_in'),
                                  (f'{crossing}_in',f'{crossing}_out'),
                                  (f'{crossing}_out',b),
                                  (f'{crossing}_out',c),
                                  ])
            resolution.set_vertex(f'{crossing}_in',{'in_orientation':(d,a)})
            resolution.set_vertex(f'{crossing}_out',{'out_orientation':(b,c)})
            edge_history.add(tuple((d,b)))
        else:
            if b > d:
                #negative crossing
                resolution.add_edges([(a,f'{crossing}_in'),
                                      (b,f'{crossing}_in'),
                                      (f'{crossing}_in',f'{crossing}_out'),
                                      (f'{crossing}_out',c),
                                      (f'{crossing}_out',d),
                                      ])
                resolution.set_vertex(f'{crossing}_in',{'in_orientation':(a,b)})
                resolution.set_vertex(f'{crossing}_out',{'out_orientation':(c,d)})
                edge_history.add(tuple((b,d)))
            elif d > b:
                #positive crossing
                resolution.add_edges([(d,f'{crossing}_in'),
                                      (a,f'{crossing}_in'),
                                      (f'{crossing}_in',f'{crossing}_out'),
                                      (f'{crossing}_out',b),
                                      (f'{crossing}_out',c),
                                      ])
                resolution.set_vertex(f'{crossing}_in',{'in_orientation':(d,a)})
                resolution.set_vertex(f'{crossing}_out',{'out_orientation':(b,c)})
                edge_history.add(tuple((d,b)))
        edge_history.add(tuple((a,c)))
    return resolution

def vertex_contract(graph, vert):
    '''
    Input: a graph, and a vertex with one in-neighbor and one out-neighbor.
    Output: a graph with the vertex removed and edges resolved in a way that respects orientation.

    v_1 -> v_2 -> v_3 ~~~~> v_1 ---> v_3

    NOTE: the operation name may not be standard. Check this later.
    '''
    in_edges = graph.incoming_edges(vert)
    out_edges = graph.outgoing_edges(vert)

    if not len(in_edges) == len(out_edges) == 1:
        raise Exception('Vertex cannot be contracted.')

    graph.add_edge(in_edges[0][0],out_edges[0][1],in_edges[0][2])
    graph.delete_vertex(vert)
    return graph

def trivalent_unzip(graph, edge):
    '''
    Input: a directed graph and an edge (u,v) where u (resp. v)
    has degree 3, with two in (resp. out) neighbors.
    Output: a new trivalent graph unzipped along (u,v).
    '''
    (u, v, label) = edge
    graph.add_edge((f'{u}_temp',f'{v}_temp',label))
    try:
        if not len(graph.get_vertex(u)['in_orientation']) == len(graph.get_vertex(v)['out_orientation']) == 2:
            raise Exception('Vertices do not have the correct degrees.')
    except Exception as e:
        print(f'An error occured: {e}')
    (m,n) = graph.get_vertex(u)['in_orientation']
    (q,r) = graph.get_vertex(v)['out_orientation']
    #print(m,n,q,r,u,v)
    n_label = graph.edge_label(n,u)
    q_label = graph.edge_label(v,q)
    graph.add_edge(n, f'{u}_temp', n_label)
    graph.add_edge(f'{v}_temp', q, q_label)
    graph.delete_edge(n,u)
    graph.delete_edge(v,q)
    vertex_contract(graph, u)
    vertex_contract(graph, f'{u}_temp')
    vertex_contract(graph, v)
    vertex_contract(graph, f'{v}_temp')
    return graph

def tri_unzip(graph, crossing):
    '''
    Shorthand, only requires the crossing. MAKE SURE IT IS LABEL-COMPATIBLE.
    '''
    crossing = list(crossing)
    edge = (f'{crossing}_in', f'{crossing}_out', None)
    return trivalent_unzip(graph, edge)

def graph_traversal(graph):
    '''
    NOTE: this requires wide edges to have vertices '[a, b, c, d]_in' and '[a, b, c, d]_out'
    '''
    visited = set()
    queue = []
    vertex_set = set(graph.vertices())
    components = list()
    while visited != vertex_set:
        if curr[0].endswitch('_in','_out'):
            source = curr[0][:12] + '_in'
            target = curr[0][:12] + '_out'
            s_neigh = graph.neighbors(source)
            t_neigh = graph.neighbors(target)
            component_vertices = s_neigh + t_neigh
            visited.add(source)
            visited.add(target)
            for vertex in set(component_vertices)-{source, target}:
                if vertex not in visited:
                    queue.append(vertex)

        elif curr[1].endswitch('_in','_out'):
            source = curr[1][:12] + '_in'
            target = curr[1][:12] + '_out'
            s_neigh = graph.neighbors(source)
            t_neigh = graph.neighbors(target)
            component_vertices = s_neigh + t_neigh
            visited.add(source)
            visited.add(target)
            for vertex in set(component_vertices)-{source, target}:
                if vertex not in visited:
                    queue.append(vertex)
        else:
            pass


A = knot_to_init_resolution(K)
