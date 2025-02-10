from sage.knots.knot import Link
from sage.knots.knot import Knot
import sage.knots.knotinfo
from sage.knots.knotinfo import KnotInfo as KnotInfo #this might not be standard
import copy
from sage.homology.chain_complex import ChainComplex
from sage.categories.category import Category
from sage.categories.morphism import Morphism
from sage.categories.modules import Modules
import datetime

now = datetime.datetime.now()
#sage.knots.link.Link.new_method = new_method

Z2 = IntegerModRing(2)


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
            resolution.set_vertex(f'{crossing}_in',{'in_orientation':(a,b), 'sign':'-'})
            resolution.set_vertex(f'{crossing}_out',{'out_orientation':(c,d), 'sign':'-'})
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
            resolution.set_vertex(f'{crossing}_in',{'in_orientation':(d,a), 'sign':'+'})
            resolution.set_vertex(f'{crossing}_out',{'out_orientation':(b,c), 'sign':'+'})
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
                resolution.set_vertex(f'{crossing}_in',{'in_orientation':(a,b), 'sign':'-'})
                resolution.set_vertex(f'{crossing}_out',{'out_orientation':(c,d), 'sign':'-'})
                edge_history.add(tuple((b,d)))
            elif d > b:
                #positive crossing
                resolution.add_edges([(d,f'{crossing}_in'),
                                      (a,f'{crossing}_in'),
                                      (f'{crossing}_in',f'{crossing}_out'),
                                      (f'{crossing}_out',b),
                                      (f'{crossing}_out',c),
                                      ])
                resolution.set_vertex(f'{crossing}_in',{'in_orientation':(d,a), 'sign':'+'})
                resolution.set_vertex(f'{crossing}_out',{'out_orientation':(b,c), 'sign':'+'})
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
        if not len(graph.get_vertex(u)['in_orientation']) == \
           len(graph.get_vertex(v)['out_orientation']) == 2:
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

def graph_traversal(graph, start_crossing):
    '''
    NOTE: this requires wide edges to have vertices '[a, b, c, d]_in' and '[a, b, c, d]_out'
    '''
    start_in = str(start_crossing) + '_in'
    start_out = str(start_crossing) + '_out'
    in_neigh = set(graph.neighbors(start_in))
    out_neigh = set(graph.neighbors(start_out))
    neigh = in_neigh.union(out_neigh)-{start_in,start_out}
    visited = {start_in, start_out}
    queue = list(neigh)
    vertex_set = set(graph.vertices())
    components = [f'C_t_{start_crossing}']
    while visited != vertex_set:
        if queue == []:
            raise Exception(f'Graph {graph} must be connected.')
        curr = queue.pop()
        neigh = graph.neighbors(curr)
        for vertex in neigh:
            if vertex not in visited:
                queue.append(vertex)
        if curr.endswith('_in', '_out'):
            components.append(f'C_t_{curr[:12]}')
        else:
            pass

def search(graph):
    visited = set()
    queue = [graph.vertices()[0]]

    while visited != set(graph.vertices()):
        if queue == []:
            differences = set(graph.vertices()) - visited
            queue.append(list(differences)[0])
        curr = queue.pop()
        if curr in visited:
            continue
        elif curr[-3] == '_' or curr[-4] == '_': #if this is a wide edge vertex
            pass
        else:
            pass
        for neigh in graph.neighbors(curr):
            queue.append(neigh)


def short_edge_factorization(source, target, n):
    '''
    given vertices A->B in a graph, returns chain complex L_A^B, that is,
    Q[x_A,x_B] -> Q[x_A,x_B] -> Q[x_A,x_B] with maps
    (x_B^(n+1)-x_A^(n+1))/(x_B-x_A) and x_B-x_A respectively. (KR Matrix Factorizations p. 49)
    '''
    R = PolynomialRing(QQ, [f'x{source}',f'x{target}'])
    x_source = 'x' + str(source)
    x_target = 'x' + str(target)
    x_source, x_target = R.gens()

    M0 = R
    M1 = R

    d0_poly = (x_target**(n+1) - x_source**(n+1))/(x_target - x_source)
    d1_poly = x_target - x_source

    d0 = M0.hom(Matrix(R,[d0_poly*x_source,d0_poly*x_target]),M1)
    d1 = M1.hom(Matrix(R,[d1_poly*x_source,d1_poly*x_target]),M0)
    print(d0,d1)
    factorization = ChainComplex({0:d0,1:d1},base_ring=R,grading_group=Z2,degree=1)

def anti_block(A,B): #DO NOT DELETE.
    '''
    Given A,B matrices, return the block matrix:

    [[0,A],
     [B,0]]
    '''
    if A.base_ring() != B.base_ring():
        raise Exception('A,B must have the same base ring.')
    m,n = A.nrows(),A.ncols()
    p,q = B.nrows(),B.ncols()
    tl = Matrix(A.base_ring(), m, q)
    br = Matrix(A.base_ring(), p, n)
    return block_matrix([[tl, A],[B, br]])


def direct_sum(mod1,mod2): #DO NOT DELETE.
    '''
    given two free modules with a shared specified basis, return their direct sum.
    WARNING: only works if mod1, mod2 over the same basis.
    '''
    if mod1.base_ring() != mod2.base_ring():
        raise Exception('Modules must be over the same base ring.')
    R = mod1.base_ring()
    combined_basis = list(mod1.gens()) + list(mod2.gens())
    return CombinatorialFreeModule(R, combined_basis)

class MatrixFactorization():
    #TODO: Check for when R/(w) is not an isolated singularity. AND CHECK THAT THERE IS A BASIS.
    def __init__(self, R, M0, M1, d0, d1):
        if not M0 in Modules(R) or not M1 in Modules(R):
            raise Exception('M0 and M1 must be modules over R.')
        if not M0.rank() == d0.ncols() == d1.nrows() or \
           not M1.rank() == d1.ncols() == d0.nrows():
            raise Exception('d0 or d1 does not have the correct dimensions.')
        #below: check that d^2m=wm
        d = anti_block(d0, d1)
        d2 = d**2
        if d2 != d2[0][0]*identity_matrix(R, d0.ncols()+d1.ncols()):
            raise Exception('d^2m != wm')
        self.R = R
        self.w = d2[0][0]
        self.M0 = M0
        self.M1 = M1
        self.M = direct_sum(M0, M1)
        self.d0 = d0
        self.d1 = d1

    def tensor(self, other):
        if self.R != other.R:
            raise Exception('Your duplex modules must be over the same ring.')
        MN0 = direct_sum(self.M0.tensor(other.M0), self.M1.tensor(other.M1))
        MN1 = direct_sum(self.M1.tensor(other.M0), self.M0.tensor(other.M1))
        Id_M0 = identity_matrix(self.R, self.M0.rank())
        Id_M1 = identity_matrix(self.R, self.M1.rank())
        Id_N0 = identity_matrix(other.R, other.M0.rank())
        Id_N1 = identity_matrix(other.R, other.M1.rank())
        D0 = block_matrix([[-Id_M0.tensor_product(other.d0),self.d1.tensor_product(Id_N1)],
                           [self.d0.tensor_product(Id_N0),Id_M1.tensor_product(other.d1)]])
        D1 = block_matrix([[-Id_M0.tensor_product(other.d1),self.d1.tensor_product(Id_N0)],
                           [self.d0.tensor_product(Id_N1),Id_M1.tensor_product(other.d0)]])
        tprod = MatrixFactorization(self.R, MN0, MN1, D0, D1)
        return tprod

def pi(x_str,y_str,n):
    '''
    given variables x,y, and an integer n, returns
    pi_xy = x^{n+1}-y^{n+1}/x-y=x^n+x^{n-1}y+...+y^n
    '''
    #x, y = var('x y')
    x, y = var(x_str), var(y_str)
    A = 0
    for i in range(n+1):
        A += x**(n-i)*y**(i)
    return A

def g(x_str,y_str,n):
    '''
    input: variables
    output: a function g such that g(x+y,xy)=x^{n+1}+y^{n+1}
    '''
    x, y = var(x_str), var(y_str)
    g = x**(n+1)
    g_1 = 0
    for i in range(1,floor((n+1)/2)+1):
        g_1 += ((-1)**(i)*binomial(n-i,i-1)*y**(i)*x**(n+1-2*i))*i**-1
    g += (n+1)*g_1
    return g.expand()

def u_1(x_str,y_str,z_str,w_str,n):
    x, y, z, w = var(x_str), var(y_str), var(z_str), var(w_str)
    return (g(x+y,x*y,n)-g(z+w,x*y,n))*(x+y-z-w)**(-1)

def u_2(x_str,y_str,z_str,w_str,n):
    x, y, z, w = var(x_str), var(y_str), var(z_str), var(w_str)
    return (g(z+w,x*y,n)-g(z+w,z*w,n))*(x*y-z*w)**(-1)

# u_1*(x+y+z-w)+u_2(xy-zw)=x^{n+1}+y^{n+1}-z^{n+1}-w^{n+1}

#TODO: implement tensor products Q[_] tensor_Q[_] Q'[_]. Take tensor products. Cohomology.
#TODO: figure out when you can tensor over a different base ring and implement that.
#I think we're in business, because we have a smallest ring--Q--so all of the data should be contained
#in that tensor product. IDEA: store the d-maps as maps over the base ring Q. Tensor over bigger
#rings using minors.
A = knot_to_init_resolution(K)

Qi = CombinatorialFreeModule(QQ, ['xi'])
Qj = CombinatorialFreeModule(QQ, ['xj'])

MF = MatrixFactorization(QQ, Qi, Qj, Matrix(QQ, [2]), Matrix(QQ, [4]))
MF2= MatrixFactorization(QQ, Qi, Qj, Matrix(QQ, [3]), Matrix(QQ, [6]))

R = QQ['x','y']
R2 = QQ['y','z']
M = CombinatorialFreeModule(QQ['x','y'],['e'])
Ma = CombinatorialFreeModule(QQ['x','y'],['a'])
M2 = CombinatorialFreeModule(R2,['e'])
M2b = CombinatorialFreeModule(R2,['b'])
Lxy = MatrixFactorization(R,M,Ma, Matrix(R, [pi('x','y',4)]), Matrix(R, [x-y]))

Lyz = MatrixFactorization(R2,M2,M2b, Matrix(R2, [pi('y','z',4)]), Matrix(R2, [y-z]))

#Lxyyz = Lxy.tensor(Lyz)

class MatFact():
    def __init__(self, d0, d1):
        for d in (d0,d1):
            if not isinstance(d, sage.matrix.matrix0.Matrix):
                raise Exception(f'{d} is not a Matrix.')
        if not d0.nrows() == d0.ncols() == d1.nrows() == d1.ncols():
            raise Exception(f"{d0} and {d1} must be square matrices of the same size.")
        if not d0.base_ring() == d1.base_ring():
            raise Exception(f"{d0} and {d1} must take entries in the same ring.")
        self.R = d0.base_ring()
        d = anti_block(d0, d1)
        d2 = d**2
        self.d0 = d0
        self.d1 = d1
        if d2 != d2[0][0]*identity_matrix(self.R, self.d0.ncols()+self.d1.ncols()):
            raise Exception(f'd^2m != wm\nd^2=\n{d2}')
        self.w = d2[0][0]


        self.rank = self.d0.ncols()

    def __repr__(self):
        return f"{self.R} -> {self.R} -> {self.R},\nd0=\n{self.d0}\nd1=\n{self.d1}"

    def tensor(self, other):
        if self.R != other.R:
            raise Exception(f'Your duplex modules must be over the same ring.\nself.R = {self.R}\nother.R = {other.R}')

        Im = identity_matrix(self.R, self.rank)
        In = identity_matrix(self.R, other.rank)

        D0 = self.d0.tensor_product(In)
        D1 = self.d1.tensor_product(In)
        E0 = Im.tensor_product(other.d0)
        E1 = Im.tensor_product(other.d1)

        d0 = block_matrix(self.R,[[D0, E1],[E0, -1*D1]])
        d1 = block_matrix(self.R,[[D1, E1],[E0, -1*D0]])
        return MatFact(d0, d1)

    def get_external_tensor_ring(self, other):
        if self.R.base() != other.R.base():
            raise Exception("Matrix Factorization must be over polynomial rings of the same base.")
        base = self.R.base()
        gens1 = list(self.R.gens())
        gens2 = list(other.R.gens())
        gens = gens1
        gens_strs = [str(i) for i in gens]
        for i in gens2:
            if str(i) not in gens_strs:
                gens.append(i)
                gens_strs.append(str(i))
        print(base, gens,gens2,gens1)
        print(gens)
        R = base[gens]
        return R


    def direct_sum(self, other):
        #if self.R != other.R:
        #    raise Exception('Base rings must match.')
        #if self.w != other.w:
        #    raise Exception('Factorizations must have same w.')
        R = self.get_external_tensor_ring(other) #<- DIDNT WORK BUT THIS IDEA
        d0 = block_matrix(R, [[self.d0, 0], [0, other.d0]])
        d1 = block_matrix(R, [[self.d1, 0], [0, other.d1]])
        return MatFact(d0, d1)

    @abstract_method
    def angle_shift(self, i=1):
        if i % 2 == 0:
            return self
        if i % 2 == 1:
            return MatFact(-self.d1, -self.d0)

    def external_tensor(self, other):
        R = self.get_external_tensor_ring(other)
        #print(R,self.d0.change_ring(R), self.d1.change_ring(R),other.d0.change_ring(R), other.d0.change_ring(R),other.d1.change_ring(R).base_ring())
        mf1 = MatFact(self.d0.change_ring(R), self.d1.change_ring(R))
        #print(mf1.d0, mf1.d0.base_ring(), mf1.R, "!")
        mf2 = MatFact(other.d0.change_ring(R), other.d1.change_ring(R))
        return mf1.tensor(mf2)


def label_tensor(list1,list2):
    '''
    given list1, list2 lists of strings, returns a list of length len(list1)*len(list2) with entries
    [l1[0]+l2[0],l1[1]+l2[0],..., l1[n]l2[0].........l1[n]l2[n]]
    '''
    return [l1 + l2 for l2 in list2 for l1 in list1]


def listsum(lst1, lst2):
    if len(lst1) != len (lst2):
        raise Exception('dimensions must match')
    return [lst1[i]+lst2[i] for i in range(len(lst1))]


#IMPLEMENT A CHECK THAT MAKES SURE PROPER # OF LABELS AND GRADING SHIFTS EXISTS.

class LabelMF(MatFact):
    def __init__(self, d0, d1, labels_0,labels_1):
        MatFact.__init__(self, d0, d1)
        self.labels_0 = labels_0
        self.labels_1 = labels_1

    def tensor(self,other):
        lab_0 = label_tensor(self.labels_0, other.labels_0) + \
            label_tensor(self.labels_1,other.labels_1)
        lab_1 = label_tensor(self.labels_1, other.labels_0) + \
            label_tensor(self.labels_0,other.labels_1)
        mf = super().external_tensor(other)
        return LabelMF(mf.d0, mf.d1, lab_0, lab_1)

    def labels(self):
        return {0: self.labels_0, 1: self.labels_1}


class GradedMatFact(MatFact):
    def __init__(self, d0, d1, deg_shift_0, deg_shift_1):
        MatFact.__init__(self, d0, d1)
        if not is_homogeneous(self.w):
            raise Exception(f'w must be homogeneous.\nw = {w}')
        if len(list(deg_shift_0)) != self.d0.nrows() or \
           len(list(deg_shift_1)) != self.d1.nrows() or \
           not type(deg_shift_0)==type(deg_shift_1)==list:
            print(deg_shift_0,deg_shift_1,self.d0.nrows(),self.d1.nrows())
            raise Exception('Degree shifts must be lists of rank length.')


        #curly brace shift
        self.deg_shift_0 = deg_shift_0
        self.deg_shift_1 = deg_shift_1


    def tensor(self, other, shift0=None, shift1=None):
        ds_0 = label_tensor(self.deg_shift_0, other.deg_shift_0) + \
            label_tensor(self.deg_shift_1, other.deg_shift_1)
        ds_1 = label_tensor(self.deg_shift_1, other.deg_shift_0) + \
            label_tensor(self.deg_shift_0, other.deg_shift_1)
        mf = super().external_tensor(other)
        #print(list(ds0),list(ds1))i
        if not shift:
            return GradedMatFact(mf, ds_0, ds_1)
        if shift0 or shift1:
            if not len(shift0) == len(shift1) == len(ds_0):
                raise Exception('shift0, shift1 must be an appropriately sized list of degree shifts')
            else:
                return GradedMatFact(mf, listsum(ds0,shift0), listsum(ds1,shift1))

def is_homogeneous(powsrs):
    '''
    Given a PowerSeries or MPowerSeries object, returns True if homogeneous and False otherwise.
    '''
    exponents = powsrs.exponents()
    degrees = [sum(exp) for exp in exponents]
    if len(set(degrees)-{0}) <= 1:
        return True
    else:
        return False


#TODO: handle homogeneity/different rings (cf: KR Matrix Factorizations Prop 9? 10?)

class LabelGMF(LabelMF, GradedMatFact):
    def __init__(self, d0, d1, labels_0, labels_1, deg_shift_0, deg_shift_1):
        LabelMF.__init__(self, d0, d1, labels_0, labels_1)
        GradedMatFact.__init__(self, d0, d1, deg_shift_0, deg_shift_1)

    def tensor(self, other, shift0=None, shift1=None):
        R = self.get_external_tensor_ring(other)
        #^^^ MAKE SURE THIS FIX IS APPLIED EVERYWHERE.
        lab_0 = label_tensor(self.labels_0, other.labels_0) + \
            label_tensor(self.labels_1,other.labels_1)
        lab_1 = label_tensor(self.labels_1, other.labels_0) + \
            label_tensor(self.labels_0,other.labels_1)
        ds_0 = label_tensor(self.deg_shift_0, other.deg_shift_0) + \
            label_tensor(self.deg_shift_1, other.deg_shift_1)
        ds_1 = label_tensor(self.deg_shift_1, other.deg_shift_0) + \
            label_tensor(self.deg_shift_0, other.deg_shift_1)
        mf = MatFact.external_tensor(self,other)
        if is_homogeneous(R(self.w) + R(other.w)):
            if shift0 or shift1:
                if not len(shift0) == len(shift1) == len(ds_0):
                    raise Exception('shift0, shift1 must be an appropriately sized list of degree shifts')
                return LabelGMF(mf.d0, mf.d1, lab_0, lab_1, listsum(ds_0,shift0), listsum(ds_1,shift1))
            return LabelGMF(mf.d0, mf.d1, lab_0, lab_1, ds_0, ds_1)
            #return LabelMF.tensor(self, other)
        else:
            return LabelMF.tensor(self, other)

    def angle_shift(self, i=1):
        if i % 2 == 0:
            return self
        if i % 2 == 1:
            return LabelGMF(-self.d1, -self.d0, self.labels_1, self.labels_0, self.deg_shift_1, self.deg_shift_0)

    def __repr__(self):
        return f"R({self.labels_0}){{{self.deg_shift_0}}} -> R({self.labels_1}){{{self.deg_shift_1}}}" + \
            f"-> R({self.labels_0}){{{self.deg_shift_0}}}" + f"\nR: {self.R}\nd0:\n{self.d0}" + \
            f"\nd1:\n{self.d1}"

    def direct_sum(self, other):
        #FIX!!!!! this is preventing you from implementing complexes
        mf = MatFact.direct_sum(self,other)
        return LabelGMF(mf.d0, mf.d1, self.labels_0 + other.labels_0, self.labels_1 + other.labels_1, \
                        self.deg_shift_0 + other.deg_shift_0, self.deg_shift_1 + other.deg_shift_1)

    def curly_shift(self, arr0, arr1):
        if len(arr0)!= len(self.deg_shift_0) or len(arr1) != len(self.deg_shift_1):
            raise Exception('Degree shifts must be of appropriate size.')
        return LabelGMF(self.d0, self.d1, self.labels_1, self.labels_0, listsum(self.deg_shift_0, arr0), \
                        listsum(self.deg_shift_1, arr1))


    def dual(self):
        '''
        consider the way this changes labels and gradings
        '''
        #return LabelGMF(self.d1.transpose(), self.d0.transpose(), )
        #TODO: Make angle shifts, duals, respect grading shifts


def Ct(x_str, y_str, z_str, w_str, n):
    x, y, z, w = var(x_str), var(y_str), var(z_str), var(w_str)
    R = QQ[x,y,z,w]
    U1 = Matrix(R,[u_1(x,y,z,w,n)])
    U2 = Matrix(R,[u_2(x,y,z,w,n)])
    UU1 = Matrix(R, [x+y-z-w])
    UU2 = Matrix(R, [x*y-z*w])
    Rt1 = LabelGMF(U1, UU1, [''],[''],[0],[1-n])
    Rt2 = LabelGMF(U2, UU2, [''],[''],[0],[3-n])
    return Rt1.tensor(Rt2, [-1,-1],[-1,-1])


def Lij(xi_str, xj_str,n):
    '''
    Matrix Factorization L^i_j assigned to arc going from j to i.
    '''
    xi, xj = var(xi_str), var(xj_str)
    R = QQ[xi, xj]
    pixixj = Matrix(R, [pi(xi,xj,n)])
    xixj = Matrix(R, [xi-xj])
    return LabelGMF(pixixj, xixj, [''],[''],[0],[1-n])


class LabelGMFComplex():
    def __init__(self, dct):
        for i in dct.keys():
            if not i in ZZ:
                raise Exception(f'Complex grading {i} is not valid')
            if not isinstance(dct[i], LabelGMF):
                raise Exception(f'All dictionary entries must be LabelGMFs')

        self.complex = dct

    def tensor(self, other):
        tensor = dict()
        for i in self.complex.keys():
            for j in other.complex.keys():
                print(i,j)
                if not i+j in tensor.keys():
                    tensor[i+j] = self.complex[i].tensor(other.complex[j])
                else:
                    tensor[i+j] = tensor[i+j].direct_sum(self.complex[i].tensor(other.complex[j]))
        return LabelGMFComplex(tensor)
                #gotta handle the maps!
                #
                #
    def __repr__(self):
        ret = ""
        for i in self.complex.keys():
            ret += f"{i}: {self.complex[i]}\n"

        return ret

def Cp(x1_str, x2_str, x3_str, x4_str, n, sign):
    if not (sign == '+' or sign == '-'):
        raise Exception(f'Sign {sign} must be either "+" or "-"')
    #choose: mu=0, so lambda=1
    x1, x2, x3, x4 = var(x1_str, x2_str, x3_str, x4_str)
    L14 = Lij(x1_str, x4_str, n)
    L23 = Lij(x2_str, x3_str, n)
    gamma0 = L14.tensor(L23)
    gamma1 = Ct(x1_str, x2_str, x3_str, x4_str, n)
    u1 = u_1(x1_str, x2_str, x3_str, x4_str, n)
    u2 = u_2(x1_str, x2_str, x3_str, x4_str, n)
    #U0 = Matrix(R, [[x4-x2, 0],[-u2+(u1+x1*u2-pi(x2_str,x3_str,n))/(x1-x4), 1]])
    if sign == '+':
        return LabelGMFComplex({-1: gamma1.curly_shift([1-n,1-n],[1-n,1-n]), \
                                0: gamma0.curly_shift([-n, -n],[-n, -n])})
    if sign == '-':
        return LabelGMFComplex({0: gamma0.curly_shift([n,n],[n,n]), 1: gamma1.curly_shift([n-1,n-1], \
                                                                                          [n-1,n-1])})
    #U1 = Matrix(R, [[],[-1, 1]])
    #

def gen_tensor(alg1, alg2, gluing):
    '''
    Given two polynomial algebras k[x_a], k[x_b], return k[x_[a cup b]] subject to the relations
    imposed by base = [(x_0,x_1)] (i.e., identifying x_0 and x_1 in the algebras resp.)
    '''
    if alg1.base() != alg2.base():
        raise Exception('Algebras must be over the same base field.')
    gens1 = alg1.gens()
    gens2 = alg2.gens()
    for relation in gluing:
        if not relation[0] in gens1 or not relation[1] in gens2:
            raise Exception('Relations must identify elements in gens1 with elements in gen2.')




class MF():
    def __init__(self, d0, d1):
        self.d0 = d0
        self.d1 = d1
        if self.d0.base_ring() != self.d1.base_ring():
            raise Exception('d0, d1 must take entries in the same base ring')
        self.R = self.d0.base_ring()
        self.w = 0 #define this as the first entry of d^2.

    def cohomology(self):
        # define this as Ker(d_i)/Im(d_i+1), i mod 2.
        #
        pass

    def tensor(self, other):
        # need this
        pass





Qxy = QQ['x','y']
Qyz = QQ['y','z']
pixy = pi('x','y',4)
piyz = pi('y','z',4)
Md0 = Matrix(Qxy, [pixy])
Md1 = Matrix(Qxy, [x-y])
Nd0 = Matrix(Qyz, [piyz])
Nd1 = Matrix(Qyz, [y-z])
u1 = u_1('x','y','z','w',4)
u2 = u_2('x','y','z','w',4)
Ring = QQ['x','y','z','w']
uu1 = Matrix(Ring, [u1])
uu1b = Matrix(Ring, [x+y-z-w])
uu2 = Matrix(Ring, [u2])
uu2b = Matrix(Ring, [x*y-z*w])
Rt1 = LabelGMF(uu1, uu1b, [''], ['a'], [0], [-4])
Rt2 = LabelGMF(uu2, uu2b, [''], ['b'], [0], [-2])
Cmin = Cp('x','y','z','w',4,'-')
Cplu = Cp('x','y','z','w',4,'+')
C1 = Cp('x1','x2','x3','x4',2,'+')
C2 = Cp('x3','x4','x5','x6',2,'+')
C3 = Cp('x5','x6','x1','x2',2,'+')
print(now)
tref_khov_1 = C2.tensor(C3)
tref_khov = C1.tensor(tref_khov_1)
print(tref_khov)
print(now)
mf1 = MatFact(Md0,Md1)
mf2 = MatFact(Nd0,Nd1)
LMF1 = LabelMF(mf1.d0, mf1.d1, [''], ['a'])
LMF2 = LabelMF(mf2.d0, mf2.d1, [''], ['b'])
LGMF1 = LabelGMF(mf1.d0, mf1.d1, [''],['a'], [0],[-1])
LGMF2 = LabelGMF(mf2.d0, mf2.d1, [''],['b'], [0],[-1])


#print(mf1.direct_sum(mf1.angle_shift()))
