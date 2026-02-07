# class SignedCrossing():
#     def __init__(self, crossing, sign):
#         #orientation of crossing entries is CCW, from lower incoming strand. The sign is specified.
#         if type(crossing) != list:
#             raise Exception(f'Crossing {xing} has invalid type {type(xing)}.')
#         if len(crossing) != 4:
#             raise Exception(f'Crossing has valence {len(crossing)} != 4')
#         self.crossing = crossing
#         if sign not in [1,-1]:
#             raise Exception(f'Crossing has invalid sign {sign}')
#         self.sign = sign

#     def __len__(self):
#         return len(self.crossing)


# class Crossing():
#     def __init__(self, sign=1, glue=None):
#         self.sign = sign
#         self.glue = glue
#         self.endpoints = (('a','b'),('c','d')) #going ccw. first pair are targets in the tangle ("+") second are sources.

# class Tang():
#     def __init__(self, crossings=None, endpoints=None):
#         self.crossings = []
#         self.endpoints = set()
#         if crossings:
#             self.crossings = crossings
#         if endpoints:
#             self.endpoints = endpoints

#     def attach(self, other, endpoint_self, endpoint_other):


# class Tangle(dict):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.crossings = self.keys()
#         for i in self.crossings:
#             if type(i) != SignedCrossing:
#                 raise Exception(f'Invalid crossing {i} of type {type(i)} != SignedCrossing')
#             if type(self[i]) != tuple:
#                 raise Exception(f'Neighbors of crossing {i} of type {type(self[i])} != tuple')
#             if len(self[i]) != 4:
#                 raise Exception(f'Crossing {i} has {len(self[i])} != 4 neighbors')
#         #DONT CHECK FOR PLANARITY. later, you'll probably use this to embed in other 3-mfds idk

# class CrossingArray():
#     def __init__(self, crossing_array):
#         self.arcs = set()
#         for xing in crossing_array:
#             if type(xing) != SignedCrossing:
#                 raise Exception(f'Crossing {xing} has invalid type {type(xing)}.')
#             if len(xing) != 4:
#                 raise Exception(f'Crossing {xing} has {len(xing)} entries.')
#             self.arcs = self.arcs.union(set(xing))
#         self.crossings = crossing_array





# def initialize_complex(crossing_array, ground_ring):
#     xings = crossing_array
#     # popping from the end cuz it's faster, doesn't really matter tbh just note to self
#     curr = xings.pop()
#

class Node():
    def __init__(self, data):
        self.data = data
        self.next = None

class CircularLinkedList:
    def __init__(self):
        self.head = None
        self.size = 0

    def append(self, data):
        self.size += 1
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            new_node.next = new_node
            return
        current = self.head
        while current.next != self.head:
            current = current.next
        current.next = new_node
        new_node.next = self.head

    def __repr__(self):
        if not self.head:
            return '[]/~'
        strg = '['
        current = self.head
        while True:
            strg += str(current.data)
            current = current.next
            if current == self.head:
                break
            strg += ','
        strg += ']/~'
        return strg

    def __len__(self):
        return self.size

    def splice_at_pos(self, other, pos): #O(pos). also BROKEN. splices after position.
        if not self.head:
            return other
        if not other.head:
            return self

        current = self.head
        for _ in range(pos):
            current = current.next

        glueself = current.next

        current.next = other.head

        glueother = other.head
        while glueother.next != other.head:
            glueother = glueother.next

        glueother.next = glueself

        return self

    def identify_at_pos(self, pos1, pos2, side):
        pass


def CLL(array):
    CLL = CircularLinkedList()
    for i in array:
        CLL.append(i)
    return CLL

class Crossing():
    def __init__(self, sign, ui, uo, li, lo):
        self.sign = sign
        self.ui = ui #Upper strand, Incoming
        self.uo = uo
        self.li = li
        self.lo = lo
        self.glue = None #no endpoint gluings
        if self.sign == 1:
            #bd = CLL([self.li, self.uo, self.lo, self.ui])
            bd = CLL([self.li, self.uo, self.lo, self.ui])
            self.boundary = [bd]
        if self.sign == -1:
            #bd = CLL([self.li, self.ui, self.lo, self.uo])
            bd = CLL([self.li, self.ui, self.lo, self.uo])
            self.boundary = [bd]



class Tangle():
    #so far, no unlinked components. (i guess if i care that much do something with unknot "infty sign")
    def __init__(self):
        self.crossings = []
        self.boundary = []

    def attach_crossing(self, crossing, glue):
        if not self.crossings:
            self.crossings = crossing
            self.boundary = crossing.boundary
            return self

        self.crossings += crossing
