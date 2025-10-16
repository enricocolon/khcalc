class SignedCrossing():
    def __init__(self, crossing, sign):
        if type(crossing) != list:
            raise Exception(f'Crossing {xing} has invalid type {type(xing)}.')
        if len(crossing) != 4:
            raise Exception(f'Crossing has valence {len(crossing)} != 4')
        self.crossing = crossing
        if sign not in [1,-1]:
            raise Exception(f'Crossing has invalid sign {sign}')
        self.sign = sign

    def __len__(self):
        return len(self.crossing)

class CrossingArray():
    def __init__(self, crossing_array):
        self.arcs = set()
        for xing in crossing_array:
            if type(xing) != SignedCrossing:
                raise Exception(f'Crossing {xing} has invalid type {type(xing)}.')
            if len(xing) != 4:
                raise Exception(f'Crossing {xing} has {len(xing)} entries.')
            self.arcs = self.arcs.union(set(xing))
        self.crossings = crossing_array


def initialize_complex(crossing_array, ground_ring):
    pass
