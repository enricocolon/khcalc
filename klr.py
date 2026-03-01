#!/usr/bin/env python3


class BoundaryWord:
    def __init__(self, word, n):
        alphabet = set(range(-(n-1), n)) #{0} is not disallowed--represents a space in a boundary word. not necessary for
        #most computations but helpful for ladderizing
        if type(word) != list:
            raise Exception(f'Boundary word {word} must be a list')
        for i in word:
            if i not in alphabet:
                raise Exception(f'Boundary label {i} of {word} out of bounds.') #this may cause more problems than its worth
        self.word = word
        self.n = n
        zero_positions = set()
        for i in range(len(self.word)):
            if self.word[i] == 0:
                zero_positions.add(i)
        self.zero_positions = zero_positions #TODO keep track of zero positions for rendering

    def __repr__(self):
        return f"{self.word}, n={self.n}"

    def __len__(self):
        return len(self.word)

    def offset_index(self,i):
        '''
        returns # of 0s strictly less than index i.
        '''
        offset = 0
        for j in range(i):
            if self.word[j] == 0:
                offset += 1
        return offset

    def word_nozero(self):
        if 0 in self.word:
            word_nozero = [x for x in self.word if x != 0]
        else:
            word_nozero = self.word
        return word_nozero

    def offset(self,i):
        '''
        Returns offset for the padded index of the ith (0-indexed) nonzero element of the boundary word.
        '''
        offset = 0
        scan = i
        truncated_boundary = self.word[0:scan+1]
        nonzeros = len([j for j in truncated_boundary if j != 0])
        while nonzeros <= i:
            scan += 1
            offset += 1
            truncated_boundary = self.word[0:scan+1]
            nonzeros = len([j for j in truncated_boundary if j != 0])
        return offset

    def merge(self, i):
        offset = self.offset(i)
        if self.word[i+offset] == 0 or self.word[i+offset+1] == 0:
            raise Exception('Cannot merge empty vertex.')

        if self.word[i+offset]/self.word[i+offset+1] < 0:
            raise Exception(f'Incompatible merge at indices ({i+offset}, {i+1+offset}).')
        new_word = self.word[0:i+offset]+[self.word[i+offset]+self.word[i+1+offset]]+self.word[i+2+offset:]
        return BoundaryWord(new_word, self.n)


    def split(self, i, j):
        '''
        Takes an edge with label n' to a pair j, n'-j (spider)
        '''
        offset = self.offset(i)
        if self.word[i+offset] == 0:
            raise Exception('Cannot split empty vertex.')
        new_word = self.word[0:i+offset]+[j]+[self.word[i+offset]-j]+self.word[i+1+offset:]
        return BoundaryWord(new_word, self.n)

    def tag(self, i):
        offset = self.offset(i)
        if self.word[i+offset] == 0:
            raise Exception('Cannot tag empty vertex.')
        if self.word[i+offset] > 0:
            new_word = self.word[0:i+offset]+[-self.n+self.word[i+offset]]+self.word[i+1+offset:]
        else:
            new_word = self.word[0:i+offset]+[self.n+self.word[i+offset]]+self.word[i+1+offset:]
        return BoundaryWord(new_word, self.n)

    def pad(self, i):
        '''
        insert a zero after (0-adjusted) index i in a boundary word. (accordingly i can be -1 or nonneg.)
        '''
        if type(i) in {set,list}:
            output = self
            for index in i:
                output = output.pad(index)
            return output
        offset = self.offset(i)
        new_word = self.word[0:i+1+offset] + [0] + self.word[i+1+offset:]
        return BoundaryWord(new_word, self.n)


    def __eq__(self, other):
        if (self.word == other.word) and (self.n == other.n):
            return True
        else:
            return False

    def __add__(self,other):
        if self.n == other.n:
            return BoundaryWord(self.word + other.word, self.n)
        else:
            raise Exception('Boundary words cannot add, words in different alphabets')

    def ascii_constructor(self):
        '''
        Given a boundary word, returns its ascii representation for rendering
        '''
        stri = ""
        for i in self.word:
            if i >= 0:
                stri = stri + " " + str(i)
            if i<0:
                stri = stri + str(i)
        return stri


class SpiderWord:
    def __init__(self, array):
        '''
        Input: an array which is a list of merges/splits/tags together with an index.
        Example: [('m', 1), ('s', 0, j), ('t', 2)]
        Output: an object that represents the composition (apply in order as the index increases) of the
        merges/splits/tags. Comes with a method to check that the output of the spider word makes sense.

        ('m', i): merge i'th and (i+1)st objects.
        ('s',i,j): split i'th vertex into j, n'-j, where the edge carried a label n'.
        ('t', i): swap orientation and identify k = -(n-k). (c.f. generators in 2.1 Cautis-Kamnitzer-Morrison)

        Attributes:
        self.word = the array itself with instructions as exemplified above.
        self.strand_number = the minimum number of strands for the word to make sense.
        '''
        maxi = 0
        for i in array:
            if i[0] not in {'m','s','t'}:
                raise Exception('invalid spider word')
            if i[0] == 's':
                if len(i) < 3:
                    raise Exception('split map requires two arguments (index, split amount)')
            if i[1] > maxi:
                if i[0] == 's':
                    maxi = i[1]+1
                else:
                    maxi = i[1]
        self.strand_number = maxi #minimum # of strands necessary
        self.word = array

    def __repr__(self):
        return str(self.word)


    def __len__(self):
        return len(self.word)

    def check_compatibility(self, source, allLayers=False):
        '''
        Given a labelling of the source object, returns the compatible labelling of the target.
        BROKEN. Ought to fix.
        '''
        if type(source) != BoundaryWord:
            raise Exception('Source is not of type BoundaryWord')
        if len(source.word) < self.strand_number:
            raise Exception(f'Word {source.word} is not long enough. Minimal strand #: {self.strand_number}')
        edge_layer = source #fix this notation. this is a BoundaryWord...
        if allLayers:
            layers = [source]
        for i in self.word: #and this is a spider/strand word.
            if i[0] == 'm':
                edge_layer = edge_layer.merge(i[1])
            if i[0] == 's':
                edge_layer = edge_layer.split(i[1],i[2])
            if i[0] == 't':
                edge_layer = edge_layer.tag(i[1])
            if allLayers:
                layers.append(edge_layer)
        if allLayers:
            return layers
        return edge_layer




class Web:
    def __init__(self,source,target,spider_word):
        '''
        Initialize a MOY web object.
        '''
        if {type(source),type(target)} != {BoundaryWord}:
            raise Exception(f'A boundary component is of incorrect type.')
        if source.n != target.n:
            raise Exception(f'Boundary objects are weights for different sl_n.')
        self.source = source
        self.target = target
        self.n = self.source.n
        #gotta do some sort of check for the spider word
        self.spider_word = spider_word
        check_comp = self.spider_word.check_compatibility(self.source)
        #print(self.target.word_nozero(),check_comp.word_nozero())
        if self.target.word_nozero() != check_comp.word_nozero():
            print('excepttriggered')
            print(self.target.word_nozero(),check_comp.word_nozero())
            #print(type(self.target.word_nozero()))
            #print(type(check_comp.word_nozero()))
            #print(self.target.word_nozero()==check_comp.word_nozero(),self.target.word_nozero()!=check_comp.word_nozero())
            raise Exception(f'Target: {self.target} does not match expected: {check_comp}')
        self.layers = self.spider_word.check_compatibility(self.source, allLayers=True)
        #self.layers.reverse()
        self.layer_width = max([len(i) for i in self.layers])

    def tensor(self,other):
        #i think there's something wrong here with shifted_other.
        if self.n != other.n:
            raise Exception('Webs for different sl_n.')
        source = self.source + other.source
        target = self.target + other.target
        #indices 0,...,len(1)-1 for 1st, indices len(1),...,len(1)+len(2) for 2nd.
        spider_word_1 = self.spider_word.word
        spider_word_2 = other.spider_word.word
        shifted_other = []
        for i in spider_word_2: #here is where i need to keep track of the "offset parameter"
            if i[0] == 's':
                shifted_other.append((i[0],i[1]+len(self.target.word),i[2]))
            else:
                shifted_other.append((i[0],i[1]+len(self.target.word)))
        spider_word = SpiderWord(spider_word_1 + shifted_other)
        #do something here to keep track of the merges and stuff
        output_web = Web(source, target, spider_word)
        return output_web

    def compose(self,other):
        """
        self a web from w_1 -> w_2,
        other a web from w_2 -> w_3.
        return the composite w_1 -> w_2 -> w_3.
        """
        if self.n != other.n:
            raise Exception('Webs for different sl_n.')
        if self.target != other.source:
            raise Exception(f"Source {self.source} does not match target {other.source}.")
        spider_word_1 = self.spider_word.word
        spider_word_2 = other.spider_word.word
        spider_word = SpiderWord(spider_word_1 + spider_word_2)
        output_web = Web(self.source, other.target, spider_word)
        return output_web

    def pad_layers(self):
        '''
        pads web boundary layers to produce a ladder word representing the web.
        '''

        pad_list = [layer for layer in self.layers]
        width = self.layer_width
        #pad_list = [self.source]
        spider_word = [i for i in self.spider_word.word]

        pad_indices = []

        for i in range(len(spider_word)):
            index = spider_word[i][1]
            #print(spider_word[i][0])
            if spider_word[i][0] == 'm':
                if True:
                #if pad_list[i+1].word[index] != 0:
                    j = i+1
                    if j == width - 1:
                        continue
                    #print(pad_list[j],j,width)
                    while len(pad_list[j].word) < width:
                        #print(len(pad_list),j)
                        pad_list[j] = pad_list[j].pad(index)
                        if j == width - 3:
                            break
                        j += 1
            if spider_word[i][0] == 's':
                if True:
                #if pad_list[i].word[index] != 0:
                    j = i
                    #if j == 0:
                    #    continue
                    while len(pad_list[j].word) < width:
                        pad_list[j] = pad_list[j].pad(index)
                        if j == 0:
                            break
                        j -= 1
            else:
                pass
        return pad_list

        # spider_word.reverse()
        # curr_spiderword_tup = spider_word.pop()
        # curr_spiderword = SpiderWord([curr_spiderword_tup])



        #prev_spiderword = None
        # while len(spider_word)>0:
        #     #print(curr_spiderword)
        #     curr_spiderword_tup = spider_word.pop()
        #     #prev_spiderword = curr_spiderword
        #     curr_spiderword = SpiderWord([curr_spiderword_tup])
        #     if curr_spiderword.word[0][0] == 's':
        #         print('s', pad_list[-1], curr_spiderword.word[0][1])
        #         pad_list[-1] = pad_list[-1].pad(curr_spiderword.word[0][1])
        #         nxt = curr_spiderword.check_compatibility(pad_list[-1])
        #         pad_list.append(nxt)
        #     if curr_spiderword.word[0][0] == 'm':
        #         print('m', pad_list[-1], curr_spiderword.word[0][1])
        #         #print(curr_spiderword.word[0][1])
        #         nxt = curr_spiderword.check_compatibility(pad_list[-1])
        #         nxt.pad(curr_spiderword.word[0][1])
        #         pad_list.append(nxt)
        #     else:
        #         print(f'{curr_spiderword.word[0][0]}', 'els', pad_list[-1], curr_spiderword.word[0][1])
        #         #print(spider_word, curr_spiderword.word)
        #         nxt = curr_spiderword.check_compatibility(pad_list[-1])
        #         pad_list.append(nxt)
        #return pad_list

    def ladder_form(self):
        pad_list = [i.word for i in self.pad_layers()]
        diffs = []
        for i in range(len(pad_list)-1):
            diffs.append([pad_list[i][j]-pad_list[i+1][j] for j in range(self.layer_width)])
        return diffs

    def ladder_form_deprecated(self):
        ladder = Web(self.source,self.source,SpiderWord([]))
        for word in self.spider_word.word:
           wordd = SpiderWord([word])
           if word[0] == 'm':
               range_ = wordd.check_compatibility(ladder.target)
               ladder = ladder.compose(Web(ladder.target, range_,wordd))
           if word[0] == 's':
               domain_adjust = ladder.target.pad(word[1])
               print(type(domain_adjust))
               print(ladder.source,domain_adjust,ladder.spider_word)
               ladder = ladder.compose(Web(ladder.source,domain_adjust))#,ladder.spider_word)
               range_ = wordd.check_compatibility(domain_adjust)
               ladder = ladder.compose(Web(domain_adjust,range_,wordd))
           if word[0] == 't':
               range_ = wordd.check_compatibility(ladder.target)
               ladder = ladder.compose(Web(ladder.target, range_,wordd))
        return ladder

    def pad(self, layer, i):
        self.layers[layer] = self.layers[layer].pad(i) #this is broken because the pad was broken.

    def __repr__(self):
        #TODO represent this free spider word GRAPHICALLY later
        return f"Source: {self.source}\nTarget: {self.target}\nSpider Word: {self.spider_word}\nn={self.n}"


    def ascii(self):
        layers = [i for i in self.layers]
        spider_word = [i for i in self.spider_word.word]
        layers.reverse()
        spider_word.reverse()
        currlayer = layers.pop()
        stri = currlayer.ascii_constructor()
        while layers:
            currlayer = layers.pop()
            currword = spider_word.pop()
            if len(spider_word) == 0:
                stri = currlayer.ascii_constructor() + "\n" + stri
                return stri
            #print(currlayer, currword)
            stri = currlayer.ascii_constructor() + "\n" + stri
            for label in currlayer.word:
                #print(label)
                #gotta get index for merge/split
                pass
        return stri

    
    def ascii_deprecated(self):
        #TODO implement handling for 0-weights
        currword = self.source
        stri = currword.ascii_constructor()

        for word in self.spider_word.word:
            currlen = len(currword.word)
            if word[0] == 'm':
                move_string = ""
                for i in range(currlen):
                    if i in currword.zero_positions:
                        move_string = move_string + "  "
                    elif i <= word[1]:
                        move_string = move_string + " |"
                    elif i == word[1]+1:
                        move_string = move_string + "\\ "
                    elif i > word[1]+1:
                        move_string = move_string + "\\ "
                currword = currword.merge(word[1])
            if word[0] == 's':
                move_string = ""
                for i in range(currlen + 1):
                    if i in currword.zero_positions:
                        move_string = move+string + "  "
                    elif i <= word[1]:
                        move_string = move_string + " |"
                    elif i == word[1]+1:
                        move_string = move_string + "/ "
                    elif i > word[1]+1:
                        move_string = move_string + "/ "
                currword = currword.split(word[1], word[2])
            if word[0] == 't': #NOT specifying an embedding/tag side. implement later if needed
                move_string = ""
                for i in range(currlen):
                    if i in currword.zeropositions:
                        move_string = move_string + "  "
                    elif i != word[1]:
                        move_string = move_string + " |"
                    elif i == word[1]:
                        move_string = move_string + " +"
                currword = currword.tag(word[1])
            new_stri = currword.ascii_constructor() + '\n' + move_string + '\n'
            stri = new_stri + stri

        return stri

    def render_deprecated(self):
        print(self.ascii_deprecated())

    def render(self):
        print(self.ascii())

class Merge(Web):
    def __init__(self,a,b,n):
        '''
        Initialize a merge-map web.
        '''
        web = None
        source = BoundaryWord([a,b],n)
        target = BoundaryWord([a+b],n)
        super().__init__([a,b],[a+b],web)
        pass


class gWeb(Web):
    def __init__(self,source,target,spider_word,grading):
        super().__init__(source,target,spider_word)
        self.q = grading

bword1= BoundaryWord([2,2],5)
bword2 = BoundaryWord([4],5)
bword3=BoundaryWord([2],5)
bword4=BoundaryWord([1,1],5)
web = Web(bword3,bword4,SpiderWord([('s',0,1)]))
web2 = Web(bword1,bword2,SpiderWord([('m', 0)]))
test_tensor = web.tensor(web2)
bword5=BoundaryWord([3,1],5)
web3= Web(bword2, bword5, SpiderWord([('s',0,3)]))
test_compose = web2.compose(web3) #compose takes A.compose(B) to B(A). (reading composition right to left)
J = test_tensor.tensor(test_compose)
JJ = J.tensor(J)
JJ1 = JJ.tensor(web)
JJ2 = JJ.tensor(web2)
#M = JJ1.ladder_form()
#MM = JJ2.ladder_form()


class Weight():
    def __init__(self, weight):
        '''
        instantiates an object representing an sl_n
        weight in the root basis.
        '''
        self.n = len(weight)+1
        self.weight = weight

    def __eq__(self, other):
        if not isinstance(other, Weight):
            return NotImplemented
        return self.weight == other.weight

    def __repr__(self):
        return f'{self.weight}, sl_{self.n}'

    def __add__(self,other):
        weight = []
        for i in range(self.n-1):
            weight.append(self.weight[i]+other.weight[i])
        sum = Weight(weight)
        return sum

    def __sub__(self,other):
        weight = []
        for i in range(self.n-1):
            weight.append(self.weight[i]-other.weight[i])
        sum = Weight(weight)
        return sum

    def shift(self, i, sign):
        '''
        Represents adding or subtracting the i'th simple root
        from self(in this basis, a one-hot vector)
        '''
        vec = [0]*(self.n-1)
        vec[i] = 1
        if sign == 1:
            return self + Weight(vec)
        if sign == -1:
            return self - Weight(vec)


    def pairing(self,i):
        '''
        input: given a weight presented (in the root lattice, as here)
        by lambda, and an index i,
        output: return the pairing <h_i,lambda>.
        '''
        return sum(self.weight[j]*a(i,j) for j in range(self.n-1))


def a(i,j):
    '''
    Method for getting entry a_ij of the infinite sl Cartan matrix, i.e.,
    an n-agnostic Cartan form for weights.
    '''
    if abs(i-j) > 1:
        return 0
    if abs(i-j) == 1:
        return -1
    if abs(i-j) == 0:
        return 2



class EF_word():
    def __init__(self,source, word):
        '''
        input:
            source: an sl_n weight,
            word: an array of pairs ('E'/'F', index).
        output:
            the corresponding quantum group word in {E,F}.
        '''
        if not isinstance(source, Weight):
            return NotImplemented
        self.source = source #weight
        self.word = word #free word in E_i,F_i and divided powers - need support for E^(a),F^(a) (over Q)
        offset = Weight([0]*(source.n-1))
        for letter in self.word:
            v = [0]*(source.n-1)
            if letter[0] == 'E':
                offset = offset.shift(i, 1)
            if letter[0] == 'F':
                offset = offset.shift(i, -1)
        self.target = self.source + offset


    def __eq__(self, other):
        # commutators, quantum serre relat. <- commutators to get F/Es at top, E/F at bottom. <- just need commutators! [E_i,F_i].
        # [NNNNNN000]->[NNNNN0000] should be reducible to k*1.
        # any word evaluates as a scalar if working in highest weight (getting rid of E's). <- passing to the quotient
        pass

    def __repr__(self):
        wordstring = ""
        for i in self.word:
            wordstring  =  i[0] + f'_{i[1]} ' + wordstring
        return f'1_{self.target.weight} ' + wordstring + f'1_{self.source.weight}'

class divided_power():
    pass




class ChainComplex():
    def __init__(self, layers, differentials):
        """
        input:
        layers = dictionary (keys: integers, values: vector spaces)
        differentials = dictionary (keys: integers, values: differentials from vs[i] to vs[i+1])

        output:
        chain complex object specified by the input data
        """
        self.layers = layers
        self.differentials = differentials



#E2 = EF_word([1,1],[('E',0),('E',0)])
