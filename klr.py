#!/usr/bin/env python3


class BoundaryWord:
    def __init__(self, word, n):
        alphabet = set(range(-(n-1), n))-{0}
        if type(word) != list:
            raise Exception(f'Boundary word {word} must be a list')
        for i in word:
            if i not in alphabet:
                raise Exception(f'Boundary label {i} of {word} out of bounds.') #this may cause more problems than its worth
        self.word = word
        self.n = n


    def __repr__(self):
        return f"{self.word}, n={self.n}"

    def merge(self, i):
        if self.word[i]/self.word[i+1] < 0:
            raise Exception(f'Incompatible merge at indices ({i}, {i+1}).')
        new_word = self.word[0:i]+[self.word[i]+self.word[i+1]]+self.word[i+2:]
        return BoundaryWord(new_word, self.n)

    def split(self, i, j):
        '''
        Takes an edge with label n' to a pair j, n'-j (spider)
        '''
        new_word = self.word[0:i]+[j]+[self.word[i]-j]+self.word[i+1:]
        return BoundaryWord(new_word, self.n)

    def tag(self, i):
        if self.word[i] > 0:
            new_word = self.word[0:i]+[-self.n+self.word[i]]+self.word[i+1:]
        else:
            new_word = self.word[0:i]+[self.n+self.word[i]]+self.word[i+1:]
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


    def check_compatibility(self, source):
        '''
        Given a labelling of the source object, returns the compatible labelling of the target.
        BROKEN. Ought to fix.
        '''
        if type(source) != BoundaryWord:
            raise Exception('Source is not of type BoundaryWord')
        if len(source.word) < self.strand_number:
            raise Exception(f'Word {source.word} is not long enough. Minimal strand #: {self.strand_number}')
        edge_layer = source #fix this notation. this is a BoundaryWord...
        for i in self.word: #and this is a spider/strand word.
            if i[0] == 'm':
                edge_layer = edge_layer.merge(i[1])
            if i[0] == 's':
                edge_layer = edge_layer.split(i[1],i[2])
            if i[0] == 't':
                edge_layer = edge_layer.tag(i[1])
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
        #check_comp = self.spider_word.check_compatibility(self.source) #UNCOMMENT WHEN YOU FIX THIS CHECK.
        #if self.target != check_comp:
        #    raise Exception(f'Target: {self.target} does not match expected: {check_comp}')

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

    def ladder_form(self):
        pass


    def __repr__(self):
        #TODO represent this free spider word GRAPHICALLY later
        return f"Source: {self.source}\nTarget: {self.target}\nSpider Word: {self.spider_word}\nn={self.n}"

    def ascii(self):
        currword = self.source
        stri = currword.ascii_constructor()

        for word in self.spider_word.word:
            currlen = len(currword.word)
            if word[0] == 'm':
                move_string = ""
                for i in range(currlen):
                    if i <= word[1]:
                        move_string = move_string + " |"
                    elif i > word[1]:
                        move_string = move_string + "\\ "
                currword = currword.merge(word[1])
            if word[0] == 's':
                move_string = ""
                for i in range(currlen + 1):
                    if i <= word[1]:
                        move_string = move_string + " |"
                    elif i > word[1]:
                        move_string = move_string + "/ "
                currword = currword.split(word[1], word[2])
            if word[0] == 't': #NOT specifying an embedding/tag side. implement later if needed
                move_string = ""
                for i in range(currlen):
                    if i != word[1]:
                        move_string = move_string + " |"
                    elif i == word[1]:
                        move_string = move_string + " +"
                currword = currword.tag(word[1])
            new_stri = currword.ascii_constructor() + '\n' + move_string + '\n'
            stri = new_stri + stri

        return stri

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
