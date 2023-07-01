'''
PAD_token = 0
SOS_token = 1
EOS_token = 2

NO UNK TOKEN!
'''

class Vocab:
    def __init__(self):        
        self.word2index = {"PAD":0, "SOS":1, "EOS":2}        
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.word2count = {}
        self.n_words = len(self.word2index) # count PAD, SOS and EOS tokens
        
    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word) # using lists of tokens so no need to split
            
    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1     
        else:
            self.word2count[word] += 1


