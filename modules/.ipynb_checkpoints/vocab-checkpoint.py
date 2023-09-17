'''
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
'''

class Vocab:
    def __init__(self):        
        self.word2index = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}        
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        # make sure that the special tokens don't get removed as too rare!
        self.word2count = {"<PAD>":9999999, "<SOS>":9999999, "<EOS>":9999999, "<UNK>":9999999, }
        self.n_words = len(self.word2index) # count PAD, SOS, EOS and UNK tokens
        
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


    def remove_word(self, word):
        idx = self.word2index[word]        
        self.word2index.pop(word)
        self.index2word.pop(idx)
        self.word2count.pop(word)
        self.n_words -= 1
        

    