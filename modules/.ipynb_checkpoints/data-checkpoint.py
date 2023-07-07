import pandas as pd
import numpy as np
import torch

from collections import namedtuple, Counter

# import and download tools necessary for text data preparation and cleanup
import nltk

from string import punctuation

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# create a list of stopwords making sure that we don't remove words used in questions, like how, why, when...

stop = list(set(stopwords.words('english')))
question_words = ['who', 'how', 'when', 'why', 'where', 'what', 'which', 'whom']
stop = [word for word in stop if word not in question_words]

cols = ['Question', 'Answer'] # columns for the dataframe


def get_dataframe(dataset):    
    '''Returns train and test dataframes with questions and answers form the dataset (SQuAD1)'''    
    data_dict = {col:[] for col in cols}
    for _, question, answer, _ in dataset:
        data_dict['Question'].append(question)
        data_dict['Answer'].append(answer[0])
    
    df = pd.DataFrame(data_dict)
    
    return df



def sample_df_perc(df, percentage: float=0.5):
    '''Return only a sampled fraction of the original dataframe'''
    return df.sample(int(df.shape[0]*percentage)).reset_index(drop=True)



def tokenize_sentence(sentence, normalization = None):
    
    '''Text preparation: 
    - make everything lower case
    - remove punctuation
    - remove stopwords to reduce the number of different tokens    
    - if neeeded normalize by either stemming or lemmatization
    Returns a list of tokens.
    '''
    
    # make everything lower case
    sentence = sentence.lower()
    
    # remove punctuation
    sentence = ''.join([ch for ch in sentence if ch not in punctuation])
    
    # split into tokens
    tokenizer = RegexpTokenizer(r'\w+')
    
    tokens = tokenizer.tokenize(sentence)
    
    # remove the stopwords
    
    tokens = [token for token in tokens if token not in stop]
    
    # normalize the text
    
    # stemming to get only the base form of the word
    if normalization == 'stem':
        
        stemmer = SnowballStemmer('english')    
        tokens = [stemmer.stem(token) for token in tokens]
        
    # lemmatizing for slower but better normalization where words are put together based on the context
    elif normalization == 'lemma':
        
        lemmatizer = WordNetLemmatizer()
        
        tokens = [lemmatizer.lemmatize(token, get_pos(token, tag_dict)) for token in tokens]
        
    else:
        pass
        
    return tokens


Pair = namedtuple('Pair', ['question', 'answer'])
def get_pairs_from_df(df, cols):
    '''returns a list of named tuples (question, answer)'''
    dicts = []
    for col in cols:
        dicts.append(df[col].to_dict().values())
    
    return [Pair(q, a) for q, a in zip(*dicts)]



def filter_sentences(df, cols, thresholds, condition='shorter'):
    '''keep only rows with sentences shorter or longer than the threshold (different thresholds for each columns'''    
    for col,threshold in zip(cols, thresholds):
        
        if condition == 'longer':        
            df = df[df[col].apply(lambda x: len(x)) > threshold] # keep dataframe rows with sentences longer than the threshold
        else:
            df = df[df[col].apply(lambda x: len(x)) < threshold] # keep dataframe rows with shorter than the threshold        
    
    return df


def to_tensor(vocab, tokens, seq_len, device):
    '''Converts a tokenized sentence into a tensor of indices of a given length.
    If too short, it uses padding at the beginning of the sentence as suggested by the mentor.'''
    
    tokens = [t for t in tokens if t in vocab.word2count.keys()]
    
    padded = [vocab.word2index['PAD']] * (seq_len-len(tokens)) + [vocab.word2index['SOS']] + [vocab.word2index[t] for t in tokens] + [vocab.word2index['EOS']]
    
    tensor = torch.Tensor(padded).long().to(device).view(-1,1)
    
    return tensor




def get_thresholds(data, cutoff = 95):    

    '''Returns cutoff values to remove long sentence outliers'''

    cutoffs = {col:0 for col in data.keys()}

    for col in data.keys():
        counts, bins = data[col]

        for bin, count in zip(bins, counts.cumsum()/sum(counts)*100):        
            cutoffs[col] = bin            
            if count > cutoff:                                            
                break
        
    return cutoffs

def get_outliers(vocab, threshold):
    '''get the least common words (occurring less than the threshold) from a vocabulary'''
    c = Counter(vocab.word2count)        
    return [word[0] for word in c.most_common() if word[1] < threshold]

def remove_least_common(df, cols, vocabs):
    for col, vocab in zip(cols, vocabs):
        df[col] = df[col].apply(lambda s: [w for w in s if w not in vocab])        
    return df