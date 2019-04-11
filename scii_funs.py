# -*- coding: utf-8 -*-
"""
scii_funs()

    ASCII - Letter Handling

@author: Markus.Meister
"""
import unittest
import torch
import string
import unicodedata
import numpy as np
from nltk.stem.snowball import SnowballStemmer

stemmEN = SnowballStemmer('english')
stemmDE = SnowballStemmer('german')

stemmers = {
        'en' : stemmEN,
        'de' : stemmDE,
        }

all_letters = string.ascii_letters + " .,;'"
all_numbers = ''.join(list(map(lambda x: str(x),range(10))))
BOUND_LOW_CHARS = 26

#%% -- functions --

# unfolder for list series
def unfold_lists(lists):
    new_list = []
    new_iids = []
    for l,li in enumerate(lists):
        for ent in li:
            new_list.append(ent)
            new_iids.append(l)
    
    return new_list, new_iids

# split words in sequence
def split_query_seq(words, delim=' ', stop_words=[]):
    new_wrds = []
    new_iids = []
    for j,wrd in enumerate(words):
        this_wrds = list(filter(lambda j: j not in stop_words, wrd.split(delim)))
        this_nowd = len(this_wrds)
        this_iids = (np.ones([this_nowd],dtype=np.int64)*j).tolist()
        for j in range(this_nowd):
            new_wrds.append(this_wrds[j])
            new_iids.append(this_iids[j])
    return new_wrds, new_iids

# split a word into a list
def split_query(word,delim=' '):
    return word.split(delim)

# tokenize words in sequence
def tok_query_seq(seq, tokenizer, stop_words=[], lemm_flg=False):
    new_wrds = []
    new_iids = []
    for j,wrds in enumerate(seq):
        this_wrds = list(filter(lambda j: j not in stop_words, tokenizer.tokenize(wrds)))
        if lemm_flg:
            this_wrds = stem_query(this_wrds)
        this_nowd = len(this_wrds)
        this_iids = (np.ones([this_nowd],dtype=np.int64)*j).tolist()
        for j in range(this_nowd):
            new_wrds.append(this_wrds[j])
            new_iids.append(this_iids[j])
    return new_wrds, new_iids

# stem query
def stem_query(query, nlp_lan='de'):
    return stemmers[nlp_lan].stem(query)

# convert each element in sequence from unicode to ascii
def u2ASCII_seq(words):
    for j,wrd in enumerate(words):
        words[j] = unicodeToAscii(wrd)
    return words

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s, big_flg=False, num_flg=True):
    limit = (1+big_flg)*BOUND_LOW_CHARS
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters[:limit] + all_numbers[num_flg*10:]
    )

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(char):
    tensor = torch.zeros(1, len(all_letters))
    tensor[0][letterToIndex(char)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, len(all_letters))*(0.0)
    for c, char in enumerate(line):
        tensor[c][0][letterToIndex(char)] = 1
    return tensor

# turn a sequence of words into a sequence of tensors
def lineSeqToTensor(lines):
    mx_length = 0
    for l in lines:
        if mx_length < len(l):
            mx_length = len(l)
    my_tensor_seq = torch.zeros( len(lines), mx_length, 1, len(all_letters))
    for l,line in enumerate(lines):
        this_tensor = lineToTensor(line)
        this_length = this_tensor.size()[0]
        my_tensor = torch.zeros( mx_length, 1, len(all_letters) )
        my_tensor[:this_length] = this_tensor
        my_tensor_seq[l] = my_tensor
    return my_tensor_seq

# drop all empty words from a sequence
def drop_empty_queries(words,iids=[],return_iids=True):
    my_wrds = []
    my_iids = []
    nw_iids = []
    for w,wrd in enumerate(words):
        if wrd != '':
            my_wrds.append(wrd)
            nw_iids.append(w)
            if iids != []:
                my_iids.append(iids[w])
    if iids == []:
        if return_iids:
            return my_wrds, nw_iids
        else:
            return my_wrds
    else:
        return my_wrds, my_iids

def rm_stopwords(words, stop_words):
    '''
    Remove stop words from the list of words
    '''
    
    filtered = filter(lambda word: word not in stop_words, words)
    
    return list(filtered)

#%% -- unit tests --
class TestSiiFuns(unittest.TestCase):
    
    def test_drop_empty_queries(self):
        
        this_text = '\nEin dummer Mann\nspringt in Wald umher.'.split('\n')
        this_empties = (np.array(this_text) == '').sum()
        neow_text = drop_empty_queries(this_text)
        neow_empties = (np.array(neow_text) == '').sum()
        
        self.assertNotEqual(
                this_empties, 
                neow_empties, 
                )
        
        self.assertEqual(
                neow_empties, 
                0, 
                )
    
    def test_unicodeToAscii(self):
        
        string_a = 'Görkhan'
        string_b = unicodeToAscii(string_a)
        
        self.assertEqual(
                np.array(string_b == 'ö').sum(),
                0, 
                )
        
        self.assertNotEqual(
                string_a,
                string_b
                )
        