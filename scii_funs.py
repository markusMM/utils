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
all_letters = string.ascii_letters + " .,;'"
#%% -- functions --

# unfolder for list series
def unfold_lists(lists):
    new_list = []
    new_iids = []
    for l,li in enumerate(list):
        this_Nlst = len(li)
        this_iids = (np.ones([this_Nlst],dtype=np.int64)*l).tolist()
        
        
    
        
# split words in sequence
def split_query_seq(words,delim=' '):
    new_wrds = []
    new_iids = []
    for j,wrd in enumerate(words):
        this_wrds = wrd.split(delim)
        this_nowd = len(this_wrds)
        this_iids = (np.ones([this_nowd],dtype=np.int64)*j).tolist()
        for j in range(this_nowd):
            new_wrds.append(this_wrds[j])
            new_iids.append(this_iids[j])
    return new_wrds, new_iids

# split a word into a list
def split_query(word,delim=' '):
    return word.split(delim)

# convert each element in sequence from unicode to ascii
def u2ASCII_seq(words):
    for j,wrd in enumerate(words):
        words[j] = unicodeToAscii(wrd)
    return words

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, len(all_letters))
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.ones(len(line), 1, len(all_letters))*(-1.0)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

# turn a sequence of words into a sequence of tensors
def lineSeqToTensor(lines):
    mx_length = 0
    for l in lines:
        if mx_length < len(l):
            mx_length = len(l)
    my_tensor_seq = torch.zeros( len(lines), mx_length, 1, len(all_letters))
    for l,line in enumerate(lines):
        my_tensor_seq[l] = lineToTensor(line)
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
        