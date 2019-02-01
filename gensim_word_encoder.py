# -*- coding: utf-8 -*-
"""
        Simple NLP Encoding

Created on Thu Jan 31 13:51:08 2019
@author: Markus.Meister
"""
import glob
import sys
import os
import torch
import pandas as pd
import numpy as np
from nltk import word_tokenize as tkn
import gensim 
from gensim import corpora, models, similarities

try:
    cd_here = glob.glob('')
except Exception as e:
    print(e)
    cd_here = []
if 'utils' in cd_here:
    sys.path.append('utils')

import scii_funs

class key_word_encoder:
    quote_size=64
    min_word_quote=1
    
    def __init__(self, dictionary=None):
        if type(dictionary).__name__.rsplit('.')[0] == 'Series':
                dictionary = dictionary.values
            
        self.diction = [tkn(sent) for sent in dictionary]
        
        self.word2vec = gensim.models.Word2Vec
    
    def encode(self, dictionary=None, quote_size=None, min_word_quote=None):
        if type(self.diction).__name__ == "NoneType":
            if type(dictionary).__name__ == "NoneType":
                 print("Sorry: No dictionary declared!")
            
            if type(dictionary).__name__.rsplit('.')[0] == 'Series':
                dictionary = dictionary.values
            
            self.diction = [tkn(sent) for sent in dictionary]
        
        if quote_size:
            self.quote_size=quote_size
        if min_word_quote:
            self.min_word_quote = min_word_quote
        
        self.wordinfo = self.word2vec(self.diction, min_count=self.min_word_quote, size = self.quote_size )
        
        return self.wordinfo
    
    def tensorize(self, dictionary=None, quote_size=None, min_word_quote=None, n_most_similar=3):
        if type(self.diction).__name__ == "NoneType":
            if type(dictionary).__name__ == "NoneType":
                 print("Sorry: No dictionary declared!")
            
            if type(dictionary).__name__.rsplit('.')[0] == 'Series':
                dictionary = dictionary.values
            
            self.diction = [tkn(scii_funs.unicodeToAscii(sent)) for sent in dictionary]
        
        if quote_size:
            self.quote_size=quote_size
        if min_word_quote:
            self.min_word_quote = min_word_quote
        
        if not hasattr(self,'worinfo'):
            self.encode()
        
        my_dictionary, my_iids = scii_funs.unfold_lists(self.diction)
        my_dictionary = np.array(my_dictionary)
        my_iids = np.array(my_iids)
        
        # malloc
        topNten = torch.zeros([my_dictionary.shape[0], n_most_similar, self.quote_size])
        wordTensors = torch.zeros([my_dictionary.shape[0], self.quote_size])
        
        #----------------------------- words ----------------------------------------
        
        for j,d in enumerate(my_dictionary):
            
            topN = self.wordinfo.most_similar(d,topn=n_most_similar)[:n_most_similar]
            topNten[j] = torch.tensor([self.wordinfo[sim] for sim in topN])
            
            wordTensors[j] = torch.tensor()
        
        return {
                'tensor':wordTensors,
                'tpoN':topNten,
                'iids':my_iids,
                'dictionary':my_dictionary,
                }
        
            
            
            