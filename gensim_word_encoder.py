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
import nltk
#from nltk import word_tokenize as tkn
import gensim 
from gensim import corpora, models, similarities

sentence_detector = nltk.data.load('tokenizers/punkt/german.pickle')
tkn = sentence_detector.tokenize


try:
    cd_here = glob.glob('')
except Exception as e:
    print(e)
    cd_here = []
if 'utils' in cd_here:
    sys.path.append('utils')

import scii_funs

class key_word_encoder:
    quote_size=300
    min_word_quote=1
    
    def __init__(self, dictionary=None, modelif=None):
#        if type(dictionary).__name__.rsplit('.')[0] == 'Series':
#                dictionary = dictionary.values
            
        self.diction = [tkn(sent) for sent in dictionary]
        
        self.word2vec = gensim.models.Word2Vec
        self.modelif = modelif
    
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
        
        
            
        self.wordinfo = self.word2vec(min_count=self.min_word_quote, size = self.quote_size )
        self.wordinfo.build_vocab(self.diction)
        
        if type(self.modelif) != type(None):
            
            model = models.KeyedVectors.load_word2vec_format(self.modelif, binary=True)
            self.wordinfo.build_vocab([list(model.vocab.keys())], update=True)
            self.wordinfo.intersect_word2vec_format(self.modelif, binary=True, lockf=1.0)
        
#        else:
#            
#            self.wordinfo = self.word2vec(min_count=self.min_word_quote, size = self.quote_size )
#            self.wordinfo.build_vocab(self.diction)
            
            
        self.wordinfo.train(self.diction, total_examples=self.wordinfo.corpus_count, epochs=self.wordinfo.iter)
        
        return self.wordinfo
    
    def save_wv(self,path="wv.model"):
        
        self.wordinfo.wv.save_word2vec_format(path, binary=True)
    
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
            
            topN = self.wordinfo.wv.most_similar(d,topn=n_most_similar)
            topNlist = np.array([self.wordinfo[sim[0]] for sim in topN])
            topNten[j] = torch.from_numpy(topNlist)
            
            wordTensors[j] = torch.from_numpy(self.wordinfo[d])
        
        return {
                'tensor':wordTensors,
                'topN':topNten,
                'iids':my_iids,
                'dictionary':my_dictionary,
                }
        
            
            
            