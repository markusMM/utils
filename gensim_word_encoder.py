# -*- coding: utf-8 -*-
"""
        Simple NLP Encoding

Created on Thu Jan 31 13:51:08 2019
@author: Markus.Meister
"""
import os
import pandas as pd
from nltk import word_tokenize as tkn
import gensim 
from gensim import corpora, models, similarities

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