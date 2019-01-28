# -*- coding: utf-8 -*-
"""
        Scoring Functions

Here, several scoring functions are collected.

@author: Markus Meister
"""
#%% -- imports --
import numpy as np
import pandas as pd
#%% -- Scoring Functions --
# RMSE
def rmse(p, y,convertExpM1=False):
    def f(x,y):
        return np.sqrt(np.mean((x - y)**2))
    return err_frame(y, p, f,convertExpM1)

# RMSLE
def rmsle(p, y,convertExpM1=False):
    
    def f(p,y):
        return np.sqrt(np.mean( ( np.log1p(p) - np.log1p(y) ) ** 2 ))
    return err_frame(y, p, f,convertExpM1)

# MAD
def mad(y, p,convertExpM1=False):
    
    def f(y,p):
        return np.mean(np.abs(y - p))
    return err_frame(y, p, f,convertExpM1)

def err_frame(y, p, f,convertExpM1=False):
    
    y = arr_form(y)
    p = arr_form(p)
    
    if convertExpM1:
        p = np.expm1(p),
        y = np.expm1(y)
        
    return f(y,p)

def arr_form(y):
    
    if type(y).__name__ == 'Series':
        y = y.values
    
    if len(y.shape) < 2:
        y = y[:,np.newaxis]
    
    # fixing transposed data
    yN,yD = y.shape
    if yN < yD:
        y = y.T
    
    return y.squeeze()

