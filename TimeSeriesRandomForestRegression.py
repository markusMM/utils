# -*- coding: utf-8 -*-
"""
Random Forest Regression on Time Series

Here I am trying to run a Random Forest on time windows of states to predict certain
target values in the future.

This may be powerful for case sensitive data with periodic cases as input features.

Also it is robust against overfitting.

Some features, such as holidays, weekdays, ect can be present term

@author: Markus Meister
"""
#%% -- imports --
import numpy as np
import pandas as pd
from pandas import datetime
import matplotlib.pyplot as plt
#from mpi4py import MPI
from sklearn.model_selection import train_test_split as tr_te_split
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append('..')
from utils.data_processing import DB_Proc
import os
try:
    os.stat('data/')
    sys.path.append('data/')
except:
    os.stat('../data/')
    sys.path.append('../data/')
###############################################################################
#%%                             -- MAIN --                                  %%#
##%%-----------------------------------------------------------------------%%##
if ( __name__ == '__main__' ):
    
    # data processor
    dproc = DB_Proc()
    
    #%% main data reading
    my_data = pd.read_excel('hour.csv')
    
    #%% data sampling 24*7 hrs with 24 hrs strafe / shift
    
    excludes = ['instant','dteday','yr','season']
    
    filter = [x for x in my_data if not x in excludes]
    wn_data = dproc.segment_dict(
            my_data.filter(filter),
            seg_size = 7*24,
            seg_strafe = 24,
            window=None,
            )
    
    #%% feature and feature time selection
    
    
    my_N , dT = wn_data['cnt'].shape
    te_N = 120
    tr_N = my_N - te_N -1
    
    
    std_features = ['cnt','casual','registered','weathersit','windspeed','hum','atemp','temp']
    fwd_features = ['hr','mnth','yr','season','weekday','workingday','holiday']
    targets = ['cnt','casual','registered']
    
    n_features  = len(std_features) + len(fwd_features)
    n_targets   = len(targets)
    
    feature_array   = np.zeros([ my_N-1, dT, n_features ])
    target__array   = np.zeros([ my_N-1, dT, n_targets  ])
    
    fwd_f_array = np.zeros([ my_N-1, dT, len(fwd_features) ])
    std_f_array = np.zeros([ my_N-1, dT, len(std_features) ])
    fwd_t_array = np.zeros([ my_N-1, dT, len(targets)      ])
    std_f_i = 0
    fwd_f_i = 0
    fwd_t_i = 0
    for d in wn_data:
        if d in std_features:
            std_f_array[:,:,std_f_i] = wn_data[d][:my_N-1]
            std_f_i += 1
        if d in fwd_features:
            fwd_f_array[:,:,fwd_f_i] = wn_data[d][+1:my_N]
            fwd_f_i += 1
        if d in targets:
            fwd_t_array[:,:,fwd_t_i] = wn_data[d][+1:my_N]
            fwd_t_i += 1
    
    feature_array[ :, :, :len(std_features)  ] = std_f_array
    feature_array[ :, :,  len(std_features): ] = fwd_f_array
    target__array                              = fwd_t_array
    
#    tr_data,te_data = tr_te_split(
#            feature_array, 
#            target__array, 
#            shuffle = False, 
#            test_size = te_N
#            )
    
    tr_features = feature_array[ :tr_N ]
    te_features = feature_array[ -te_N:]
    tr__targets = target__array[ :tr_N ]
    te__targets = target__array[ -te_N:]
    
    
    #%%
    
    
    
    