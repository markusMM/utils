# -*- coding: utf-8 -*-
"""

    Advenced String Analysis Methods

contains of the following functions:
    
    - Wordstemm Cluster : wordstemm_clusters
        ... clusters lines of text into wordstemm items, groups them one-hot encodes 
            them and returns a filtered dataframe with the original data plus 
            corresponding wordstemms and one-hot encoding and one dataframe with just
            all the string lines falling into a corresponding wordstemm.
        
    -...

Created on Thu Mar 14 10:43:34 2019
@author: Markus.Meister
"""
#%% -- imports --
import pandas as pd
import numpy as np
from scii_funs import *
from df_funs import write_df_to_excel, eval_value_dfs, dict_to_df, mean_values_df
#%% -- globals --
w_words_ = np.array([
    "wer ",
    "wem ",
    "wen ",
    "wessen ",
    "wie ",
    "wann ",
    "wo ",
    "welche",
    "was ",
    "wobei ",
    "womit ",
    "woran ",
    "wohin ",
    "wobei ",
    "wo ",
    "weshalb ",
    "warum ",
    "wieso ",
    "wieviel"
    "worauf ",
    "worum ",
    "wovor ",
    "wodurch ",
    "woher ",
    "weswegen ",
    "woraus ",
    ])
q_words_ = np.array([
    "who ",
    "whom ",
    "whose ",
    "when ",
    "which ",
    "what ",
    "what's",
    "where ",
    "why ",
    "how ",
    ])
#%% -- functions --
"""
    Wordstemm Clusters

This function clusters strings in "wordstemms"

"""
def wordstemm_clusters(
        my_data = None, 
        str_key = 'Keyword', 
        filter_keys = [
                'Wettbewerber / Aufteilung', 
                'Organisch vs. Paid', 
                'Keyword', 
                'Ø Suchanfragen pro Monat', 
                ], 
        en_qflg = 0, de_qflg = 1, 
        n_st_min = 4, 
        thresholds = (4,500), 
        ):
    
    # re-defining globals to avoid possible overwrites (probably unnecessary)
    q_words = q_words_.copy()
    w_words = w_words_.copy()
    
    if type(my_data) == type(None):
        return None
    
    raw = my_data[str_key].values.tolist()
    
    if not en_qflg:
        q_words = np.array([])
        
    if not de_qflg:
        w_words = np.array([])
    
    #generate set of all possible groupings
    groups = set()
    for line in raw:
        data = line.strip().split()
        for item in data:
            if len(item) >= n_st_min:
                groups.add(item)
    
    group_dict = {g:[] for g in groups}
    group_dict['questions'] = []
    
    #parse input into groups
    for group in groups:
        if len(group) < n_st_min:
            continue
        print("Group \'%s\':" % group)
        for line in raw:
            # lists for each specific question type to be present
            w_check = list(map(lambda x: ' '+x in ' '+line+' ', w_words))
            q_check = list(map(lambda x: ' '+x in ' '+line+' ', q_words))
            if np.array(w_check).sum():
                group_dict['questions'].append(line.strip())
                if w_words[w_check][0] not in group_dict:
                    group_dict[w_words[w_check][0]] = [line.strip()]
                else:
                    group_dict[w_words[w_check][0]].append(line.strip())
                    
            if np.array(q_check).sum():
                group_dict['questions'].append(line.strip())
                if q_words[q_check][0] not in group_dict:
                    group_dict[q_words[q_check][0]] = [line.strip()]
                else:
                    group_dict[q_words[q_check][0]].append(line.strip())
                    
            if line.find(group) is not -1:
                print(line.strip())
                group_dict[group].append(line.strip())
        print()
    
    # all questions will be a specific exception
    exceptions = np.array([],dtype=str)
    exceptions = np.append(exceptions,np.array(w_words))
    exceptions = np.append(exceptions,np.array(q_words))
    group_df = dict_to_df(group_dict, thresholds=thresholds,exceptions=exceptions)
    group_df[:][group_df=='nan'] = ''
    group_df = group_df.reindex(sorted(group_df.columns), axis=1)
    
    data_df = my_data.filter(filter_keys)
    data_df['wordstemms'] = pd.Series(np.empty(data_df[str_key].values.shape,dtype=str))
    for gr in sorted(group_df.columns):
        data_df.loc[data_df[str_key].isin(group_df[gr]),'wordstemms'] = \
        data_df['wordstemms'].loc[data_df[str_key].isin(group_df[gr])].values + gr+', '
        data_df[gr] = pd.Series(data_df[str_key].isin(group_df[gr]).astype(int))
    
    return data_df, group_df

def wordstemm_bag(
        my_data = None, 
        str_key = 'Keyword', 
        filter_keys = [
                'Wettbewerber / Aufteilung', 
                'Organisch vs. Paid', 
                'Keyword', 
                'Ø Suchanfragen pro Monat', 
                ], 
        en_qflg = 0, de_qflg = 1, 
        n_st_min = 4, 
        thresholds = (4,500), 
        ):
    
    # re-defining globals to avoid possible overwrites (probably unnecessary)
    q_words = q_words_.copy()
    w_words = w_words_.copy()
    
    if type(my_data) == type(None):
        return None
    
    raw = my_data[str_key].values.tolist()
    
    if not en_qflg:
        q_words = np.array([])
        
    if not de_qflg:
        w_words = np.array([])
    
    # dictionary with possible n-gramms and all its cases
    group_dict = {}
    group_dict['questions'] = []
    
    # generate set of all possible groupings
    groups = set()
    for line in raw:
        data = line.strip().split()
        for group in data:
            
            if len(group) < n_st_min:
                continue
            
            groups.add(group)
            
            w_check = list(map(lambda x: ' '+x in ' '+line+' ', w_words))
            q_check = list(map(lambda x: ' '+x in ' '+line+' ', q_words))
            if np.array(w_check).sum():
                group_dict['questions'].append(line.strip())
                if w_words[w_check][0] not in group_dict:
                    group_dict[w_words[w_check][0]] = [line.strip()]
                else:
                    group_dict[w_words[w_check][0]].append(line.strip())
            if np.array(q_check).sum():
                group_dict['questions'].append(line.strip())
                if q_words[q_check][0] not in group_dict:
                    group_dict[q_words[q_check][0]] = [line.strip()]
                else:
                    group_dict[q_words[q_check][0]].append(line.strip())
                    
            if line.find(group) is not -1:
                if not group in group_dict.keys():
                    group_dict[group] = []
                group_dict[group].append(line.strip())
    
    if not en_qflg:
        q_words = np.array([])
        
    if not de_qflg:
        w_words = np.array([])
    
    exceptions = np.array([],dtype=str)
    exceptions = np.append(exceptions,np.array(w_words))
    exceptions = np.append(exceptions,np.array(q_words))
    
    return dict_to_df(group_dict, thresholds=thresholds,exceptions=exceptions)

#def indep_ngrams(text,stop_words=[]):
#    
#    
#    
#    for 
#    if type()
#    
#    
#    return ngram_list
