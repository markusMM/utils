# -*- coding: utf-8 -*-
"""
Data processing
 
a simple use case script for tabular data as dict or aray
 
we have:
     
    segment_arr:
        segments arrays into samples with a specific overlapp
        data        :       the array to be segmented
         
         
 
Created on Fri Dec 07 19:08:57 2018
@author: Nils-Markus Meister
"""
#%% -- imports --
import unittest
import numpy as np
import numba as nb
import scipy
import pandas as pd
from pandas import datetime
from skimage.util.shape import view_as_windows
from sklearn.ensemble import RandomForestRegressor as RFR
from statsmodels.tsa.seasonal import seasonal_decompose
#%% -- main module: DB_Proc --
class DB_Proc:
    
    def __init__(
            self,
            data = None
            ):
        self.data = data
         
    
    def na_rf_interp(self, my_data, na_variables, features='all', rf_params=None):
        
        if not rf_params:
            rf = RFR()
        else:
            rf = RFR(rf_params)
        
        if type(my_data).__name__ == 'dict':
            my_data = pd.DataFrame.from_dict(my_data)
        
        # parse features
        if type(features).__name__ == 'str' or type(features).__name__ != 'dict':
            features_ = {}
            for t in na_variables:
                if      features == 'all':
                    features_[t] = [x for x in my_data if not x in na_variables]
                elif    type(features).__name__ == 'str':
                    features_[t] = [ features ]
                else:
                    features_[t] =   features
            features=features_
        
        my_rfs = {}
        for f in na_variables:
            rf_ = rf
            
            #  nans
            id_na = np.isnan(my_data[f])
            if id_na.sum() in [0,my_data[f].size]:
                continue # nothing to interpolate
            
            rf_.fit(
                    my_data.filter(features[f])[(id_na-1).astype(bool)].values,
                    my_data[f][(id_na-1).astype(bool)].values,
                    )
            
            my_data[f][id_na] = rf_.predict(
                    my_data.filter(features[f])[id_na].values
                    )
            
            my_rfs[f] = rf_
            rf_ = None
        
        return my_data,my_rfs
            
    
    def parse_win(self,window,seg_size):
        if   type(window).__name__ == 'ndarray':
                window = window[:seg_size]
                 
        else:
            if type(window).__name__ == 'str':
                if      window.lower() == 'hamming':
                    a0 = 0.53836
                elif    window.lower() == 'hann':
                    a0 = 0.5
                else:
                    print('warning: window not found!')
                    a0 = 0.53836
            else:
                try:
                    a0 = window
                except:
                    print('warning: coefficient not read!')
                    a0 = 0.53836
            window = np.arange(0,seg_size)
            window = a0 - (1-a0)*np.cos(2*np.pi*window/(seg_size-1))
         
        return window
     
    def segment_frame(
            self, 
            data = None,
            seg_size   = 24*7,
            seg_strafe = 24*1,
            fields = 'all',
            window = 'hamming',
            ):
        '''
        Uses the function segment_dict with the same parameters.
        It simply first converts the data into a dictionary runs the function
        segment_dict and converts the result back into a data frame.
        '''
        data = pd.from_dict(
                self.segment_dict(
                        data.to_dict(),
                        seg_size,
                        seg_strafe,
                        fields,
                        window,
                        )
                )
        return data
     
    def segment_dict(
            self, 
            data = None,
            seg_size   = 24*7,
            seg_strafe = 24*1,
            fields = 'all',
            window = 'hamming',
            ):
        if type(data).__name__ == 'NoneType':
            if self.data != None:
                data = self.data.copy()
            else:
                raise('Error: Data not declared!')
                 
        if not type(window).__name__ == 'NoneType':
             
            window = self.parse_win(window,seg_size)
         
        if fields == 'all':
            fields = list(map(
                    lambda d : d, data
                    ))
        elif type(fields).__name__ == 'str':
            print('Error: Field mode not understood!')
        
        nw_data = {}
        for fld in fields:
            nw_data[fld] = self.segment_arr(
                    data[fld],
                    seg_size   = seg_size,
                    seg_strafe = seg_strafe,
                    window = window,
                    )
        return nw_data
     
    def segment_arr(
                    self,
                    arr,
                    seg_size   = 24*7,
                    seg_strafe = 24*1,
                    window = None,
            ):
        # is (Ih-Dh+1, Iw-Dw+1, Dh, Dw)
        if type(arr).__name__ == 'Series':
            arr = arr.values.astype(float)
            
        arr = view_as_windows(
                arr, 
                window_shape = [seg_size], 
                step = seg_strafe,
                )
        if type(window).__name__ == 'ndarray':
            window = self.parse_win(window,seg_size)
            for j in range(arr.shape[0]):
                arr[j] = np.convolve( arr[j], window, 'same' )
        
#        if ser_flg:
#            data = pd.Series(data)
        
        return arr
    
    def ovadd_arr(
            self,
            arr,
            shift=24,
            ):
        
        size = arr.shape
        wN,wL = size[0],size[1]
        
        assert wL > shift
        
        N = wN * ( shift ) + wL - shift
        
        sh = int(shift)
        dim = 'N'
        if len(size) > 2:
            wD = [size[2]]
            dim += ', %i' %size[2]
            for d in range(3,len(size)):
                wD.append(size[d])
                dim += ', %i' %size[d]
        
        
        one = np.ones_like(arr)
        
        def ovadd_iter(new_arr,arr,dn,sh,wL):
            new_arr[ dn*( sh ):( dn + 1)*( sh ) ] += arr[dn,:sh]
            new_arr[ dn*( sh ) + wL : ( dn + 1)*( sh ) + wL ] += arr[dn+1,-sh:] 
            new_arr[ ( dn + 1 )*( sh ):( dn + 1)*( sh ) + wL - sh ] = (
                    arr[dn,sh:] +
                    arr[dn+1,:-sh]
                    )
            return new_arr
        
        new_arr = eval('np.zeros([%s])' %dim)
        one_arr = new_arr.copy()
        for dn in range(wN-1):
            
            new_arr = ovadd_iter(new_arr,arr,dn,sh,wL)
            one_arr = ovadd_iter(one_arr,one,dn,sh,wL)
            
        return new_arr / one_arr
    
    # additive seasonal decomposition
    def sel_seasonal_dec(self, my_data, targets, fs):
        my__targets = targets
        new__targets = []
        for t in my__targets:
            if t not in my_data:
                print('WARNING: %s not found in data!' %t)
                continue
            serdec = seasonal_decompose(my_data[t], model='additive',freq=fs)
            my_data[t+'_seasonal'] = serdec.seasonal
            my_data[t+'_trend'] = serdec.trend
            my_data[t+'_residual'] = serdec.resid
            new__targets.append(t+'_residual')
            new__targets.append(t+'_trend')
        or__targets = my__targets
        my__targets = new__targets
        
        return my_data, my__targets, or__targets
    
    def sel_seasonal_cmp(self, my_data, targets):
        
        for j,t in enumerate(targets):
            if t+'_seasonal' not in my_data:
                print('WARNING: Decomposition for %s not found in data!' %t)
                continue
            my_data[t]  = my_data[t+'_trend']
            my_data[t] += my_data[t+'_seasonal']
            my_data[t] += my_data[t+'_residual']
            
        return my_data
    
#%% -- unit tests --
class TestStringMethods(unittest.TestCase):
    
    def test_sel_seasonal_cmp(self):
        
        dproc = DB_Proc()
        
        unit = np.random.rand(  500 )
        norm = np.random.randn( 500 )
        
        dictionary = {
                'unit':unit,
                'norm':norm
                }
        
        dec_elem = ['unit','norm','rndo']
        fs = 10
        
        my_data, my_elem, or_elem = dproc.sel_seasonal_dec(
                dictionary, dec_elem, fs
                )
        
        if type(my_data).__name__ == 'dict':
            my_data = pd.DataFrame.from_dict(my_data)
        
        new_data = dproc.sel_seasonal_cmp(
                my_data,
                or_elem,
                )
        
        self.assertEqual( 
                my_data.filter(or_elem).values, 
                new_data.filter(or_elem).values 
                )
    
    def test_sel_seasonal_dec(self):
        
        dproc = DB_Proc()
        
        unit = np.random.rand(  500 )
        norm = np.random.randn( 500 )
        
        dictionary = {
                'unit':unit,
                'norm':norm
                }
        
        dec_elem = ['unit','norm','rndo']
        fs = 10
        
        my_data, my_elem, or_elem = dproc.sel_seasonal_dec(
                dictionary, dec_elem, fs
                )
        
        if type(my_data).__name__ == 'dict':
            my_data = pd.DataFrame.from_dict(my_data)
        
        assert 'rndo_trend' not in my_elem
        assert 'unit_trend' in my_elem
        assert my_data['unit_seasonal'].std() <= 0.1
    
    def test_na_rf_interp(self):
        
        dproc = DB_Proc()
        
        N = 18
        D = 4
        te_data = {}
        
        te_data['bads'] = np.random.rand(N)-.5
        te_data['bads'][4] = 0 
        te_data['bads'][2] = 0 
        te_data['bads'][te_data['bads']==0] = np.nan
        
        id_na = np.isnan(te_data['bads'])
        a = te_data['bads'][(id_na-1).astype(bool)]
        
        for j in range(D-1):
            te_data[str(j)] = np.random.randn(N)-.5
        
        it_data,my_regs = dproc.na_rf_interp(te_data,['bads'])
        d = te_data['bads'][(id_na-1).astype(bool)]
        
        it_data,my_regs = dproc.na_rf_interp(te_data,['bads'],[str(2),str(0)])
        e = te_data['bads'][(id_na-1).astype(bool)]
        
        self.assertEqual(d, e)
        self.assertEqual(e, a)
        self.assertEqual(a, d)
    
    def test_ovadd_arr(self):
        dproc = DB_Proc()
        a = np.array([8,8,8,9,9,9,1,1,1,2,2,2])
        b = np.array([
                [8,8,8],
                [8,8,9],
                [8,9,9],
                [9,9,9],
                [9,9,1],
                [9,1,1],
                [1,1,1],
                [1,1,2],
                [1,2,2],
                [2,2,2]
                ])
        c = dproc.segment_arr(a,3,1)
        
        d = dproc.ovadd_arr(c,1)
        e = dproc.ovadd_arr(b,1)
        
        self.assertEqual(d, e)
        self.assertEqual(e, a)
        self.assertEqual(a, d)
    
    def test_segment_arr(self):
        dproc = DB_Proc()
        a = np.array([8,8,8,9,9,9,1,1,1,2,2,2])
        b = np.array([
                [8,8,8],
                [8,8,9],
                [8,9,9],
                [9,9,9],
                [9,9,1],
                [9,1,1],
                [1,1,1],
                [1,1,2],
                [1,2,2],
                [2,2,2]
                ])
        c = dproc.segment_arr(a,3,1)
        mse = np.mean((b-c)**2)
        print('MSE: %f' %mse)
        self.assertEqual(b, c)
        
    
    def test_segment_dict(self):
        dproc = DB_Proc()
        v = np.array([8,8,8,9,9,9,1,1,1,2,2,2])
        a = {}
        a['x'] = v
        a['y'] = np.flipud(v)
        a['z'] = np.append(a[:6],a[-6:])
        b = {}
        for d in a:
            b[d] = dproc.segment_arr(a[d],3,1)
        c = dproc.segment_dict(a,3,1)
        mse = []
        for d in b:
            mse.append(np.mean((b[d]-c[d])**2))
        print('MSE: %f' %mse/len(mse))
        self.assertEqual(b, c)
    
    def test_segment_frame(self):
        '''
        Because the segment_frame function does only output a dict anyways, we here 
        just compare a dict with the output.
        
        This is done because Pandas does not seem to support multi dimensional 
        frames.
        '''
        dproc = DB_Proc()
        v = np.array([8,8,8,9,9,9,1,1,1,2,2,2])
        a = {}
        a['x'] = v
        a['y'] = np.flipud(v)
        a['z'] = np.append(a[:6],a[-6:])
        a = pd.DataFrame.from_dict(a)
        b = {}
        for d in a:
            b[d] = dproc.segment_arr(a[d],3,1)
        c = dproc.segment_dict(a,3,1)
        mse = []
        for d in b:
            mse.append(np.mean((b[d]-c[d])**2))
        print('MSE: %f' %mse/len(mse))
        self.assertEqual(b, c)
