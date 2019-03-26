# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:40:20 2019

@author: Markus.Meister
"""
import numpy as np
import torch as to
import torch.distributed as dist
from torch.multiprocessing import Process

def fastPCA(x,H,w_flag=False,niter=1):
    
    if type(x).__name__ == 'ndarray':
        x = to.tensor(x,requires_grad=True)
    
    for j in range(niter):
            
        if j < 1:
            y=None
            w=None
        y,w,_ = fastPCA_step(x,H,w_flag,y,w)
        
    t = x @ w
    
    return t,w,y
    

def fastPCA_step(x,H,w_flag=False,y=None,w=None):
    
    if type(x).__name__ == 'ndarray':
        x = to.tensor(x,requres_grad=True)
    
    assert H <= x.size(1)
    
    if type(y) == type(None):
        y = to.zeros(H,*x.size()[1:])
        w = to.tensor(np.random.randn(H,*x.size()[1:]))
    
    if H > 0:
        
        y,w,s = fastPCA_step(x,H-1,w_flag,y,w)
        
        s += y @ w[H-1] @ w[H-1].transpose(1,0)
        
        y[H-1] = x - s
        
        var = w.transpose(1,0) @ y[H-1].transpose(1,0) @ y[H-1] @ w / (w.transpose(1,0) @ w)
        
        w[H-1] = w[to.argmax(var,dim=0)]
    
    else:
        
        y[H-1] = x[H-1]
        
        var = w.transpose(1,0) @ x.transpose(1,0) @ x @ w / (w.transpose(1,0) @ w)
        
        g = w[to.argmax(var,dim=0)]
        print(g.size())
        
        w[H-1] = g
        
        s = 0 #y @ w[H-1] @ w[H-1].transpose(1,0)
    
    return y,w,s
    