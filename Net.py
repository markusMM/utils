#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 13:24:09 2018
 
@author: Markus Meister
@instute: University Oldenburg (Olb)
@devision: Machine Learning
@faculty:FVI Math.&Nat.Sci.
"""
#%% imports
import torch
 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd, optim
import numpy as np
#from OneHotEncoder import one_hot_encode
 
#%% prefix
 
if 'xrange' not in locals():
    def xrange(mx,mn=0):
        return range(mn,mx)
    globals()['xrange'] = xrange
 
#%% MLP class
class Net(nn.Sequential):
 
    def __init__(self, n_features=2, n_h=[], n_Classes=2,mu=0,std=.25,
                 early_stopping=True, tol=.5e-08, loss_fun='CrossEntropy',
                 validation_frac=.1,lafcns=None,ltypes=None,max_iter=250,
                 opt='Adam',learning_rate=0.001,batch_size='auto',fc_init=0,
                 layers=None, warm_start=True):
        super(Net,self).__init__() #Net, self
         
        #std loss
        #self.loss_fun = torch.nn.CrossEntropyLoss()
        if type(n_h).__name__ == 'str':
            n_h = eval(n_h)
        n_h = np.array(n_h)
        n_units = np.append(np.append(np.array(n_features), n_h), n_Classes).tolist()
         
        if ltypes == None and not fc_init:
            fc_init = 0
            self.n_layers = len(n_h) + 1
        elif ltypes != None and fc_init == 1:
            fc_init = True
            self.n_layers = np.min(\
                                   len(ltypes['type']),\
                                   len(ltypes['args'])\
                                   )
        if layers != None:
            fc_init = 3
            self.n_layers = len(layers)
         
        if fc_init != 3:
            if lafcns == None and fc_init != 3:
                lafcns = np.repeat('relu',len(n_h)+1).tolist()
            elif fc_init and layers != None and fc_init != 3:
                if len(lafcns) < len(layers):
                    dl = np.abs(len(lafcns) - len(layers))
                    lafcns = np.array(lafcns)
                    lafcns = np.append(lafcns, np.repeat(lafcns[-1],dl))
            if ltypes == None and fc_init != 3:
                ltypes = {}
                ltypes['type'] = np.repeat('Linear',len(n_h)+1).tolist()
                ltypes['args'] = {}
            if not hasattr(ltypes,'args',) and fc_init != 3:
                ltypes['args'] = {}
            if ltypes != None and len(ltypes['args']) < len(ltypes['type']) and fc_init != 3:
                for j in range(1,len(n_h)+2):
                    ltypes['args'][j-1] = "%d,%d" \
                    %(int(n_units[j-1]),int(n_units[j]))
                 
         
        #self.lafcns = lafcns
        #self.ltypes = ltypes['name']
         
        #parse loss function
#        if loss_fun != 'default' and loss_fun != None:
#            if type(loss_fun) == str:
#                self.loss_fun = loss_fun.strip('Loss')
#                #exec('self.loss_fun = torch.nn.%sLoss()' %(loss_fun))
#            elif    type(loss_fun).__name__.strip('function')    !=\
#                    type(loss_fun).__name__\
#            or      type(loss_fun).__name__.strip('method')      !=\
#                    type(loss_fun).__name__:
#                self.loss_fun = loss_fun
#        else:
#            self.loss_fun = torch.nn.MSELoss()
         
        if tol != None:
            self.tol = tol
        else:
            self.tol = 1e-8
         
        n_hl = len(n_h)
         
         
        if fc_init <= 2:
            # for each layer with given data
            for l in range(len(ltypes['type'])):
                 
#                exec('self.%s = nn.%s(%s)' \
#                     %(\
#                       ltypes['type'][l],\
#                       ltypes['args'][l]\
#                       )\
#                     )
###################################################################################################################################################################################                     
                self.add_module(\
                                "(%d)" %(l),\
                                eval(\
                                     'nn.%s(%s)' \
                                     %(ltypes['type'],\
                                       ltypes['args'][l])\
                                    )\
                                )
#                exec('torch.nn.init.normal(self.%s.weight, mean=mu, std=std)' \
#                     %ltypes['name'][l]\
#                     )
        else:
            # for each given layer from dict
            for l in layers:
                self.add_module(l,layers[l])
                #exec('torch.nn.init.normal(self.%s.weight, mean=mu, std=std)' \
                #     %l\
                #     )
         
#        if fc_init and fc_init < 2:
#            #self.ltypes = ltypes
#            #self.lafcns = lafcns
#            self.ini_fwd_types()
#        if fc_init > 1:
#            self.ini_fwd_layer()
         
        self.n_hl   = n_hl
        self.n_h    = n_h
        self.n_features = n_features
        self.n_Classes  = n_Classes
        self.validation_frac = early_stopping*validation_frac
        self.ealy_stopping = early_stopping
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.warm_start = warm_start
         
        self.c_losses = np.array([
                'CrossEntropy',
                'NLL',
                ],dtype=str)
         
         
        if type(opt).__name__ == 'str':
            exec(\
                 'self.opt=optim.%s(%s,lr=%f)' \
                 %(\
                   opt,\
                   'params=self.parameters()',\
                   learning_rate\
                   )\
                 )
        else:
            self.opt = opt
         
    def fit(self,x,l,loss_fun=None,opt=None, batch_size=None, epochs=None, verb=False):
         
        l = l.squeeze()
         
        if batch_size == None:
            batch_size = self.batch_size
         
        n_total = x.shape[0]
        D = x.shape[1]
        N = n_total
         
        assert D == self.n_features
        if batch_size == 'auto' or type(batch_size).__name__ == 'str':
            batch_size = N
         
        if loss_fun == None:
            try:
                losses = eval('torch.nn.%sLoss()' %(self.loss_fun))
                loss_fun = self.loss_fun
            except:
                losses = torch.nn.MSELoss()
                loss_fun = 'MSE'
         
        if np.sum(self.c_losses == loss_fun) == 0 \
        and len(l.shape) <= 1:
            l = self.binarize(l,self.n_Classes)
         
        if epochs == None:
            epochs = self.max_iter
        if opt == None:
            opt = self.opt #optim.SGD(params=self.parameters(),lr=0.01)
         
        i_shuffle   = np.arange(n_total)
        num_batches = int(n_total/batch_size)
        this_loss   = np.ones((epochs,1))
        loss_valid  = np.ones((epochs,1))
        valid_size  = int(batch_size*self.validation_frac)
        bdata_size  = batch_size - valid_size
         
        e_tol = 0
        v_tol = 0
        for e in range(epochs):
             
            # verbous print
            if verb:
                print('Epoch %s out of %s'%((e+1),epochs))
             
            # data shuffle
            np.random.shuffle(i_shuffle)
            x=x[i_shuffle]
            l=l[i_shuffle]
             
            # data torchify
            torch_data      = autograd.Variable(torch.from_numpy(x).float())
            if np.sum(self.c_losses == loss_fun) != 0:
                torch_labels    = autograd.Variable(torch.from_numpy(l)).long()
            else:
                torch_labels    = autograd.Variable(torch.from_numpy(l)).float()
             
            # batch train
            for n in range(num_batches):
                dat = torch_data[(batch_size*n):(batch_size*(n+1)),:]
                lab = torch_labels[(batch_size*n):(batch_size*(n+1))]
                if self.validation_frac > 0.0:
                    val = self(dat[-valid_size:])
                out = self(dat[:bdata_size])
                loss = losses(out,lab[:bdata_size])
                self.zero_grad()
                loss.backward()
                opt.step()
                #print(loss.data.mean())
                this_loss[e] += loss.data.mean().numpy()
                #print(loss)
                if valid_size > 0:
                    loss_valid[e] = losses(val,lab[-valid_size:]).data.mean().numpy()
             
            this_loss[e]   = this_loss[e] /num_batches
            if valid_size > 0:
                loss_valid[e]  = loss_valid[e] /num_batches
             
             # verbos print
            if verb:
                 print('current loss',this_loss[e])
                  
            if e > 0:
                if np.abs((this_loss[e-1] - this_loss[e]))\
                <  self.tol:
                    e_tol += 1
                else:
                    e_tol = 0
                if valid_size > 0:
                    if  loss_valid[e] - loss_valid[e-1] \
                    >   self.tol:
                        v_tol += 1
                    else:
                        v_tol = 0
            elif hasattr(self,'loss_valid_'):
                if valid_size > 0:
                    if  loss_valid[e] - self.loss_valid_ \
                    >   self.tol:
                        v_tol += 1
             
            # tolerance for 'convergence' reached
            if e_tol >= 2 or v_tol >= 2:
                break
         
        if hasattr(self,'loss_valid_'):
            self.loss_valid_ = min(np.mean(loss_valid[e]), self.loss_valid_)
        else:
            self.loss_valid_ = np.mean(loss_valid[e])
         
        if hasattr(self,'loss_curve_'):
            self.loss_curve_ = np.append(self.loss_curve_,this_loss)
        else:
            self.loss_curve_ = this_loss
        if hasattr(self,'Iters_'):
            self.Iters_ += e+1
        else:
            self.Iters_ = e+1
     
#    def forward(self, V):
#        return super().forward(V).squeeze()
     
    def predict_proba(self,x):
        x = self.tensor_check(x)
        return self.forward(x)
     
    def predict(self,x):
        x = self.tensor_check(x)
        return F.softmax(self.predict_proba(x),dim=-1)
     
    def binarize(self,target,n_Classes=None):
        if n_Classes == None:
            n_Classes = np.unique(target).shape[0]
         
        labels = np.zeros([target.shape[0],n_Classes],dtype=int)
        for j in range(n_Classes):
            labels[:,j] = target == j
         
        return labels.astype(int)
             
    def tensor_check(self,x):
        if type(x).__name__ == 'ndarray':
            x = autograd.Variable(torch.from_numpy(x).float())
        return x
     
#%% example code
if __name__ == "__main__":
     
    #%% load data
 
    import matplotlib.pyplot as plt
     
#    import os
#    import tables
#    import sys
#    
#    sys.path.append("../../Data/simple_clusters/")
     
    #params=np.loadtxt('../../Data/simple_clusters/param_save.txt')
    #storage=np.loadtxt('../../Data/simple_clusters/data_half_moon.txt')
     
    params=np.loadtxt('param_save.txt')
    storage=np.loadtxt('data_half_moon.txt')
     
    data=storage[:,:2]
    labels=storage[:,2]
     
    n_features = data.shape[1]
    n_Classes  = np.unique(labels).shape[0]
    #n_Classes=int(params[0])
    #n_Cluster=int(params[1])
    #n_per_Cluster=int(params[2])
    #n_total=n_per_Cluster*n_Cluster
    n_total = data.shape[0]
    #n_h1=2*n_Cluster
     
    n_h1 = 32
     
    #%% define parameters
     
    learning_rate = .001
     
    opt_fun = 'Adam'#'SGD'
     
    #%% instanciate MLP and optimizer
     
    epochs = 50
    batch_size = 10
     
#    device = torch.device('cpu')
#    if torch.cuda.is_available():
#        device = torch.device('cuda')
     
    nt_mods = {
            '0':nn.Linear(n_features,n_h1),
            '1':nn.Softsign(),
            '2':nn.Linear(n_h1,n_h1),
            '3':nn.Tanh(),
            '4':nn.Linear(n_h1,n_h1*2),
            '5':nn.LeakyReLU(),
            '6':nn.Linear(n_h1*2,n_Classes),
            '7':nn.ReLU(),
            }
     
    net = Net(n_features,[n_h1*2,n_h1*(1+n_Classes),n_h1*2],n_Classes,\
                          early_stopping=True,validation_frac=.1,
                          batch_size=batch_size,max_iter=epochs,\
                          layers=nt_mods,fc_init=0,tol=1e-10,opt='Adam'\
                          , loss_fun='CrossEntropy')
    print(net)
     
#    net = nn.Sequential()
#    for ty in nt_mods:
#        net.add_module(ty,nt_mods[ty])
#    print(net)
#    
#    loss_fun='CrossEntropy'
#    if type(loss_fun) == str:
#                loss_fun = loss_fun.strip('Loss')
#                exec('net.loss_fun = torch.nn.%sLoss()' %(loss_fun))
#    opt = 'Adam'
#    if type(opt).__name__ == 'str':
#        exec(\
#             'opt=optim.%s(%s,lr=%f)' \
#             %(\
#               opt,\
#               'params=net.parameters()',\
#               learning_rate\
#               )\
#             )
#    net.opt = opt
#    
#    opt=None
#    if opt == None:
#        exec('opt=optim.%s(%s,lr=%f)' %(opt_fun,'params=net.parameters()',\
#                           learning_rate)\
#        )
#    
#    net.batch_size=batch_size
#    net.max_iter=epochs
#    net.validation_frac=.25
#    net.early_stopping=True
#    net.tol=.5e-08
#    net.n_features=n_features
#    net.n_hidden = (n_h1*2,)
#    net.n_Clases = n_Classes
#    
#    net.fit = Net.fit
    #net.forward = Net.forward
    #net.opt = opt
     
    #%% convert data
    point=autograd.Variable(\
                            torch.from_numpy(\
                                             np.matrix(\
                                                       [[1.0,2.0],\
                                                        [2.0,3.0],\
                                                        [3.0,4.0]])\
                            ).float()\
    )
     
    lab = labels
    labels = np.repeat(np.zeros_like(lab)[:,np.newaxis],3,axis=1)
    for j in range(n_Classes):
        labels[:,j] = lab == j
    labels = lab.astype(int)
    torch_data      = autograd.Variable( torch.from_numpy( data ) ).float() 
    torch_labels    = autograd.Variable( torch.from_numpy( labels) ).double()
     
     
    #%% start training
     
    net.fit(data,labels,verb=True)
     
    #%% accuracy test
     
    # accuracy on test set
    pred_labels = net(torch_data)
    pred_labels = np.argmax(pred_labels.data.numpy(),axis=1)
    print(pred_labels)
    print(lab)
    count_right = np.sum((pred_labels==lab.astype(int))).astype(float)
     
    # verbose output
    print('accuracy on train-set: %.4f' %(count_right/n_total))
     
    #%% plots: pre-allocation
    x_grid_min=np.amin(data[:,0])
    x_grid_max=np.amax(data[:,0])
    y_grid_min=np.amin(data[:,1])
    y_grid_max=np.amax(data[:,1])
    x_grid=200
    y_grid=200
    xx_em=np.linspace(x_grid_min,x_grid_max,x_grid)
    yy_em=np.linspace(y_grid_min,y_grid_max,y_grid)
    m=np.zeros((x_grid,y_grid))
     
    col_count=1
    for i in range(0,x_grid):
            for j in range(0,y_grid):
                 
                point=autograd.Variable(\
                                        torch.from_numpy(np.array([[xx_em[i]],\
                                        [yy_em[j]]]).T).float()\
                                        )
                 
                f=(net(point)).data.numpy()
                 
                for c in range(0,n_Classes):
                    if(np.amax(f)==f[0,c]):
                        m[y_grid-(j+1),i]=c
             
            if(col_count*0.2<=i/x_grid):
                print('Column %s of %s'%(i,x_grid))
                col_count=col_count+1
     
    #%% plots: images
    plt.figure()
    plt.imshow(m,interpolation='none',extent=[x_grid_min,x_grid_max,y_grid_min,y_grid_max])
    plt.scatter(data[:,0],data[:,1],s=1,c=labels,cmap='Spectral')
     
    #x_loss=np.arange(net.Iters_+1)
    plt.figure()
    plt.plot(np.arange(net.Iters_)+1,net.loss_curve_[:net.Iters_])
