# -*- coding: utf-8 -*-
"""
EDA module

This module can do some Exploratory data analysis.
It is originally dedicated to explore 'scores', 'fitnesses' or 'counts' based 
on arbitary dependencies.

Thus, many different correlations can be analyzed.

However, this module is far from completion, as it is in its early development
state. Also, based on personal use, more functionality will be included.

Parameters:
    
    variables   :   <list<str>> of variable names to be considered
    features    :   <list<str>> of feature  -||-  || ||  -- || --
    
functions:
    histo(self,data,targets):
        ''does a histogram of variables with targets or self.vaiables''
    scatter(self,data,feature_names,variable_names)


Created on Sat Dec 01 22:09:22 2018

@author: Nils-Markus Meister <markus.meister@uol.de>
"""
#%% -- imports --
import unittest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import distutils.dir_util as dirutil
import pandas as pd
#from mpi4py import MPI
#from utils.data_processing import DB_Proc
#%% -- helper fcns --
def dict2arr(dict,fields):
    select = []
    fields = np.array(fields,dtype=str)
    for fld in fields:
        if fld in dict:
            select.append(dict[fld])
    return select
        
#%% -- EDA_MSC Module --
class EDA_MSC:
    
    def __init__(
            self,
            data=None,
            features=None,
            targets=None,
            sample_size=24*7,
            sample_strafe=24*1,
            ):
        self.targets = targets
        self.features = features
        self.data = data
    
    def count(
            self,
            data=None,
            features=[],
            feature_names=[],
            scales={},
            save = False,
            figdir = 'plot/count',
            ):
        '''
        histo(data)
        prints a standard 2D histogram of the data value count
        
        Parameters:
            
            data       :   a dictionary containing formatted data of features
                            and targets to be considered
                            
            features   :   list of the input features to be considered for
                            the regression model.
            
           . *_names   :   list of the names for features respectively 
            
            error_std  :   if loading a standard deviation error bar
            
            save       :   if saving the output figures
            
            figdir     :   where to save the figures (relative or full path)
            
        '''
        # ask for data
        if type(data).__name__ == 'NoneType':
            data = self.data.copy()
            if type(data).__name__ == 'NoneType':
                print('Error: No data found!')
        
        # ask for targets and features
        if features == []:
            features = self.features
        if feature_names == []:
            feature_names = features
        
        # use numpy array convention
        features = np.array(features)
        
        # for each variable and each features plot a histogram
        count = np.zeros(features.shape)
        
        plt.figure(facecolor='None')
        vx = data[features[0]].unique()
        #vm = np.arange(vx.shape[0])
        for t,vn in enumerate(vx):
            # for each feature value
            for f,fx in enumerate(features):
                count[f] = (data[fx]==vn).sum()
            
            plt.bar(
                    feature_names, 
                    count*scales.get(fx,1), 
                    alpha = .45, 
                    label = str(vn),
                    #bins=bins
                    )
        # labels and legend
        #plt.xlabel(feature_names[f])
        #plt.ylabel()
        plt.xticks(rotation=80)
        plt.legend()
        
        # saving figure if turned on
        if save:
            fignam = \
            fx+'__'+\
            '.png'
            dirutil.mkpath(figdir)
            plt.savefig(figdir+'/'+fignam)
        
        plt.tight_layout()
        plt.show()
        
        return plt.gcf()
    
    def histo(
            self,
            data=None,
            features=[],
            targets=[],
            feature_names=[],
            target_names=[],
            bins=False,
            error_std=False,
            scales={},
            save = False,
            figdir = 'plot/histo',
            ):
        '''
        histo(data)
        prints a standard 2D histogram of the data utility count utilities are  
        separate in targets and the x-axis in features
        
        Parameters:
            
            data       :   a dictionary containing formatted data of features
                            and targets to be considered
                            
            features   :   list of the input features to be considered for
                            the regression model.
                            
            targets    :   list of the variables to be dependenton the input 
                            features
            
           . *_names   :   list of the names for features and targets 
                             respectively 
            
            error_std  :   if loading a standard deviation error bar
            
            save       :   if saving the output figures
            
            figdir     :   where to save the figures (relative or full path)
            
        '''
        # ask for data
        if type(data).__name__ == 'NoneType':
            data = self.data.copy()
            if type(data).__name__ == 'NoneType':
                print('Error: No data found!')
        
        # ask for targets and features
        if targets == []:
            targets = self.targets
        if features == []:
            features = self.features
        if target_names == []:
            target_names = targets
        if feature_names == []:
            feature_names = features
        
        # use numpy array convention
        features = np.array(features)
        targets = np.array(targets)
        
        # for each variable and each features plot a histogram
        for f,fx in enumerate(features):
            plt.figure()
            vx = data[fx].unique()
            for t,vn in enumerate(targets):
                mean = np.zeros_like(vx)
                stdd = np.zeros_like(vx)
                # for each feature value
                for i,v in enumerate(vx):
                    mean[i] = data[vn][data[fx]==v].mean()
                    try:
                        stdd[i] = data[vn][data[fx]==v].std()
                    except:
                        stdd[i] = 0
                #vs = va.std(axis=1)
                #va = va.mean(axis=1)
                if not error_std:
                    plt.bar(
                            vx, 
                            mean*scales.get(vn,1), 
                            alpha = .45, 
                            #yerr = st,
                            label = target_names[t],
                            )
                else:
                    plt.bar(
                            vx, 
                            mean*scales.get(vn,1), 
                            alpha = .45, 
                            yerr = stdd*scales.get(vn,1),
                            label = target_names[t],
                            #bins=bins
                            )
            # labels and legend
            plt.xlabel(feature_names[f])
            #plt.ylabel()
            plt.legend()
    
            # saving figure if turned on
            if save:
                trgnam = ''
                for t in targets:
                    trgnam += '_'+t
                fignam = \
                fx+'__'+\
                trgnam+\
                '.png'
                dirutil.mkpath(figdir)
                plt.savefig(figdir+'/'+fignam)
    
            plt.show()
            
            return plt.gcf()
    
    def line(
            self,
            data=None,
            features=[],
            targets=[],
            feature_names=[],
            target_names=[],
            scales={},
            save = False,
            figdir = 'plot/lines',
            mode = 'exact', # 'exact', 'FFT' or callable,
            ):
        '''
        line(data)
        prints a standard line plots of the data
        
        Parameters:
            
            data       :   a dictionary containing formatted data of features
                            and targets to be considered
                            
            features   :   list of the input features to be considered for
                            the regression model.
                            
            targets    :   list of the variables to be dependenton the input 
                            features
                            
           . *_names   :   list of the names for features and targets 
                             respectively 
            
            error_std  :   if loading a standard deviation error bar
            
            save       :   if saving the output figures
            
            figdir     :   where to save the figures (relative or full path)
            
        '''
        # ask for data
        if type(data).__name__ == 'NoneType':
            data = self.data.copy()
            if type(data).__name__ == 'NoneType':
                print('Error: No data found!')
        
        # ask for targets and features
        if targets == []:
            targets = self.targets
        if features == []:
            features = self.features
        if target_names == []:
            target_names = targets
        if feature_names == []:
            feature_names = features
        
        # use numpy array convention
        features = np.array(features)
        targets = np.array(targets)
        
        def freqz(x):
            np.fft.fftfreq(x.shape[-1])
        
        md_flg = False
        # for each variable and each features plot a histogram
        for f,fx in enumerate(features):
            plt.figure()
            if type(data[fx]).__name__.split('.')[-1] == 'Series':
                vx = data[fx].unique()
            else:
                vx = data[fx].unique()
            N = vx.shape[-1]
            for t,vn in enumerate(targets):
                mean = np.zeros_like(vx)
                stdd = np.zeros_like(vx)
                # for each feature value
                for i,v in enumerate(vx):
                    mean[i] = data[vn][data[fx]==v].mean()
                    try:
                        stdd[i] = data[vn][data[fx]==v].std()
                    except:
                        stdd[i] = 0
                #vs = va.std(axis=1)
                #va = va.mean(axis=1)
                if type(mode).__name__ == 'str':
                    if mode == 'FFT':
                        mode = np.fft.fft
                        md_x = np.fft.fftfreq
                
                if type(mode).__name__ != 'str':
                    plt.plot(
                        md_x(vx.shape[-1])[:N//2],
                        mode(mean)[:N//2], 
                        alpha = .45, 
                        #yerr = vs,
                        label = target_names[t] 
                        )
                    md_flg = True
                else:
                    plt.plot(
                        vx, 
                        mean*float(scales.get(vn,1.0)), 
                        alpha = .45, 
                        #yerr = vs,
                        label = target_names[t] 
                        )
            # labels and legend
            if not md_flg:
                plt.xlabel(feature_names[f])
            else:
                plt.xlabel('1 / '+feature_names[f])
            #plt.ylabel()
            plt.legend()
            
            if save:
                trgnam = ''
                for t in targets:
                    trgnam += '_'+t
                fignam = \
                fx+'__'+\
                trgnam+\
                '.png'
                dirutil.mkpath(figdir)
                plt.savefig(figdir+'/'+fignam)
            
            plt.show()
    
    def boxplot(
            self,
            data=None,
            features=[],
            targets=[],
            feature_names=[],
            target_names=[],
            save = False,
            hue = None,
            figpath = 'plot/boxplots/',
            ):
        '''
                Box Plots
        
            This function creates "M" subplots for "N" targets given the m'th 
            feature.
            Also hue can divide the Boxplot into different categories from the 
            row with the name declared as hue.
            
            
        Parameters:
            
            data       :   a dictionary containing formatted data of features
                            and targets to be considered
                            
            features   :   list of the input features to be considered for
                            the regression model.
                            
            targets    :   list of the variables to be dependent on the input 
                            features
                            
           . *_names   :   list of the names for features and targets 
                             respectively - not used in this function
            
            save       :   if saving the output figures
            
            figpath    :   where to save the figures (relative or full path)
            
            hue        :   the axis the data is split into for different 
                             representations
        '''
        # ask for data
        if type(data).__name__ == 'NoneType':
            data = self.data.copy()
            if type(data).__name__ == 'NoneType':
                print('Error: No data found!')
        
        # ask for targets and features
        if targets == []:
            targets = self.targets
        if features == []:
            features = self.features
        if target_names == []:
            target_names = targets
        if feature_names == []:
            feature_names = features
        
        # use numpy array convention
        features = np.array(features)
        targets = np.array(targets)
        
        n = len(targets)
        for i,fx in enumerate(features):
            fig, axes = plt.subplots(nrows=n,ncols=1,sharex=False)
            fig.set_size_inches(8,3*n)
            #plt.figure(figsize=(8,3*n))
            for j,t in enumerate(targets):
                #plt.subplot(n,1,j+1, frameon=False)
                sns.boxplot( 
                        data=data, x=fx, y=t, hue=hue, 
                        ax=axes[j],
                        )
                axes[j].set( 
                        ylabel=target_names[j],
                        #sharex=axes[n-1],
                        )
                if j < n-1:
                    axes[j].set(xlabel='',)
                    #axes[j].xaxis.set_ticklabels([])
            axes[n-1].set( xlabel=feature_names[i],)
            # saving the figures if desired
            if save:
                figname = t+'_over_'+fx+'_hue_'+str(hue)+'.png'
                dirutil.mkpath(figpath)
                plt.savefig(figpath+figname)
            plt.show()
    
    def scatter(
            self,
            data=None,
            features=[],
            targets=[],
            feature_names=[],
            target_names=[],
            line_flg=True,
            scales={},
            save = False,
            figdir = 'plot/scatter',
            ):
        '''
        scatter(data)
        scatters a dictionary's variables over different features
        
        Parameters:
            
            data       :   a dictionary containing formatted data of features
                            and targets to be considered
                            
            features   :   list of the input features to be considered for
                            the regression model.
                            
            targets    :   list of the variables to be dependent on the input 
                            features
                            
           . *_names   :   list of the names for features and targets 
                             respectively 
        '''
        # ask for data
        if type(data).__name__ == 'NoneType':
            data = self.data.copy()
            if type(data).__name__ == 'NoneType':
                print('Error: No data found!')
        
        # ask for targets and features
        if targets == []:
            targets = self.targets
        if features == []:
            features = self.features
        if target_names == []:
            target_names = targets
        if feature_names == []:
            feature_names = features
        
        # use numpy array convention
        features = np.array(features)
        targets = np.array(targets)
        
        # for each features plot a scatter plot
        for i,fx in enumerate(features):
            # exclude object type
            if data[fx].dtype == 'object':
                continue
            for j,fy in enumerate(features):
                # exclude object type
                if data[fx].dtype == 'object':
                    continue
                
                # declare the feature names
                fxnm = feature_names[i]
                fynm = feature_names[j]
                if fx != fy:
                    # in case we have a feature with more than one dimension
                    # we, for now, just take the last dimension.
                    if len(data[fx].shape) > 1:
                        data[fx] = data[fx][:,-1]
                    if len(data[fy].shape) > 1:
                        data[fy] = data[fy][:,-1]
                    
                    # open a new figure with a subplot for the scatter plot
                    # and another one just for the two features we compared
                    figw = 5*(1 + 2*int(line_flg))
                    fig = plt.figure(figsize=(figw,5))
                    ax1 = fig.add_subplot(1,1+int(line_flg),1)
                    amax = np.argmax(data.filter(targets).values,axis=1)
                    for t,vn in enumerate(targets):
                        ax1.scatter(
                                data[fx][amax==t],
                                data[fy][amax==t],
                                label='%s %s' %('more',target_names[t]),
                                alpha=.75,
                                )
                    ax1.set_xlabel(
                            fxnm
                            )
                    ax1.set_ylabel(
                            fynm
                            )
                    plt.legend()
                    # line plot over all instances on the right
                    if line_flg:
                        ax2 = fig.add_subplot(122)
                        ax2.plot(
                                data[fx]*1.0*float(scales.get(fx,1.0)),
                                color='b', 
                                label=fxnm, 
                                alpha=.7
                                )
                        ax2.plot(
                                data[fy]*1.0*float(scales.get(fy,1.0)),
                                color='g', 
                                label=fynm, 
                                alpha=.7
                                )
                        ax2.set_xlabel('index')
                        #ax2.plot(data[vn],color='r', label=vn, alpha=.7)
                        #plt.ylabel()
                        plt.legend()
                    
                    if save:
                        trgnam = ''
                        for t in targets:
                            trgnam += '_'+t
                        fignam = \
                        fx+'_'+fy+'__'+\
                        trgnam+\
                        '.png'
                        dirutil.mkpath(figdir)
                        plt.savefig(figdir+'/'+fignam)
                    
                    plt.show()
    
    def ica_red(
            self,
            data=None,
            features=[],
            targets=[],
            target_names=[],
            var_frac=.97,
            num_latents='same',
            whiten=False,
            svd_solver='auto',
            randomized=False,
            add_ca=False,
            plot=False,
            savefig=False,
            figdir='plot/ica',
            #comm=MPI.COMM_WORLD, # in case of multi-processor parallelization
            ):
        '''
        Independent Component Analysis Reduction
        Does a ICA analysis on the data and tries to find independent of
        representative latents. The eigen-space shall become orthogonal.
        
        Please try to declare a number of latent variables 'num_latents', which
        is at least one more than the expected number of dependancies.
        
        For the case of uncertainty, leave the number of latent variables on
        'auto'. In this case, a normal ZCA analysis is done, which keeps the 
        original dimension of the input data, but reduces its correlations.
         close form of zero-mean and unit-variance, dependant on the fraction
        of variance kept.
        
        If there is a very hight occupation in the latent variable space, the 
        ICA reduction is very likely to be over-fitted!
        
        Parameters:
            data        :   a dictionary containing formatted data of features
                            and variables to be considered
            targets     :   names of the variables to be dependent on the input 
                            features
            features    :   names of the input features to be considered for
                            the regression model.
            var_frac    :   fraction of the variance to be kept.
        '''
        
        this_data = data.copy()
        
        # ask for data
        if type(this_data).__name__ == 'NoneType':
            this_data = self.data.copy()
            if type(this_data).__name__ == 'NoneType':
                print('Error: No data found!')
        
        # ask for targets and features
        if targets == []:
            targets = self.targets
        if features == []:
            features = self.features
        
        # load the independent compounent analysis module
        from sklearn.decomposition import FastICA as ICA
        
        D = len(features)
        
        if num_latents == 'same' and type(num_latents).__name__ == 'str':
            H = D
        else:
            H = num_latents
        
        my_N = this_data[features[0]].shape[0]
        
        feature_space = np.zeros([my_N,D],dtype=float)
        
        for f,feature in enumerate(features):
            feature_space[:,f] = this_data[feature]
        
        # initialize independent component analysis
        ica = ICA(
                n_components=H,
                whiten=whiten,
                )
        
        new_feature_space = ica.fit_transform(feature_space)
        
        # checking dimensionality - should be (my_N, num_latents)
        ca_N,ca_H = new_feature_space.shape
        if ca_N < ca_H:
            new_feature_space = new_feature_space.T
            pc_N,ca_H = new_feature_space.shape
        
        if ca_N > my_N:
            print(
                'Warning pc space (%i) seems to be larger than no. data points (%i)!'
                %(ca_N,my_N)
                )
            print('Discarding the last %i points.' %(int( ca_N - my_N )))
            new_feature_space = new_feature_space[:my_N]
        
        print('Shape of CA features:')
        print(new_feature_space.shape)
        
        # adding CAs to the data / visualization
        if add_ca or plot:
            new_features = np.zeros(H,dtype=str).tolist()
            for f in range(new_feature_space.shape[1]):
                this_data['ica_%i' %f] = new_feature_space[:,f]
                new_features[f] = 'ica_%i' %f
        
        if plot:
            #plt.figure(figsize=(10,10))
            self.scatter(
                    this_data,
                    new_features,
                    targets,
                    target_names=target_names,
                    save=savefig,
                    figdir=figdir
                    )
        
        # return the new data with the independent conponents
        # only if we want to just add them to the data structure
        if add_ca:
            return this_data,ica
        return new_feature_space # return the new feature space as numpy array
    
    def pca_red(
            self,
            data=None,
            features=[],
            targets=[],
            target_names=[],
            var_frac=.97,
            num_latents='same',
            whiten=False,
            svd_solver='auto',
            randomized=False,
            add_ca=False,
            plot=False,
            savefig=False,
            figdir='plot/pca',
            #comm=MPI.COMM_WORLD, # in case of multi-processor parallelization
            ):
        '''
        Pricipal Component Analysis Reduction
        Does a PCA analysis on the data and tries to find a sparse number of
        representative latents.
        
        Please try to declare a number of latent variables 'num_latents', which
        is at least one more than the expected number of dependancies.
        
        For the case of uncertainty, leave the number of latent variables on
        'auto'. In this case, a normal ZCA analysis is done, which keeps the 
        original dimension of the input data, but reduces its correlations.
         close form of zero-mean and unit-variance, dependant on the fraction
        of variance kept.
        
        If there is a very hight occupation in the latent variable space, the 
        PCA reduction is very likely to be over-fitted!
        
        Parameters:
            data        :   a dictionary containing formatted data of features
                            and variables to be considered
            targets     :   names of the variables to be dependent on the input 
                            features
            features    :   names of the input features to be considered for
                            the regression model.
            var_frac    :   fraction of the variance to be kept.
        '''
        
        this_data = data.copy()
        
        # ask for data
        if type(this_data).__name__ == 'NoneType':
            this_data = self.data.copy()
            if type(this_data).__name__ == 'NoneType':
                print('Error: No data found!')
        
        # ask for targets and features
        if targets == []:
            targets = self.targets
        if features == []:
            features = self.features
        
        # load the principal compounent analysis module
        from sklearn.decomposition import PCA
        
        D = len(features)
        
        if num_latents == 'same' and type(num_latents).__name__ == 'str':
            H = D
        else:
            H = num_latents
        
        my_N = this_data[features[1]].shape[0]
        
        feature_space = np.zeros([my_N,D],dtype=float)
        
        for f,feature in enumerate(features):
            feature_space[:,f] = this_data[feature]
        
        # initialize principal component analysis
        pca = PCA(
                n_components=H,
                whiten=whiten,
                )
        
        new_feature_space = pca.fit_transform(feature_space)
        
        # checking dimensionality - should be (my_N, num_latents)
        ca_N,ca_H = new_feature_space.shape
        if ca_N < ca_H:
            new_feature_space = new_feature_space.T
            pc_N,ca_H = new_feature_space.shape
        
        if ca_N > my_N:
            print(
                'Warning pc space (%i) seems to be larger than no. data points (%i)!'
                %(ca_N,my_N)
                )
            print('Discarding the last %i points.' %(int( ca_N - my_N )))
            new_feature_space = new_feature_space[:my_N]
        
        print('Shape of CA features:')
        print(new_feature_space.shape)
        
        # adding CAs to the data / visualization
        if add_ca or plot:
            new_features = np.zeros(H,dtype=str).tolist()
            for f in range(new_feature_space.shape[1]):
                this_data['pca_%i' %f] = new_feature_space[:,f]
                new_features[f] = 'pca_%i' %f
        
        if plot:
            #plt.figure(figsize=(10,10))
            self.scatter(
                    this_data,
                    new_features,
                    targets,
                    target_names=target_names,
                    save=savefig,
                    figdir=figdir
                    )
        
        # return the new data with the principal conponents
        # only if we want to just add them to the data structure
        if add_ca:
            return this_data,pca
        return new_feature_space # return the new feature space as numpy array
        
        
    def feature_regression(
            self,
            data = None,
            features = None,
            targets = None,
            N_train = 10000,
            N_test  =  5000,
            validation_fraction = 0.2,
            solver = 'sgd',
            max_iter = 150,
            #comm = MPI.COMM_WORLD, # just when doing some MPI parallelization
            ):
        '''
            Feature Regression
        
        This function just does a single layer regression of the data and tries
        to fit linear model (mainly Perceptron) to this data and which shall 
        then infer given output values of a given validation fraction from the 
        data.
        
        Here, we consider a given number of input features 'features' and try 
        to fit a linear regressor into a given number of output variables
        'targets'. Thus gradient descent is used. 
        
        Parameters:
            data        :   a dictionary containing formatted data of features
                            and variables to be considered
            targets   :   names of the variables to be dependenton the input 
                            features
            fae_names   :   names of the input features to be considered for
                            the regression model.
                Model parameters
            N_train     :   Number of training samples
            N_test      :   Number of test samples
            validation_fraction
                        :   fraction of the data always been used for 
                            validation of the moel to avoid over-fitting
            solver      :   which numerical solver will be used: 'sgd', 'lbfs'
            max_iter    :   maximum iterations
                            
            Here, Scikit-Learn's "MLP_Regressor" is being used for simplicity.
            If any other hidden layer sizes are defined in the additional 
            parameters, they are not considered here! The hidden layer size is 
            always empty. 
            If you need to use hidden layers, you need to use a (deep) learning
            library / module instead!
                            
            NOTE: This is a simple single layer linear 
        '''
        
        if type(data).__name__ == 'NoneType':
            data = self.data.copy()
            if type(data).__name__ == 'NoneType':
                print('Error: No data found!')
        
        from sklearn.neural_network.multilayer_perceptron import MLPRegressor
        
        Ntr,Nte = N_train,N_test
        
        mlp = MLPRegressor(
                hidden_layer_sizes = [], 
                validation_fraction = validation_fraction,
                solver = solver,
                max_iter = max_iter,
                )
        
        Y = dict2arr(data,targets)
        X = dict2arr(data,features)
        
        X_tr = X[:Ntr]
        Y_tr = Y[:Ntr]
        X_te = X[Ntr:Ntr+Nte]
        Y_te = Y[Ntr:Ntr+Nte]
        
        print('Training MLP regressor. . .')
        
        mlp.fit(X_tr,Y_tr)
        tr_scr = mlp.score(X_tr,Y_tr)
        te_scr = mlp.score(X_te,Y_te)
        
        print('done')
        print()
        print('Training score: %0.2f' %tr_scr)
        print('Tesr score: %0.2f' %te_scr)
        
        try: 
            tr_loss = mlp.loss_curve_ 
        except: 
            tr_loss = mlp._loss_grad_lbfgs
        
        
        
        plt.figure()
        plt.plot(tr_loss)
        plt.xlabel('Interations')
        plt.ylabel('Loss')
        plt.title('Gradient Dscent')
        
        self.mlp_reg = mlp
#%% unit tester
class TestStringMethods(unittest.TestCase):
    
    
    def test_pca_red(self):
        
        eda = EDA_MSC()
        
        g1 =  12 + np.random.randn(800)
        g2 = -12 + np.random.randn(800)
        
        my_df = {}
        my_df['g1'] = g1
        my_df['g2'] = g2
        my_df['t'] = np.zeros_like(my_df['g1'])
        
        new_df,pca = eda.pca_red(
            data=my_df,
            features=['g1','g2'],
            targets=['t'],
            var_frac=.97,
            num_latents='same',
            whiten=False,
            svd_solver='auto',
            randomized=False,
            add_ca=True,
            plot=False,
            savefig=False,
            )
        
        assert 'pca_0' in new_df
        assert 'pca_1' in new_df
        
        print('Gaussians variances:')
        print(new_df['g1'].var())
        print(new_df['g2'].var())
        
        print('PCA variances:')
        print(new_df['pca_0'].var())
        print(new_df['pca_1'].var())
        
        u1 =  12 + np.random.rand(800)
        u2 = -12 + np.random.rand(800)
        my_df['u1'] = u1
        my_df['u2'] = u2
        
        print()
        
        new_df,pca = eda.pca_red(
            data=my_df,
            features=['u1','u2','g1','g2'],
            targets=['t'],
            var_frac=.97,
            num_latents='same',
            whiten=True,
            svd_solver='auto',
            randomized=False,
            add_ca=True,
            plot=False,
            savefig=False,
            )
        
        assert 'pca_2' in new_df
        assert 'pca_3' in new_df
        
        print('Uniform distribution variances:')
        print(new_df['u1'].var())
        print(new_df['u2'].var())
        
        print('PCA variances:')
        print(new_df['pca_0'].var())
        print(new_df['pca_1'].var())
        print(new_df['pca_2'].var())
        print(new_df['pca_3'].var())
        
        new_df,pca = eda.pca_red(
            data=my_df,
            features=['u1','u2','g1','g2'],
            targets=['t'],
            var_frac=.97,
            num_latents=2,
            whiten=True,
            svd_solver='auto',
            randomized=False,
            add_ca=True,
            plot=False,
            savefig=False,
            )
        
        
        print('PCA with 2 components variances:')
        print(new_df['pca_0'].var())
        print(new_df['pca_1'].var())
        
        import pandas as pd
        corrMat = pd.DataFrame.from_dict(new_df)[['pca_0','pca_1']].corr()
        
        print("correlation matrix:")
        print(corrMat)
        
        for s in range(corrMat.shape[0]):
            corrMat[s,s] = 0
        
        assert np.abs(corrMat.values).all() < .5
        
    def test_ica_red(self):
        
        eda = EDA_MSC()
        
        g1 =  12 + np.random.randn(800)
        g2 = -12 + np.random.randn(800)
        
        my_df = {}
        my_df['g1'] = g1
        my_df['g2'] = g2
        my_df['t'] = np.zeros_like(my_df['g1'])
        
        new_df,ica = eda.ica_red(
            data=my_df,
            features=['g1','g2'],
            targets=['t'],
            var_frac=.97,
            num_latents='same',
            whiten=True,
            svd_solver='auto',
            randomized=False,
            add_ca=True,
            plot=False,
            savefig=False,
            )
        
        assert 'ica_0' in new_df
        assert 'ica_1' in new_df
        
        print('Gaussians variances:')
        print(new_df['g1'].var())
        print(new_df['g2'].var())
        
        print('ICA variances:')
        print(new_df['ica_0'].var())
        print(new_df['ica_1'].var())
        
        u1 =  12 + np.random.rand(800)
        u2 = -12 + np.random.rand(800)
        my_df['u1'] = u1
        my_df['u2'] = u2
        
        print()
        
        new_df,ica = eda.ica_red(
            data=my_df,
            features=['u1','u2','g1','g2'],
            targets=['t'],
            var_frac=.97,
            num_latents='same',
            whiten=True,
            svd_solver='auto',
            randomized=False,
            add_ca=True,
            plot=False,
            savefig=False,
            )
        
        assert 'ica_2' in new_df
        assert 'ica_3' in new_df
        
        print('Uniform distribution variances:')
        print(new_df['u1'].var())
        print(new_df['u2'].var())
        
        print('ICA variances:')
        print(new_df['ica_0'].var())
        print(new_df['ica_1'].var())
        print(new_df['ica_2'].var())
        print(new_df['ica_3'].var())
        
        for i in range(4):
            assert new_df['ica_%i' %i].var() < .1
        
        new_df,ica = eda.ica_red(
            data=my_df,
            features=['u1','u2','g1','g2'],
            targets=['t'],
            var_frac=.97,
            num_latents=2,
            whiten=True,
            svd_solver='auto',
            randomized=False,
            add_ca=True,
            plot=False,
            savefig=False,
            )
        
        
        print('ICA with 2 components variances:')
        print(new_df['ica_0'].var())
        print(new_df['ica_1'].var())
        
        for i in range(2):
            assert new_df['ica_%i' %i].var() < .1
        
        import pandas as pd
        corrMat = pd.DataFrame.from_dict(new_df)[['ica_0','ica_1']].corr()
        
        print("correlation matrix:")
        print(corrMat)
        
###############################################################################
#%%                             -- MAIN --                                  %%#
##%%-----------------------------------------------------------------------%%##
if ( __name__ == '__main__' ):
    
    import pandas as pd
    from pandas import datetime
    # example how to read data in chunks
    n_proc = 1
    rank = 0
    my_N = 17378 // n_proc
    
    path = '../data/hour.csv'
    my_data = pd.read_csv(path)[my_N*rank:my_N*(rank+1)]
    
    print(my_data['cnt'].shape)
    
    targets = ['cnt','casual','registered']
    
    # excluding instant, because it does not say anything about the data
    # exclude dteday, as long as we do not format this feature correctly
    # also, dteday does only contain a compressed version of the date
    excludes = ['dteday','instant']
    for target in targets:
        excludes.append(target)
    
    # loading all enties instead of 'cnt' and 'dteday'
    all_features = [x for x in my_data if not x in excludes]
    
    eda = EDA_MSC(my_data, targets=targets, features=all_features)
    
    #%% Histograms
    # defining extra features and targets we want to see in the histograms
    features = ['hr','weekday','mnth']
    feature_names = ['day hour','week day', 'month']
    targets = ['cnt','registered','casual']
    target_names = ['all rents', 'registered rents', 'casual rents']
    eda.histo(
            my_data, 
            features, targets, 
            feature_names=feature_names, 
            target_names=target_names,
            save=True,
            )
    features = ['holiday','workingday']
    feature_names = ['holiday','working day']
    targets = ['cnt','registered','casual']
    target_names = ['all rents', 'registered rents', 'casual rents']
    eda.histo(
            my_data, 
            features, targets, 
            feature_names=feature_names, 
            target_names=target_names,
            save=True,
            )
    #%% Scatter Plots
    features = ['atemp','hum','windspeed']
    feature_names = ['temperature','hum', 'wind speed']
    targets = ['cnt','registered','casual']
    target_names = ['all rents', 'registered rents', 'casual rents']
    eda.scatter(
            my_data, 
            features, targets, 
            feature_names=feature_names, 
            target_names=target_names,
            save=True,
            )
    #%% Time Series t/h
    features = ['instant']
    feature_names = ['t / h']
    targets = ['cnt','registered','casual']
    target_names = ['all rents', 'registered rents', 'casual rents']
    eda.line(
            my_data, 
            features, targets, 
            feature_names=feature_names, 
            target_names=target_names,
            save=True,
            )
    #%% Principal Components - All features
    
    # without whitening
    
    # convention:
        # H ... numbers of principal components
        # D ... numbers of original features
    
    D = np.array(all_features).shape[0] # is 12
    
    # H = D
    principals_D1 = eda.pca_red(
            my_data,
            plot=True,
            savefig=1,
            figdir='plot/pca_H12',
            )
    