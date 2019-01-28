# -*- coding: utf-8 -*-
"""
Random Forest Parameter Parse

this are standard paramter parsing functions

@author: Master
"""
#%% -- imports --
import numpy as np
import argparse
#%% -- argparsing function --
if ( __name__ == '__main__' ):
    
    params = locals()['params']
    
    # defaults
    Nva = params.get('Nva',379)
    Nte = params.get('Nte',2000)
    Ntr = params.get('Ntr',15000)
    
    ica_flg = params.get('ica_flg',False)
    pca_flg = params.get('pca_flg',True)
    H_whiten = params.get('H_whiten','same')
    ca_fit = params.get('ca_fit',True)
    ca_plt = params.get('ca_plt',False)
    
    l1p_flg = params.get('l1p_flg',True)
    nrm_flg = params.get('nrm_flg',True)
    hdf_pred = params.get('hdf_pred',True)
    H_whiten = params.get('ofile', 'data/hour_pred.hdf5')
    
    df_ifile = params.get('df_ifile','data/hour.csv')
    
    my_features = params.get('my_features',
            [
            'season',
            'hr',
            #'weekday',
            'workingday',
            #'windspeed',
            'temp',
            'hum',
            ])
    my__targets = params.get('my__targets',['registered','casual'])
    target_names = params.get('target_names',['registered rents', 'casual rents'])
    
    # rf params
    rf_fit = params.get('rf_fit',True)
    rf_Njob = params.get('rf_Njob',4)
    
    rf_Nest = params.get('rf_Nest',np.arange(10,450,40).tolist())
    rf_max_features = params.get('rf_max_features',['auto',3,4,5])
    rf_min_samples_leaf = params.get('rf_min_samples_leaf',np.arange(3,6).tolist())
    
    rf_model = params.get('rf_model','')
    ca_model = params.get('ca_model','')
    
    reg_plt = params.get('reg_plt',True)
    
    data_append = params.get('data_append',True)
    
    ofile = params.get('ofname','output/rf_train_bike.hdf5')
    
    params_read = params.get('params_read',False)
    
    #%% -- Arg Parser --
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
            '-p', dest='p', type=str,
            help='The parameter file path and name.', default='rf_train_pca_bike.py'
            )
    
    parser.add_argument(
            '--Nva', dest='Nva', type=int,
            help='no. validation sample', default=Nva
            )
    
    parser.add_argument(
            '--Nte', dest='Nte', type=int,
            help='no. test samples', default=Nte
            )
    
    parser.add_argument(
            '--Ntr', dest='Ntr', type=int,
            help='no. training samples', default=Ntr
            )
    
    # initial flags
    parser.add_argument(
            '--pca_flg', dest='pca_flg', type=bool,
            help='whether to use whitened PCA features out of the input features. '
            +'Does calculate H_whiten latent principle components out of the '
            +'given input features. NOTE: H_whiten can not be higher than '
            +'the number of input features! '
            +'The model will then be trained on the principle components! '
            +'Do not set both pca_flg and ica_flg! In this case only an ICA '
            +'will be done!', 
            default=pca_flg
            )
    
    parser.add_argument(
            '--ica_flg', dest='ica_flg', type=bool,
            help='whether to use whitened ICA features out of the input features. '
            +'Does calculate H_whiten latent independent components out of the '
            +'given input features. NOTE: H_whiten can not be higher than '
            +'the number of input features! '
            +'The model will then be trained on the independent components! '
            +'Do not set both ica_flg and pca_flg! In this case only an ICA '
            +'will be done!', 
            default=ica_flg
            )
    
    parser.add_argument(
            '--H_whiten', dest='H_whiten',
            help=
             'Number of components to be calcujlated for new whitened feature '
            +'space. You can set it to "same" to keep the feature dimension!', 
            default=H_whiten
            )
    
    parser.add_argument(
            '--ca_fit', dest='ca_fit', type=bool,
            help=
             'whether fitting the component analysis whitening or not. '
            +'NOTE: It has to be fitted, though this program would try to load '
            +'prefits from the data! CA file support is under construction!', 
            default=ca_fit
            )
    
    parser.add_argument(
            '--ca_plt', dest='ca_plt', type=bool,
            help=
             'whether plotting the component analysis whitening or not. '
            +'WARNING: This is only usable in graphical evironments! '
            +'Non-graphical OS is not supportet jet!', 
            default=ca_plt
            )
    
    # data specific params
    
    parser.add_argument(
            '--df_ifile', dest='df_ifile', type=str,
            help=
             'The name of the input file reading the dataset for the run.',
            default=df_ifile
            )
    
    parser.add_argument(
            '--df_ofile', dest='df_ofile', type=str,
            help=
             'The name of the output file storing the dataset from the run.',
            default='data/hour_pred.csv'
            )
    
    parser.add_argument(
            '--data_append', dest='data_append', type=bool,
            help=
             'If set, the data will target values will be set to the '
            +'predictions of the test set (Nte). The data then will be stored '
            +'in the output file determined by df_ofile.',
            default=data_append
            )
    
    parser.add_argument(
            '--l1p_flg', dest='l1p_flg', type=bool,
            help=
             'If set, the regression will be done for x = log(1+target_values)! '
            +'This is useful for exponentially distributed data!',
            default=l1p_flg
            )
    
    parser.add_argument(
            '--nrm_flg', dest='nrm_flg', type=bool,
            help=
             'If set, all data rows will be normed to max(row) = 1. '
            +'This is only useful for 1D data rows in this implementation! '
            +'It makes feature comparison easier. Even though it does NOT add '
            +'specific weightings to some features.',
            default=nrm_flg
            )
    
    parser.add_argument(
            '--reg_plt', dest='reg_plt', type=bool,
            help=
             'If set, the prediction is shown in a graph.',
            default=reg_plt
            )
    
    parser.add_argument(
            '--hdf_pred', dest='hdf_pred', type=bool,
            help=
             'If set, the prediction is stored into a HDF5 log file with a name '
            +'set by --lg_ofile.',
            default=hdf_pred
            )
    
    parser.add_argument(
            '--lg_ofile', dest='ofile', type=str,
            help=
             'The name of the HDF5 log file storing data from the run.',
            default=ofile
            )
    
    parser.add_argument(
            '--features', dest='my_features', nargs='+', type=str,
            help=
             'Names of input features to be considered for the prediction.',
            default=my_features
            )
    
    parser.add_argument(
            '--targets', dest='my__targets', nargs='+',
            help=
             'Names of output targets to be considered for the prediction.',
            default=my__targets
            )
    
    parser.add_argument(
            '--target_names', dest='target_names', nargs='+',
            help=
             'Labels for the output targets which are to be predicted.',
            default=target_names
            )
    
    # rf params
    parser.add_argument(
            '--rf_fit', dest='rf_fit', type=bool,
            help=
             'If set, the regression will be fit with a gridseach trying to '
            +'fint the best paramter set. it will be fit from --features to '
            +'--targets. ',
            default=rf_fit
            )
    
    parser.add_argument(
            '--rf_dpm', dest='rf_dmp', type=bool,
            help=
             'If saving the RF into a jlib file or not.',
            default=params.get('rf_dmp',True)
            )

    parser.add_argument(
            '--rf_ofile', dest='rf_ofile', type=str,
            help=
             'The name of the jlib file to store rf in. ',
            default=params.get('rf_ofile','model/rf_train_pca_bike.jlib')
            )
    
    parser.add_argument(
            '--rf_Njob', dest='rf_Njob', type=int,
            help='Number of processes in parallel for rf. '
            +'NOTE: This is a sklearn internal job parallelization!', 
            default=rf_Njob
            )
    
    parser.add_argument(
            '--rf_Nest_init', dest='rf_Nest_init', type=int,
            help='Number of estimators in parallel for rf.', 
            default=params.get('rf_Nest_init',200)
            )
    
    parser.add_argument(
            '--rf_Nest', dest='rf_Nest', nargs='+', type=int,
            help='list of estimator number which be tried in grid search.', 
            default=rf_Nest
            )
    
    parser.add_argument(
            '--rf_max_features', dest='rf_max_features', nargs='+',
            help='no. of parallel processes the rf is used/fit on.', 
            default=rf_max_features
            )
    
    parser.add_argument(
            '--rf_min_samples_leaf', dest='rf_min_samples_leaf', nargs='+',
            help='nlist of min samples per leave leaf.', 
            default=rf_min_samples_leaf
            )
    
    # external model parsing
    parser.add_argument(
            '--rf_model', dest='rf_model', type=str,
            help='File of rf model if loading from a jlib file.', 
            default=rf_model
            )
    
    parser.add_argument(
            '--ca_model', dest='ca_model', type=str,
            help='File of component analysis model if loading from a jlib file.', 
            default=ca_model
            )
    
    args = parser.parse_args()
    
    print(args)
    # parsing parser into dict
    new_params = vars(args)
    print(new_params)