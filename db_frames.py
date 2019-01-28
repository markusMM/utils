# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:56:03 2018

@author: Markus.Meister
"""
#%% --- imports --
import numpy as np
import pandas as pd
#%% -- main module sonar --
class db_frames:
    
    extensions = {
            'xlsx'  :   pd.read_excel,
            'xls'   :   pd.read_excel,
            'csv'   :   pd.read_csv,
            'txt'   :   pd.read_table,
            'sas'   :   pd.read_sas,
            'html'  :   pd.read_html,
            'sql'   :   pd.read_sql,
            'gbq'   :   pd.read_gbq,
            'excel' :   pd.read_excel,
            'hdf5'  :   pd.read_hdf,
            }
    
    def __init__(
            self, 
            files   = ['data_norm.xlsx'],
            data    =       None,
            ):
        
        self.files = files
        self.db = data
    
    #@staticmethod
    def sheet_num_sheet(
        sheet_name='M1',                            # name of the sheet
        col_names=20,                               # names of each column
        col_format='<sheet name>r<column name>',    # format of the data column names
        leg_format='<sheet name>',                  # format of the legend column names
        leg_sheet = '',
        ):
    
        if leg_sheet == '':
            leg_sheet = sheet_name
        
        if type(col_names).__name__ == 'int':
            col_names = np.array(range(1,col_names+1),dtype=str)
        
        dict_col = {}
        for x in col_names:
            col_name = col_format.replace( '<sheet name>', sheet_name ).replace('<column name>',x)
            leg_name = leg_format.replace( '<sheet name>', leg_sheet  ).replace('<column name>',x)
            dict_col[col_name] = leg_name
        return dict_col
    
    def load_excel_with_sheets(
            self, 
            data_file = '', 
            leg_table = '', 
            sheets = {'A1':{}},
            write_table = False, 
            new_table = '', 
            names = {},
            ):
        leg = {}
        dfs = {}
        # forall sheets
        for sht in sheets:
            dat = pd.read_excel(data_file,sht)
            leg[sht] = []
            # A1 has no legend map jet
            if sht != 'A1':
                # forall legends of the sheet
                for j,l in enumerate(sheets[sht]):
                    
                    if      type(sheets[sht]).__name__ == 'dict':
                        cd = sheets[sht][l]
                    elif    type(sheets[sht]).__name__ == 'list':
                        try:
                            cd = sheets[sht][j]
                        except:
                            cd = sheets[sht]
                    else:
                        cd = l
                    if leg_table != '':
                        this_leg = pd.read_excel(
                        leg_table,
                        cd
                        )
                    else:
                        this_leg = {}
                    leg[sht].append( this_leg )
                    sh_mp = {}
                    for j,c in enumerate(this_leg['Code']):
                        sh_mp[c] = str(this_leg['Label'][j])
                    
                    #AttributeError: 'DataFrame' object has no attribute 'M2r1'
                    dat[l] = eval("dat.%s.map(sh_mp)" %l)
            
            exec('dfs[%s] = dat' %names.get(sht,sht))
        
        if write_table:
            
            if new_table == '':
                new_table = data_file.split('.xls')[0]+'_cat.xlsx'
            
            writer = pd.ExcelWriter(new_table, engine='xlsxwriter')
            
            for d in sheets:
                b = names.get(d,d)
                exec(b+'.to_excel(writer,sheet_name=b)')
                
            writer.save()
            writer.close()
        
        return dfs,leg
        
    def load_file(self,file,db_type):
        
        data = 'NONE'
        try:
            data = self.extensions[db_type](file)
            data = data.fillnan(0)
        except:
            print('Sorry "%s" not found!' %file)
        return data
    
    def load_data(self, 
                  files= None, 
                  cat_axis =   0, 
                  cat_keys = 'none', 
                  types  = 'extension', 
                  mrg_keys    = 'Subsid',
                  addtype  = 'merge',
                  ):
        '''
            load data   :   Loads data from files and concatenates them in a specific
                            dimension "cat_axis"
        params:
            
            files       :   list of strings containing the data file path/url
            cat_axis    :   dimension / axis how to concatenate the files together...
                                0   :   row vise
                                1   :   cloumn vise
                                2   :   depth vise
                                3 ...
            mrg_keys    :   single key or key list on which rows we want to merge
            cat_keys    :   'none', 'auto' or list of data source labels for each
                            file
                            if 'auto'   :   the labels will be the name of the files
                            if 'none'   :   no labels will be set
                            else        : same as 'none'
            types       :   'extension' or list of url/file database types
                            database tapes:
                                sql     :   sql database
                                gbq     :   google big query sql
                                excel   :   Excel binary
                                html    :   HTML file / table
                                txt     :   text file / table
                                csv     :   CSV  file / table
                                hdf5    :   HDF5 binary
            addtype     :   how the data should be added together
                            types:
                                'merge' :   merge from pandas
                                'cat'   :   concatenate from pandas
            
        '''
        if type(files).__name__ == 'NoneType':
            if type(self.files).__name__ == 'noneType':
                raise 'No file names defined!'
            else:
                files = self.files
        # list convention
        if not type(files).__name__ == 'list':
            files = [files]
        if not type(mrg_keys).__name__ == 'list':
            mrg_keys = [mrg_keys]
        # types parsing
        if types == 'extension':
            types       = list(map(lambda x : x.split('.')[-1], files))
#        else:
#            types       = list(map(lambda x : x.split('.')[-1], files))
        # parse catenate keys
        if cat_keys == 'auto':
            cat_keys    = list(map(lambda x : x.split('/')[-1].split('.')[-2], files))
        print(types)
        # list of pandas frames loaded from each file
        dbs = list(map(
                lambda d : self.load_file( files[d], types[d] ), range(len(files))
                ))
        #dbs = list(filter('NONE',dbs))
        # distinguish between the two data loading types
        if addtype == 'cat':
            if type(cat_keys).__name__ == 'list':
                self.db = pd.concat(dbs, axis=cat_axis, keys=cat_keys)
            else:
                self.db = pd.concat(dbs, axis=cat_axis)
        else:
#            # define "empty" data frame with the merge keys
#            key_dict = {} 
#            for k in mrg_keys:
#                key_dict[k] = np.array([])
            self.db = dbs[0]
            for d in range(1,len(dbs)):
                # merger self.db with db
                self.db = pd.merge(self.db, dbs[d], on=mrg_keys)
        
        return  self.db
        
        
        