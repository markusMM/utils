# -*- coding: utf-8 -*-
"""
       ~~~ Load Data wmSonar Example ~~~
       
       
@author: Markus Meister
"""
#%% -- imports --
import pandas as pd
import numpy as np
#%% -- data definition function for readin --
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

#%% -- init paramteres --

# file names    
data_table = 'C:/Users/Markus.Meister/Documents/v4 (Enid v3 + MarkusMayerLegende)/wmSonarEdit.xlsx'
leg_table  = 'C:/Users/Markus.Meister/Documents/v4 (Enid v3 + MarkusMayerLegende)/wmSonarLegende.xlsx'
new_table  = 'C:/Users/Markus.Meister/Documents/v4 (Enid v3 + MarkusMayerLegende)/wmSonar_merge.xlsx'

# flags
write_table = 0
to_hdf = False
auto_mapping = False
col_naming = False
merge_tables = True
merge_key = "uuid"

#%% -- specification paramters ( subject to be parsed! ) --

# "special" nameing convention here ( -> most likely to happen moreoften )
M2_names = list(range(1,14+1))
M2_names.append(99)
M2_names = np.array(M2_names,dtype=str)

# one very big dictionary with all data - and legend column formats for each sheet
sheets = {
	'A1':[],
	'dCat':[],
	'S3':sheet_num_sheet(
            'S3',
            30,
            leg_format='<sheet name>',
            ),
	'S5':sheet_num_sheet(
            'S5',
            15,
            leg_format='<sheet name>-<sheet name>r<column name>',
            ),
	'dM4':sheet_num_sheet(
            'M4',
            32,
            col_format='d<sheet name>r<column name>',
            ),
	'dM5':sheet_num_sheet(
            'M5',
            21,
            col_format='d<sheet name>r<column name>',
            ),
	'dM6':sheet_num_sheet(
            'dM6',
            7,
            leg_format='<sheet name>B'
            ),
	'M1':sheet_num_sheet(
            'M1',
            20,
            ),
	'M2':sheet_num_sheet(
            'M2',
            M2_names,
            ),
	'M3':sheet_num_sheet(
            'M3',
            24,
            ),
	'M4':sheet_num_sheet(
            'M4',
            32,
            ),
	'M5':sheet_num_sheet(
            'M5',
            21,
            ),
	'M6':sheet_num_sheet(
            'M6',
            7,
            ),
	'M7':sheet_num_sheet(
            'M7',
            ['dPipe',''],
            col_format='<column name><sheet name>'
            ),
	'M8':[],
	'M9':sheet_num_sheet(
            'M9',
            20,
            ),
	'M10':sheet_num_sheet(
            'M10', 
            M2_names, 
            leg_sheet='M2', 
            ),
	'M11-13':sheet_num_sheet(
            'M11-13',
            np.array(list(range(11,14)),dtype=str),
            col_format='M<column name>'
            ),
	}

names = {}

col_names = {}

#%% -- load and categorize data based on legends --

# forall sheets
for jk,sht in enumerate(sheets):
    
    dat = pd.read_excel(data_table,sht)
        
    exec('%s = dat' %names.get(sht,sht.replace('-','to')))

if merge_tables:
    
    for jk,d in enumerate(sheets):
        b = names.get(d,d.replace('-','to'))
        if jk < 1:
            
            merged = eval(b)
            
        else:
            
            df = eval(b)
            merged = merged.merge( df, on=merge_key )

#filtered = merged[merged['S2'].isin(range(18,30))]
            
