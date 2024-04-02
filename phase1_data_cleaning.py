#phase 1 data curation notebook
"""
-performs steps outlined in Methods 'Data curation protocol'
-distinguishes discrete and continuous variables
-drops data columns, any phenotype with a ratio of >0.99 for the most common compared to the second most common value
-performs imputation, one-hot encoding
-Replaces continuous values greater than four standard deviations away from the mean in magnitude (z > 4) with the next largest magnitude value within four standard deviations of the mean (winsorization)
-retains collection events which included >95% of the number of families present at the time of baseline assessment
-links ABCD predefined categories to each phenotype (loadings_base_screen_6_1yr_category_updated.csv)

Key output is cleaned dataframe baseline_screen_6_1yr_z_4_cleaned.csv

"""
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
import os
from os import path
import json
import requests
import glob
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
#import PCA
from sklearn.decomposition import PCA
#import plotting
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from config import SAVE_DIRECTORY_PHASE1, ANALYSIS_NDA_OUTPUTS

#Load and prepare data dictionary, choices coding files (used to separate continuous and discrete columns)

#set save directory for everything
save_dir = SAVE_DIRECTORY_PHASE1

#load data dict, load choices coding csvs
#create abcd_data_dictionary, choices_coding_nda.3.0.csv, and choices_coding_nda.4.0.csv by cloning repository and following instructions at https://github.com/ABCD-STUDY/analysis-nda
#run 01_merge_data_dictionaries.R (set download_abcd_instruments <- T and download_abcd_data_dict <- T)
abcd_data_dictionary = pd.read_csv(f'{ANALYSIS_NDA_OUTPUTS}/ABCD_data_dictionary.csv')


choices_coding_nda3 = pd.read_csv(f'{ANALYSIS_NDA_OUTPUTS}/choices_coding_nda.3.0.csv', encoding='latin-1')

#run 04_create_choice_coding_nda_4.R
choices_coding_nda4 = pd.read_csv(f'{ANALYSIS_NDA_OUTPUTS}/choices_coding_nda.4.0.csv')
#add category values
#gather all study shortnames from NDA API

data_file = dict()
rel_4 = 'ABCD Release 4.0'
data_file[rel_4] = 'https://ndar.nih.gov/api/datadictionary/v2/datastructure?'
dd = requests.get(data_file[rel_4])

short_name_map_category_all = dict()

for d in dd.json():
    short_name_map_category_all[d['shortName']] = d['categories']
short_name_map_category_all

#shorten length of categories for each shortname to just the first one
for key, value in short_name_map_category_all.items():
    # size desired
    k = 1
    
    # using pop()
    # to truncate list
    n = len(value)
    for i in range(0, n - k ):
        value.pop()

#verify that all keys have only 1 value now
multiple_cat = 0
num_cat_assigned = []

for key, value in short_name_map_category_all.items():
    #print value
    #print(key, len([item for item in value if item]))
    num_cat_assigned.append(len([item for item in value if item]))
    if len([item for item in value if item]) > 1:
        multiple_cat += 1


with open(save_dir + 'short_name_map_category_all.json', 'w') as f:
    json.dump(short_name_map_category_all, f, indent=4)
    
short_name_map_category_all_df = pd.read_json(f'{save_dir}/short_name_map_category_all.json')
short_name_map_category_all_df = short_name_map_category_all_df.T
short_name_map_category_all_df.index.name = 'shortname'
short_name_map_category_all_df.columns = ['category']

#save shortname category mapping to csv
short_name_map_category_all_df.to_csv(f'{save_dir}/short_name_map_category_all.csv', header=True)
abcd_data_dict_categories_all = pd.merge(abcd_data_dictionary, short_name_map_category_all_df, how='left', left_on='NDA.Instrument', right_on='shortname')

#save abcd_data_dict with category information to csv
abcd_data_dict_categories_all.to_csv(f'{save_dir}/abcd_data_dict_categories_all.csv', header=True, index=False)

#Load data
#original data size
abcd_df1_orig = pd.read_csv(f'{save_dir}/abcd_stacked.csv', index_col=[0,1])
abcd_df1_orig
#unstack df
abcd_df1_orig_unstacked = abcd_df1_orig.unstack()
abcd_df1_orig_unstacked
#number of timing cols dropped
len(abcd_df1_orig.columns[abcd_df1_orig.columns.str.contains('_elap')])
#number of id cols dropped
len(abcd_df1_orig.columns[abcd_df1_orig.columns.str.endswith('_id')])
#loading unstacked data

abcd_df1 = abcd_df1_orig.copy()
abcd_df1

#Drop timing and ID columns
id_cols_df = abcd_df1.columns.str.endswith('_id')
#drop timing and ID columns here

abcd_df1 = abcd_df1.drop(abcd_df1.columns[id_cols_df], axis=1)
timing_cols_df = abcd_df1.columns.str.contains('_elap')

abcd_df1.iloc[:, abcd_df1.columns.str.contains('tfmri_nb_')].columns
abcd_df1 = abcd_df1.drop(abcd_df1.columns[timing_cols_df], axis=1)
abcd_df1

#drop all columns with only 1 value in them
unique_per_col = abcd_df1.nunique()
unique_per_col
#count of how many columns have each number of unique values
unique, counts = np.unique(unique_per_col, return_counts=True)

#print(np.asarray((unique, counts)).T)
non_columns = np.where(unique_per_col == 1)[0]
#drop all columns with only 1 value in them


abcd_df1 = abcd_df1.drop(abcd_df1.columns[non_columns], axis=1)
abcd_df1

#unstack df
abcd_df1_unstacked = abcd_df1.unstack()
abcd_df1_unstacked

#drop entirely nan columns
nans_per_col = abcd_df1_unstacked.isnull().sum(axis=0).tolist()
nans_per_col = np.array(nans_per_col)
non_nan_columns = np.where(nans_per_col != 11875)[0]
abcd_df1_unstacked_nonan = abcd_df1_unstacked.iloc[:, non_nan_columns]

#number of entirely nan columns dropped
len(np.where(nans_per_col == 11875)[0])
#number of constant columns dropped
len(non_columns)

#Prepare to separate into numeric and categorical using choices coding file (nda 3.0 and 4.0)
#merge two choices coding csvs
choices_coding = pd.merge(choices_coding_nda4, choices_coding_nda3, how='outer', on='name')
choices_coding
#there are some duplicate names from nda3/nda4 choices coding files
len(choices_coding.name.unique())
choices_coding[choices_coding.name == 'race_ethnicity']
choices_coding[choices_coding.name == 'site_id_l']
#add the categorical column names I changed in my df to this df
choices_coding.loc[len(choices_coding.index)] = ['sex','' ,'' ,'' ,'' ]
choices_coding.loc[len(choices_coding.index)] = ['site_id_l','' ,'' ,'' ,'' ]
choices_coding
#retrieve all column names from abcd_df1 to cross reference with choices coding

abcd_df1_cols = pd.Series(abcd_df1.columns)
abcd_df1_cols.name = "Column_names"
abcd_df1_cols
#if column is in choices coding file it is considered categorical

abcd_df1_cat_cols = abcd_df1_cols[abcd_df1_cols.isin(choices_coding.name)]
#drop interview_age as this is numeric/continuous
abcd_df1_cat_cols = abcd_df1_cat_cols.drop(2)
abcd_df1_cat_cols
abcd_df1_cat = abcd_df1[abcd_df1_cat_cols]
abcd_df1_cat
#partway to isolating all numeric variables, still has some categorical


abcd_df1_num_intermediate = abcd_df1.drop(abcd_df1_cat_cols, axis = 1)
abcd_df1_num_intermediate
abcd_df1_num_intermediate_obj = abcd_df1_num_intermediate.select_dtypes(include=['object'])
abcd_df1_num_intermediate = abcd_df1_num_intermediate.select_dtypes(exclude=['object'])
abcd_df1_num_intermediate
#retrieve all column names from intermediate df to merge with abcd data dict

abcd_df_cols_series = pd.Series(abcd_df1_num_intermediate.columns)
abcd_df_cols_series.name = "Column_names"
abcd_df_cols_series #includes partially filtered numeric columns based on not being type object and not being in choices coding files

#Load in ABCD Data Dict
#load data dictionary with categories per study/variable
abcd_data_dict = pd.read_csv(f'{save_dir}/abcd_data_dict_categories_all.csv')
abcd_data_dict = abcd_data_dict.drop_duplicates(subset=['Element.Name'], keep='first')
abcd_data_dict
abcd_data_dict['Element.Name']
#all the columns I still need to sort as numeric or categorical (after removing object type above)
abcd_mydata_dict = pd.merge(abcd_df_cols_series, abcd_data_dict, how='left', left_on='Column_names', right_on='Element.Name')
abcd_mydata_dict


#restrict to only my variables 
data_dict_df1_cols = pd.merge(abcd_df1_cols, abcd_data_dict, how='left', left_on='Column_names', right_on='Element.Name')
#find rows named 'sex', 'site_id_l','race_ethnicity' and replace with 'sex_official', 'site', and 'ethnicity' to match cleaned_df naming/pearson_corr df naming

data_dict_df1_cols.loc[data_dict_df1_cols.Column_names == 'sex','Column_names'] = 'sex_official'
data_dict_df1_cols.loc[data_dict_df1_cols.Column_names == 'site_id_l','Column_names'] = 'site'
data_dict_df1_cols.loc[data_dict_df1_cols.Column_names == 'race_ethnicity','Column_names'] = 'ethnicity'

#Separate into numeric and categorical dataframes via Heuristic
#find where valueRange col is NaN, so I can remove temporarily and find where there are semicolons
nan_rows = np.where(abcd_mydata_dict.valueRange.isna())[0]
#create version of mydata_dict w/o nans so its searchable by string
abcd_mydata_dict_no_nan = abcd_mydata_dict.drop(nan_rows, axis = 0)
abcd_mydata_dict_no_nan
#find all rows in mydata_dict valueRange col with semicolon, these are categorical
abcd_mydata_dict_semicolon = abcd_mydata_dict_no_nan[abcd_mydata_dict_no_nan['valueRange'].str.contains(';')]
abcd_mydata_dict_semicolon
#drop rows where valueRange has semicolons as these are categorical
abcd_mydata_dict_no_semi = abcd_mydata_dict.drop(abcd_mydata_dict_semicolon.index, axis = 0)
abcd_mydata_dict_no_semi
#identify rows where type = string as these are categorical
abcd_mydata_dict_string = abcd_mydata_dict_no_semi.query('type == "String"')
#drop rows where type = string as these are categorical
abcd_mydata_dict_no_str = abcd_mydata_dict_no_semi.drop(abcd_mydata_dict_string.index, axis = 0)
abcd_mydata_dict_no_str
#identify rows where notes starts with a digit equals (ex. '1=' or '1 =' or '1  =' or '999=' or '-1='), these are categorical

abcd_mydata_dict_notes = abcd_mydata_dict_no_str[abcd_mydata_dict_no_str['notes'].str.match('-?\d+[=]|-?\d+ [=]|-?\d+  [=]') == True]
#drop rows where notes starts with a digit equals (ex. '1=' or '1 ='), these are categorical

abcd_mydata_dict_no_notes = abcd_mydata_dict_no_str.drop(abcd_mydata_dict_notes.index, axis = 0)
abcd_mydata_dict_no_notes
#identify where type = float as these are numeric

abcd_mydata_dict_float = abcd_mydata_dict_no_notes.query('type == "Float"')

abcd_mydata_dict_float
#drop all float types so I can continue filtering 

abcd_mydata_dict_nofloat = abcd_mydata_dict_no_notes.drop(abcd_mydata_dict_float.index, axis = 0)
abcd_mydata_dict_nofloat

#export remaining 666 cols to look for pattern I can drop other categorical cols by
#abcd_mydata_dict_nofloat.to_csv('remaining1.csv', header=True, index=False, sep=",")
#look at remaining cols in abcd_df1_num_intermediate (starting point) to see unique values

abcd_df1_undetermined = abcd_df1_num_intermediate.loc[:, abcd_mydata_dict_nofloat.Column_names]
abcd_df1_undetermined
unique_per_col = abcd_df1_undetermined.nunique()
unique_per_col
#count of how many columns have each number of unique values
unique, counts = np.unique(unique_per_col, return_counts=True)

#print(np.asarray((unique, counts)).T)
#filter to view any columns with more than 10 unique vals, these will be considered numeric

num_cols = np.where(unique_per_col > 10)[0]
#keep only columns with >10 unique vals from previous version of df that already had floats removed

abcd_df1_numeric_non_float = abcd_df1_undetermined.loc[:,abcd_df1_undetermined.columns[num_cols]]
abcd_df1_numeric_non_float
#create first part of numeric df, using all columns that remiained after filtering efforts above
#not object type, not with ; in valueRange field of dict, not type string, not notes column begins
#with number equals (ex. 1=), remove float and keep more than 10 unique vals per column of remaining

abcd_df1_num_nonfloat = abcd_df1_num_intermediate.loc[:, abcd_df1_numeric_non_float.columns]
abcd_df1_num_nonfloat

#create second part of numeric df, using all columns that are of type float
#but are not object, not with ; in value range, not notes column begins
#with number equals (ex. 1=)

abcd_df1_num_float = abcd_df1_num_intermediate.loc[:, abcd_mydata_dict_float.Column_names]
abcd_df1_num_float
#create numeric out of concatenating the filtered df and the float df
abcd_df1_num = pd.concat([abcd_df1_num_nonfloat, abcd_df1_num_float], axis = 1)

#create categorical df out of all cols from abcd_df1_num_intermediate not part of abcd_df1_num above
abcd_df1_cat_2 = abcd_df1_num_intermediate.drop(abcd_df1_num_intermediate[abcd_df1_num.columns], axis =1) 
abcd_df1_cat_2
#create cat df out of choices_coding result and the heursitic filtering applied afterward to the numeric df

abcd_df1_cat_choices_heuristic = pd.concat([abcd_df1_cat, abcd_df1_cat_2], axis = 1)
abcd_df1_cat_choices_heuristic
#create final cat df out of above df with object type cols from abcd_df1_num_intermediate
#I ended up with 260 fewer categorical cols after combining classification methods

abcd_df1_cat = pd.concat([abcd_df1_cat_choices_heuristic, abcd_df1_num_intermediate_obj.select_dtypes(include=['object'])], axis = 1)
abcd_df1_cat

#Checking for nature of 111/222/333/444/555/666/777/888/999 columns

#construct subset data dict only containing columns in my df (pre heuristic so its all encompassing even if heuristic enhanced)
abcd_df1_dict = abcd_data_dict[abcd_data_dict['Element.Name'].isin(abcd_df1_cols)]
abcd_df1_dict

#Check 777
#filter based on presence of 777 in notes col rather than valueRange
#more cols
abcd_df1_dict_777_notes = abcd_df1_dict[abcd_df1_dict.notes.str.contains('777', na=False)]
abcd_df1_dict_777_notes

abcd_df1_dict_777_notes[abcd_df1_dict_777_notes.notes.str.lower().str.contains("refuse to answer|refused to answer|refuse|decline")]

#save the isip related cols that need to be made numerical
abcd_df1_isip = abcd_df1_cat.loc[:,['su_isip_1_calc', 'su_isip_1_calc_l', 'isip_1', 'isip_1_l']]
abcd_df1_isip
#concatenate onto existing abcd_num df
abcd_df1_num = pd.concat([abcd_df1_num, abcd_df1_isip], axis = 1)
#drop from cat df
abcd_df1_cat = abcd_df1_cat.drop(['su_isip_1_calc', 'su_isip_1_calc_l', 'isip_1', 'isip_1_l'], axis=1)
abcd_df1_cat

#Check 888 and 555 (always together)

abcd_choices_coding_888 = pd.concat([choices_coding[choices_coding.choices_y.str.contains('888', na=False)], choices_coding[choices_coding.choices_x.str.contains('888', na=False)]], axis = 0)
abcd_choices_coding_888
len(abcd_choices_coding_888.name.unique())

#no 555s in data dict valueRange field, will have to search in choices coding to find (both choices_x and choices_y cols corresponding to nda3 and nda4

abcd_choices_coding_555 = pd.concat([choices_coding[choices_coding.choices_y.str.contains('555', na=False)], choices_coding[choices_coding.choices_x.str.contains('555', na=False)]], axis = 0)
abcd_choices_coding_555
abcd_choices_coding_888.choices_y.unique()
abcd_choices_coding_555.choices_y.unique()

#Check 999
#filter based on presence of 999 in notes col rather than valueRange
abcd_df1_dict_999_notes = abcd_df1_dict[abcd_df1_dict.notes.str.contains('999', na=False)]
abcd_df1_dict_999_notes
abcd_df1_dict_999_notes[abcd_df1_dict_999_notes.notes.str.lower().str.contains("don't know|dont know")]
abcd_df1_dict_999_notes_outliers = abcd_df1_dict_999_notes[~abcd_df1_dict_999_notes.notes.str.lower().str.contains("don't know|dont know")]
abcd_df1_dict_999_notes_outliers


abcd_df1_dict_999_notes_missing = abcd_df1_dict_999_notes[abcd_df1_dict_999_notes.notes.str.lower().str.contains("999 = missing")]
abcd_df1_dict_999_notes_missing

#change the 6 medication dosage related columns to numeric
#replace 999 (missing) with nan so I can then use these cols as numeric
for cur_col in abcd_df1_dict_999_notes_missing['Element.Name']:
    abcd_df1_cat[cur_col].replace(999.0, np.nan, inplace=True)
#save the med dosage related cols that need to be made numerical
abcd_df1_med_dosage = abcd_df1_cat.loc[:,['medication3_dosage', 'medication4_dosage', 'medication5_dosage', 'medication6_dosage', 'medication7_dosage', 'medication8_dosage']]
abcd_df1_med_dosage
#concatenate onto existing abcd_num df
abcd_df1_num = pd.concat([abcd_df1_num, abcd_df1_med_dosage], axis = 1)
#drop from cat df
abcd_df1_cat = abcd_df1_cat.drop(['medication3_dosage', 'medication4_dosage', 'medication5_dosage', 'medication6_dosage', 'medication7_dosage', 'medication8_dosage'], axis=1)
abcd_df1_cat

#Unstacking dfs and dropping nan columns in cat and numeric dfs
#Unstacking numeric df

abcd_df_num = abcd_df1_num.unstack()
abcd_df_num

#Drop entirely Nan Columns Numeric

#are any columns entirely nan in numeric df?
nans_per_col = abcd_df_num.isnull().sum(axis=0).tolist()
#out of 20670 cols, 15412 are entirely nan (these are the certain events where a particular feature was not measured for all subjects)
#can be dropped
unique, counts = np.unique(nans_per_col, return_counts=True)

#print(np.asarray((unique, counts)).T)
nans_per_col = np.array(nans_per_col)
non_nan_columns = np.where(nans_per_col != 11875)[0]
#create new df with only those cols that were not nan from abcd_num_unstacked
abcd_df_num = abcd_df_num.iloc[:, non_nan_columns]

#This is where I must look at which subjects are in which numeric events
#create version of numeric df at this point with string type column names, to call upon later
abcd_df_num_nan_check = abcd_df_num.copy()
abcd_df_num_nan_check.columns = abcd_df_num_nan_check.columns.values
abcd_df_num_nan_check.columns = abcd_df_num_nan_check.columns.astype('string')

#Unstacking categorical df

#save sex column
abcd_sex_col = abcd_df1_cat.loc[(slice(None), 'baseline_year_1_arm_1'), 'sex']
#save site column
abcd_site_col = abcd_df1_cat.loc[(slice(None), 'baseline_year_1_arm_1'), 'site_id_l']
#save ethnicity column
abcd_eth_col = abcd_df1_cat.loc[(slice(None), 'baseline_year_1_arm_1'), 'race_ethnicity']

#drop eventname index level from sex col, site col, eth col
abcd_sex_col = abcd_sex_col.droplevel(level=1)
abcd_site_col = abcd_site_col.droplevel(level=1)
abcd_eth_col = abcd_eth_col.droplevel(level=1)

#remove columns before unstacking
abcd_df2_cat = abcd_df1_cat.drop(['sex', 'site_id_l','race_ethnicity'], axis=1)


abcd_df_cat = abcd_df2_cat.unstack()
abcd_df_cat
#re-insert sex site, and ethnicity columns into unstacked df
abcd_df_cat.insert(0,'sex_official', abcd_sex_col)
abcd_df_cat.insert(0,'site', abcd_site_col)
abcd_df_cat.insert(0,'ethnicity', abcd_eth_col)


abcd_df_cat

#Drop entirely Nan Columns Categorical

#are any columns entirely nan in cat df?
nans_per_col = abcd_df_cat.isnull().sum(axis=0).tolist()
#out of 87003 cols, 69124 are entirely nan (these are the certain events where a particular feature was not measured for all subjects)
#can be dropped
unique, counts = np.unique(nans_per_col, return_counts=True)

#print(np.asarray((unique, counts)).T)
nans_per_col = np.array(nans_per_col)
non_nan_columns = np.where(nans_per_col != 11875)[0]
#create new df with only those cols that were not nan from abcd_cat_unstacked
abcd_df_cat = abcd_df_cat.iloc[:, non_nan_columns]

abcd_df_cat

#Numeric Imputation
#Dropping numerical columns that are <80% populated

#are any columns entirely nan?
nans_per_col = abcd_df_num.isnull().sum(axis=0).tolist()
nans_per_col = np.array(nans_per_col)


#filter to keep only columns with <=20% nan
non_nan_columns = np.where(nans_per_col <= 2375)[0]
#construct new df with only the cols with <=20% nan
abcd_df_num = abcd_df_num.iloc[:, non_nan_columns]

abcd_df_num
#column names from abcd_df_num
abcd_df_num_cols = abcd_df_num.columns
#index names from abcd_df_num
abcd_df_num_ind = list(abcd_df_num.index.values)
abcd_df_num.shape
#imputation function per col 1

np.random.seed(0)
def my_impute(df, df_index):
    new_df = pd.DataFrame(index = df_index)
    for col in tqdm(df.columns):
        #print(df[col].dtypes)
        arr = pd.to_numeric(df[col])
        #print(arr.value_counts())
        arr = np.array(arr)
        #print(f'Replacing %i NaN values for {col}!' % np.sum(np.isnan(arr)))
        b_nan = np.isnan(arr)
        arr[b_nan] = np.random.choice(arr[~b_nan], np.sum(b_nan))
        new_df[col] = arr
    return new_df
abcd_df_num = my_impute(abcd_df_num, abcd_df_num.index)
abcd_df_num

abcd_df_std = abcd_df_num.std()
#filter to out all cols with std zero
std_zero_cols = np.where(abcd_df_std == 0)[0]
#construct new df with only the cols with 0 std
abcd_num_std_zero = abcd_df_num.iloc[:, std_zero_cols]
abcd_df_num = abcd_df_num.drop(abcd_num_std_zero, axis=1)

len(abcd_df_num.columns)


scaler = StandardScaler()
z_scored = scaler.fit_transform(abcd_df_num)
z_df = pd.DataFrame(z_scored, columns = abcd_df_num.columns, index = abcd_df_num.index)
abcd_df_num_z = z_df

abcd_df_mean = abcd_df_num_z.mean(axis=0)

#all cols have mean zero
abcd_df_mean[abcd_df_mean < -0.00000001]
abcd_df_mean[abcd_df_mean > 0.00000001]

abcd_df_num_z
#save numeric columns

abcd_df_num_z.to_csv(f'{save_dir}/numeric_df')


#create indices of outlier columns and histograms for each column at each threshold

indices_abs_4 = np.where((abcd_df_num_z.abs() > 4).any())[0]

indices_abs_100 = np.where((abcd_df_num_z.abs() > 100).any())[0]
abcd_df_num_z_4 = abcd_df_num_z.copy()
abcd_df_num_z_100 = abcd_df_num_z.copy()
winsorizing_count_4 = pd.DataFrame(0, index=np.arange(len(abcd_df_num_z)), columns=abcd_df_num_z.columns)
winsorizing_count_100 = pd.DataFrame(0, index=np.arange(len(abcd_df_num_z)), columns=abcd_df_num_z.columns)
indices_abs = [indices_abs_4, indices_abs_100]
abcd_df_num_z_list = [abcd_df_num_z_4, abcd_df_num_z_100]
winsorizing_array_list = [winsorizing_count_4, winsorizing_count_100]
thresh_list = [4, 100]

#winsorizing
for num_z_df, thresh, indices, winsorizing_array in zip(abcd_df_num_z_list,thresh_list, indices_abs, winsorizing_array_list): #loop through each threshold value and perform winsorizing on copy of z scored numeric df

    for i in indices: #i is column indices where outlier entries exist above thresh
        #at what row indices is the value over z-score threshold
        outlier_indices_pos = np.where((num_z_df.iloc[:, i]) > thresh)[0] #find row indices where value above pos thresh
        outlier_indices_neg = np.where((num_z_df.iloc[:, i]) < -thresh)[0] #find row indices where value below neg thresh
        
        if len(outlier_indices_pos > 0): #if there are positive outliers
        #convert entry at each outlier row index to next most positive value below thresh
        #sweep through next largest values in column until value <thresh is found
            if np.any((num_z_df.iloc[:,i].nlargest(11875).unique() < thresh) & (num_z_df.iloc[:,i].nlargest(11875).unique() > 0)):#if there exists an entry below threshold
                for j in range(1,len(num_z_df.iloc[:,i].nlargest(11875).unique())):
                    if num_z_df.iloc[:,i].nlargest(11875).unique()[j] < thresh: #check in z-scored df
                        num_z_df.iloc[outlier_indices_pos, i] = num_z_df.iloc[:,i].nlargest(11875).unique()[j] #apply in zscored df
                        winsorizing_array.iloc[outlier_indices_pos, i] = 1 #set flag to indicate subjects whose values have been winsorized in particular column
                        break
            else:
                num_z_df.iloc[outlier_indices_pos, i] = thresh #if no entry in column below threshold value, change column values to threshold value
        if len(outlier_indices_neg > 0): #if there are negative outliers
            #convert entry at each outlier row index to next most neg value below thresh
            #sweep through next smallest values in column until value < thresh is found
            if np.any((num_z_df.iloc[:,i].nsmallest(11875).unique() > -thresh) & (num_z_df.iloc[:,i].nlargest(11875).unique() < 0)): #if there exists an entry below threshold magnitude
                for j in range(1,len(num_z_df.iloc[:,i].nsmallest(11875).unique())): 
                    if num_z_df.iloc[:,i].nsmallest(11875).unique()[j] > -thresh: #check in z-scored df
                        num_z_df.iloc[outlier_indices_neg, i] = num_z_df.iloc[:,i].nsmallest(11875).unique()[j] #apply in zscored df
                        winsorizing_array.iloc[outlier_indices_neg, i] = 1 #set flag to indicate subjects whose values have been winsorized in particular column
                        break
            else:
                num_z_df.iloc[outlier_indices_neg, i] = -thresh #if no entry in column below threshold value, change column values to threshold value
#from here I go to df merge section to merge with different iterations of z scored/winsorized df        

#Converting 777 to 'np.nan' in categorical cols and dropping where categorical columns most common value makes up >=99% of responses
abcd_df_cat_copy = abcd_df_cat.copy()
abcd_df_cat_copy
#mask approach to find all columns that have at least one occurence of 777.0 using only columns from the notes column of the abcd_dict that had 777
#must ensure col from dict is present in cat_df otherwise index won't match, so ignore


cols_777_ = []

for cur_col in abcd_df1_dict_777_notes['Element.Name']: #loop through all column names from dict that had 777 in notes column

        if any(abcd_df_cat.columns.get_level_values(0).str.fullmatch(cur_col)): #check if the current column from the dict is in the cat_df
        
                for i in range(0, len(abcd_df_cat.loc[:, (cur_col, slice(None))].columns)): #for those columns that in cat_df, loop through the individual events for each column name
                        if any(
                                [777.0 in abcd_df_cat[cur_col, abcd_df_cat.loc[:, (cur_col, slice(None))].columns[i][1]].unique(), #if 777 present in any row of a particular column in question, add this col to a list of cols to be 'nan'/and or dropped
                                ]
                                ):
                                
                                cols_777_.append((cur_col, abcd_df_cat.loc[:, (cur_col, slice(None))].columns[i][1]))




#now replace 777 in all categorical columns with 777 with "nan" and check if most common value appears 99% of the time or more
#if so, drop that column

for cur_col in cols_777_:
    abcd_df_cat[cur_col].replace(777.0, np.nan, inplace=True)

    if abcd_df_cat[cur_col].nunique() <= 1: #check number of unique vals after replacing 777 with nan, if its 0 or 1 then drop the column
        abcd_df_cat = abcd_df_cat.drop(cur_col, axis=1)
        #print('dropped bc 1 val:', cur_col)

    elif abcd_df_cat[cur_col].nunique() > 1: #check number of unique vals after replacing 777 with nan, if its >1 then drop the column if >=99% of values are most common
        #print('checking:', cur_col)
        most_common_val_count = abcd_df_cat[cur_col].sort_values(ascending=False).value_counts().iloc[0] #number of times the most common value occurs
        total_val_count = abcd_df_cat[cur_col].sort_values(ascending=False).value_counts().sum() #total number of values in column
        percent_most_common_val = most_common_val_count/total_val_count #percentage of entries the most common entry accounts for
        if percent_most_common_val >= 0.99: #drop if >=99%
            #print('dropped bc >=99%:', cur_col)
            abcd_df_cat = abcd_df_cat.drop(cur_col, axis=1)

    
abcd_df_cat

#Convert 555 to "np.nan" and 888 columns to "0" and drop if over 99>
#must ensure col from dict is present in cat_df otherwise index won't match, so ignore


cols_888 = []

for cur_col in abcd_choices_coding_888['name']: #loop through all column names from choices that had 888/555 in choice column

        if any(abcd_df_cat.columns.get_level_values(0).str.fullmatch(cur_col)): #check if the current column from the choice_coding is in the cat_df
        
                for i in range(0, len(abcd_df_cat.loc[:, (cur_col, slice(None))].columns)): #for those columns that in cat_df, loop through the individual events for each column name
                        if any(
                                [888.0 in abcd_df_cat[cur_col, abcd_df_cat.loc[:, (cur_col, slice(None))].columns[i][1]].unique(), #if 888 (and 555 by extension) present in any row of a particular column in question, add this col to a list of cols to be 'nan'/and or dropped
                                 555.0 in abcd_df_cat[cur_col, abcd_df_cat.loc[:, (cur_col, slice(None))].columns[i][1]].unique()
                                ]
                                ):
                                
                                cols_888.append((cur_col, abcd_df_cat.loc[:, (cur_col, slice(None))].columns[i][1]))
#now replace 888 and 555 in all categorical columns with 888/555 with "nan" and check if most common value appears 99% of the time or more
#if so, drop that column

for cur_col in cols_888:
    abcd_df_cat[cur_col].replace(888.0, 0, inplace=True) #replace not adminstered due to gating with 0
    abcd_df_cat[cur_col].replace(555.0, np.nan, inplace=True)

    if abcd_df_cat[cur_col].nunique() <= 1: #check number of unique vals after replacing 888/555 with nan/0, if its 0 or 1 then drop the column
        abcd_df_cat = abcd_df_cat.drop(cur_col, axis=1)
        #print('dropped bc 1 val:', cur_col)

    elif abcd_df_cat[cur_col].nunique() > 1: #check number of unique vals after replacing 888/555 with nan, if its >1 then drop the column if >=99% of values are most common
        #print('checking:', cur_col)
        most_common_val_count = abcd_df_cat[cur_col].sort_values(ascending=False).value_counts().iloc[0] #number of times the most common value occurs
        total_val_count = abcd_df_cat[cur_col].sort_values(ascending=False).value_counts().sum() #total number of values in column
        percent_most_common_val = most_common_val_count/total_val_count #percentage of entries the most common entry accounts for
        if percent_most_common_val >= 0.99: #drop if >=99%
            #print('dropped bc >=99%:', cur_col)
            abcd_df_cat = abcd_df_cat.drop(cur_col, axis=1)
abcd_df_cat

#Convert and Drop 999 columns
#must ensure col from dict is present in cat_df otherwise index won't match, so ignore

abcd_df1_dict_999_notes = abcd_df1_dict[abcd_df1_dict.notes.str.contains('999', na=False)]
abcd_df1_dict_999_notes


cols_999 = []

for cur_col in abcd_df1_dict_999_notes['Element.Name']: #loop through all column names from dict that had 999 in dict column

        if any(abcd_df_cat.columns.get_level_values(0).str.fullmatch(cur_col)): #check if the current column from the dict is in the cat_df
        
                for i in range(0, len(abcd_df_cat.loc[:, (cur_col, slice(None))].columns)): #for those columns that in cat_df, loop through the individual events for each column name
                        if any(
                                [999.0 in abcd_df_cat[cur_col, abcd_df_cat.loc[:, (cur_col, slice(None))].columns[i][1]].unique(), #if 999 present in any row of a particular column in question, add this col to a list of cols to be 'nan'/and or dropped
                                ]
                                ):
                                
                                cols_999.append((cur_col, abcd_df_cat.loc[:, (cur_col, slice(None))].columns[i][1]))

#now replace 999 in all categorical columns with 999 with "nan" and check if most common value appears 99% of the time or more
#if so, drop that column

for cur_col in cols_999:
    abcd_df_cat[cur_col].replace(999.0, np.nan, inplace=True)

    if abcd_df_cat[cur_col].nunique() <= 1: #check number of unique vals after replacing 999 with nan, if its 0 or 1 then drop the column
        abcd_df_cat = abcd_df_cat.drop(cur_col, axis=1)
        #print('dropped bc 1 val:', cur_col)

    elif abcd_df_cat[cur_col].nunique() > 1: #check number of unique vals after replacing 999 with nan, if its >1 then drop the column if >=99% of values are most common
        #print('checking:', cur_col)
        most_common_val_count = abcd_df_cat[cur_col].sort_values(ascending=False).value_counts().iloc[0] #number of times the most common value occurs
        total_val_count = abcd_df_cat[cur_col].sort_values(ascending=False).value_counts().sum() #total number of values in column
        percent_most_common_val = most_common_val_count/total_val_count #percentage of entries the most common entry accounts for
        if percent_most_common_val >= 0.99: #drop if >=99%
            #print('dropped bc >=99%:', cur_col)
            abcd_df_cat = abcd_df_cat.drop(cur_col, axis=1)

abcd_df_cat

#Categorical Imputation, Dropping categorical columns with more than 50 unique values and >20% nan, and columns that are 99% const, dummy encoding
#This is where I must look at which subjects are present in which categorical events

#create version of categorical df at this point with string type column names, to call upon later
abcd_df_cat_nan_check = abcd_df_cat.copy()
abcd_df_cat_nan_check.columns = abcd_df_cat_nan_check.columns.values
abcd_df_cat_nan_check.columns = abcd_df_cat_nan_check.columns.astype('string')
#concatenate categorical and numeric nan check dfs
abcd_df_nan_check = pd.concat([abcd_df_num_nan_check, abcd_df_cat_nan_check], axis = 1)
#abcd_df_nan_check.to_csv(f'{save_dir}/abcd_df_nan_check.csv')

#sum nans per column
nans_per_col = abcd_df_cat.isnull().sum(axis=0).tolist()
#filter to keep only columns with <=50 categories
cat_thresh_columns = np.where(abcd_df_cat.nunique() <= 50)[0]
#construct new df with only the cols with <=50 categories
abcd_df_cat_thresh = abcd_df_cat.iloc[:, cat_thresh_columns]

#filter to keep only columns with >1 category nunique() doesn't consider 'nans' as unique vals. Thus, when nunique is 1 we truly have only one categorical response
varying_columns = np.where(abcd_df_cat_thresh.nunique() > 1)[0]
#construct new df with only the cols with >1 categories
abcd_df_cat1 = abcd_df_cat_thresh.iloc[:, varying_columns]

abcd_df_cat1
#Dropping categorical columns with more than 20% nan
#are any columns entirely nan?
nans_per_col = abcd_df_cat1.isnull().sum(axis=0).tolist()
nans_per_col = np.array(nans_per_col)
#filter to keep only columns with <=20% nan
non_nan_columns = np.where(nans_per_col <= 2375)[0]
non_nan_columns
#5214 columns remain after filtering any with >20% nan and more than 50 categories
len(non_nan_columns)
#construct new df with only the cols with <=20% nan
abcd_df_cat2 = abcd_df_cat1.iloc[:, non_nan_columns]

#change all cols to 'object' type so get_dummies can operate
abcd_df_cat2_obj = abcd_df_cat2.astype('object')

abcd_df_cat2_obj_copy = abcd_df_cat2_obj.copy()

#now check if most common value of each col appears 99% of the time or more
#if so, drop that column

for cur_col in tqdm(abcd_df_cat2_obj):
    

    if abcd_df_cat2_obj[cur_col].nunique() == 1: #check number of unique vals, if its all 1 then drop the column
        abcd_df_cat2_obj = abcd_df_cat2_obj.drop(cur_col, axis=1)
        #print('dropped bc 1 val:', cur_col)

    elif abcd_df_cat2_obj[cur_col].nunique() > 1: #check number of unique vals, if its >1 then drop the column if >=99% of values are most common value
        #print('checking:', cur_col)
        most_common_val_count = abcd_df_cat2_obj[cur_col].sort_values(ascending=False).value_counts().iloc[0] #number of times the most common value occurs
        total_val_count = abcd_df_cat2_obj[cur_col].sort_values(ascending=False).value_counts().sum() #total number of values in column
        percent_most_common_val = most_common_val_count/total_val_count #percentage of entries the most common entry accounts for
        if percent_most_common_val >= 0.99: #drop if >=99%
            #print('dropped bc >=99%:', cur_col)
            abcd_df_cat2_obj = abcd_df_cat2_obj.drop(cur_col, axis=1)

abcd_df_cat2_obj_post_thresh = abcd_df_cat2_obj.copy()

#must impute categorical data here to remove nans before dummy encoding

np.random.seed(0)
def my_cat_impute(df, df_index):
    new_df = pd.DataFrame(index = df_index)
    for col in tqdm(df.columns):
        #print(df[col].dtypes)
        arr = df[col]
        #arr = pd.to_numeric(df[col])
        #print(arr.value_counts())
        arr = np.array(arr)
        #print(f'Replacing %i NaN values for {col}!' % np.sum(np.isnull(arr)))
        b_nan = pd.isnull(arr)
        arr[b_nan] = np.random.choice(arr[~b_nan], np.sum(b_nan))
        new_df[col] = arr
    return new_df
abcd_df_cat2_obj_post_impute = my_cat_impute(abcd_df_cat2_obj_post_thresh, abcd_df_cat2_obj_post_thresh.index)
abcd_df_cat2_obj_post_impute
#dummy-code all categorical data
abcd_df_cat_dummy = pd.get_dummies(abcd_df_cat2_obj_post_impute, dummy_na=False)

#dummy-coding creates a large amount of very sparse columns
abcd_df_cat_dummy

#Recombine numeric and categorical data
#merge back categorical and numerical variables

abcd_df_cleaned_4 = abcd_df_num_z_4.join(abcd_df_cat_dummy) #winsorized dataframe
abcd_df_cleaned =abcd_df_num_z.join(abcd_df_cat_dummy) #non-winsorized dataframe
#save winsorized dataframe before dropping columns that are >=99% constant

abcd_df_cleaned_4.to_csv(f'{save_dir}/abcd_df_pre_const_drop_4.csv')
abcd_df_cleaned.to_csv(f'{save_dir}/abcd_df_pre_const_drop_non_winsorized.csv')
abcd_df_cleaned_list = [abcd_df_cleaned_4, abcd_df_cleaned]

abcd_df_cleaned_dropped_list = []

for cleaned_df in abcd_df_cleaned_list: #loop through each version of cleaned df to perform99%const drops
    
    constant_cols_count = 0
    nearly_constant_cols_count = 0

    #now check if most common value of each col appears 99% of the time or more
    #if so, drop that column

    for cur_col in tqdm(cleaned_df):
        

        if cleaned_df[cur_col].nunique() == 1: #check number of unique vals, if its all 1 then drop the column
            cleaned_df = cleaned_df.drop(cur_col, axis=1)
            #print('dropped bc 1 val:', cur_col)
            #print("constant: ", cur_col)
            constant_cols_count += 1

        elif cleaned_df[cur_col].nunique() > 1: #check number of unique vals, if its >1 then drop the column if >=99% of values are most common value
            #print('checking:', cur_col)
            most_common_val_count = cleaned_df[cur_col].sort_values(ascending=False).value_counts().iloc[0] #number of times the most common value occurs
            total_val_count = cleaned_df[cur_col].sort_values(ascending=False).value_counts().sum() #total number of values in column
            percent_most_common_val = most_common_val_count/total_val_count #percentage of entries the most common entry accounts for
            if percent_most_common_val >= 0.99: #drop if >=99%
                #print('dropped bc >=99%:', cur_col)
                cleaned_df = cleaned_df.drop(cur_col, axis=1)
                #print("99 percent constant: ", cur_col)
                nearly_constant_cols_count += 1
        
    #save df to list after dropping columns
    abcd_df_cleaned_dropped_list.append(cleaned_df)
    
    #print("constant columns", constant_cols_count)
    #print("99 percent constant columns", nearly_constant_cols_count)
len(abcd_df_cleaned_dropped_list)
#save winsorized dataframe post dropping columns that are >=99% constant

abcd_df_cleaned_dropped_list[0].to_csv(f'{save_dir}/abcd_df_post_drop_4.csv')
abcd_df_cleaned_dropped_list[1].to_csv(f'{save_dir}/abcd_df_post_drop_non_winsorized.csv')
abcd_df_cleaned_dropped_non_winsorized = abcd_df_cleaned_dropped_list[1]

#Link descriptions to column shortnames
#create descriptor column to pair with abcd_df_cleaned column names

#create pandas empty series, fill array with categories matching column name of abcd_df_cleaned
#save for insertion in printing of PCA loadings
abcd_df_cleaned_col_descript = []

for i in range(0, len(abcd_df_cleaned_dropped_non_winsorized.columns)):
    
    abcd_df_cleaned_col_descript.append(data_dict_df1_cols["Element.Description"][data_dict_df1_cols.Column_names == abcd_df_cleaned_dropped_non_winsorized.columns.astype('string')[i].split(',')[0][1:].strip("''")].values[0])

#merge with abcd_df_cleaned column names so I can associate with any subsequent df (like just baseline, just 1 year, 4 event, etc.)
col_desc_map_df = pd.DataFrame(data=abcd_df_cleaned_col_descript, index = abcd_df_cleaned_dropped_non_winsorized.columns.astype('string'), columns = ['description'])

#save col_desv_map_df to csv
col_desc_map_df.to_csv(f'{save_dir}/abcd_cleaned_col_desc_map.csv')


baseline_df_dict={}
screener_df_dict={}
month6_df_dict={}
year1_df_dict={}
baseline_screen_6_1yr_df_dict={}

for cleaned_df, i in zip(abcd_df_cleaned_dropped_list, range(1,len(abcd_df_cleaned_dropped_list)+1)): #loop through cleaned dfs and create event specific dataframe
    #add back in sex, ethnicity, site columns
    abcd_df_cleaned_sex = cleaned_df.loc[:, cleaned_df.columns.astype('string').str.contains('sex_official')]
    abcd_df_cleaned_eth = cleaned_df.loc[:, cleaned_df.columns.astype('string').str.contains('ethnicity')]
    abcd_df_cleaned_site = cleaned_df.loc[:, cleaned_df.columns.astype('string').str.contains('site')]

    abcd_df_cleaned_sex_site_eth = pd.concat([abcd_df_cleaned_sex, abcd_df_cleaned_eth, abcd_df_cleaned_site], axis = 1)

    df_baseline_screen_6_1yr=cleaned_df.loc[:, cleaned_df.columns.astype('string').str.contains('baseline_year_1_arm_1|6_month_follow_up_arm_1|1_year_follow_up_y_arm_1|screener_arm_1')]
    baseline_screen_6_1yr_df_dict[i] = pd.concat([abcd_df_cleaned_sex_site_eth, df_baseline_screen_6_1yr], axis = 1)
    

#save with name as "cleaned" for use in CVAE

baseline_screen_6_1yr_df_dict[1].to_csv(f'{save_dir}/baseline_screen_6_1yr_z_4_cleaned.csv')


#PCA (only events with ~all subjects) using |z|=4

#make instance of pca for baseline + screener + 6 month + 1 year events
pca_base_screen_6_1yr = PCA(n_components=5, random_state= 0)

#fit model with training data and apply dim reduction on training data
pc_base_screen_6_1yr = pca_base_screen_6_1yr.fit_transform(baseline_screen_6_1yr_df_dict[1])
#link description of phenotypes to loadings

loadings_base_screen_6_1yr = pd.DataFrame(pca_base_screen_6_1yr.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], index= baseline_screen_6_1yr_df_dict[1].columns.astype('string'))
loadings_base_screen_6_1yr = pd.merge(loadings_base_screen_6_1yr, col_desc_map_df.description, how='left', left_index=True, right_index=True)

#save
loadings_base_screen_6_1yr.to_csv(f'{save_dir}/loadings_base_screen_6_1yr.csv')

#load data dict with categories
abcd_data_dict_categories_all = pd.read_csv(f'{save_dir}/abcd_data_dict_categories_all.csv', lineterminator='\n')
abcd_data_dict_categories_all
#change certain variable names to match phenotypes in the loadings df

#find rows named 'sex', 'site_id_l','race_ethnicity' and replace with 'sex_official', 'site', and 'ethnicity' to match cleaned_df naming/pearson_corr df naming

abcd_data_dict_categories_all.loc[abcd_data_dict_categories_all['Element.Name'] == 'sex','Element.Name'] = 'sex_official'
abcd_data_dict_categories_all.loc[abcd_data_dict_categories_all['Element.Name'] == 'site_id_l','Element.Name'] = 'site'
abcd_data_dict_categories_all.loc[abcd_data_dict_categories_all['Element.Name'] == 'race_ethnicity','Element.Name'] = 'ethnicity'
loadings_base_screen_6_1yr = pd.read_csv(f'{save_dir}/loadings_base_screen_6_1yr.csv', index_col=0)
#add category column into loadings_base_screen_6_1yr

#create pandas empty series, fill array with categories matching index of loadings_base_screen_6_1yr_category, append to loadings_base_screen_6_1yr_category
loadings_base_screen_6_1yr_index_categories = []

for i in range(0, len(loadings_base_screen_6_1yr.index)):
    #if the column exists i my df, append the category to the list
    if np.any(abcd_data_dict_categories_all['Element.Name'] == loadings_base_screen_6_1yr.index[i].split(',')[0][1:].strip("''")):
        loadings_base_screen_6_1yr_index_categories.append(abcd_data_dict_categories_all["category"][abcd_data_dict_categories_all['Element.Name'] == loadings_base_screen_6_1yr.index[i].split(',')[0][1:].strip("''")].values[0])

loadings_base_screen_6_1yr.insert(len(loadings_base_screen_6_1yr.columns), "category", loadings_base_screen_6_1yr_index_categories) #add category column
loadings_base_screen_6_1yr_category = loadings_base_screen_6_1yr.copy()
#change site variable to 'Summary' category
loadings_base_screen_6_1yr_category.category[loadings_base_screen_6_1yr_category['category'] == 'Omics'] = 'Demographics'
#change sex variable to 'Demographics category
loadings_base_screen_6_1yr_category.iloc[0:2,-1] = 'Demographics'

#update interview age category to Demographics
loadings_base_screen_6_1yr_category.loc["('interview_age', 'baseline_year_1_arm_1')", 'category'] = 'Demographics'
loadings_base_screen_6_1yr_category.loc["('interview_age', 'screener_arm_1')", 'category'] = 'Demographics'
loadings_base_screen_6_1yr_category.loc["('interview_age', '6_month_follow_up_arm_1')", 'category'] = 'Demographics'
loadings_base_screen_6_1yr_category.loc["('interview_age', '1_year_follow_up_y_arm_1')", 'category'] = 'Demographics'
#add shortname column into loadings_base_screen_6_1yr_category

#create pandas empty series, fill array with shortname matching index of loadings_base_screen_6_1yr_category, append to loadings_base_screen_6_1yr_category
loadings_base_screen_6_1yr_category_index_shortname = []

for i in range(0, len(loadings_base_screen_6_1yr_category.index)):
    #if the column exists i my df, append the category to the list
    if np.any(abcd_data_dict_categories_all['Element.Name'] == loadings_base_screen_6_1yr_category.index[i].split(',')[0][1:].strip("''")):
        loadings_base_screen_6_1yr_category_index_shortname.append(abcd_data_dict_categories_all["NDA.Instrument"][abcd_data_dict_categories_all['Element.Name'] == loadings_base_screen_6_1yr_category.index[i].split(',')[0][1:].strip("''")].values[0])

loadings_base_screen_6_1yr_category.insert(len(loadings_base_screen_6_1yr_category.columns), "nda_instrument", loadings_base_screen_6_1yr_category_index_shortname) #add studyname column
#save
loadings_base_screen_6_1yr_category.to_csv(f'{save_dir}/loadings_base_screen_6_1yr_category_updated.csv')
loadings_base_screen_6_1yr_category