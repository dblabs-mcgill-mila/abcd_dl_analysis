#phase 1 data collection notebook
"""
this script merges all individual study files from ABCD release 4.0 non-imaging on the combination of 'subjectkey' and 'eventname' columns to create merged_df.csv
	-cleans up duplicate columns created (abcd_nonimaging.csv)
	-creates version of dataframe with MultiIndex using both 'subjectkey' and 'eventname' as indices (abcd_stacked.csv) used in subsequent analyses

"""
import pandas as pd
import glob
import numpy as np
import os
from config import SAVE_DIRECTORY_PHASE1, ABCD_DATA_PATH

#directory to save output of merge
save_dir = SAVE_DIRECTORY_PHASE1
os.makedirs(save_dir, exist_ok = True)
load_dir = ABCD_DATA_PATH

#retrieve all .txt files of downloaded nda4 non-imaging data release
allfiles = glob.glob(load_dir)
allfiles
#find number of files
len(allfiles)

"""
Verify that subjectkey is key column for merging
print shape of each file and whether or not it has a subject key column
"""

#identify files that do not have subjectkey column

for i, file in enumerate(allfiles): #finding files that do not have "subjectkey" column
    df = pd.read_csv(file, delimiter='\t', on_bad_lines='skip', low_memory=False)
    if 'subjectkey' not in df.columns: print(file.split("/")[-1], i)
    
allfiles.pop(108) #remove the files that do not have subjectkey field
allfiles.pop(59) #remove the file that does not have subjectkey field
allfiles.pop(37)
#eventname field is another key column to be merged on since we have sometimes have multiple events for each subject
#find files without eventname column

for i, file in enumerate(allfiles): #finding files that do not have "eventname" column
    df = pd.read_csv(file, delimiter='\t', on_bad_lines='skip', low_memory=False)
    if 'eventname' not in df.columns: print(file.split("/")[-1], i)
useful_files = allfiles.copy() 
#create a copy of allfiles to run merge on all useful files, that have an "eventname" column
print(useful_files.pop(117)) #remove files that do not have an "eventname" column, will operate on this subset from here on
print(useful_files.pop(80))
print(useful_files.pop(63))
print(useful_files.pop(58))
print(useful_files.pop(27))
#instantiate the df to merge all .txt files into

merged_df = pd.read_csv(useful_files[0], delimiter='\t', low_memory = False)
print ("Dataframe created with " + useful_files[0].split("/")[-1], merged_df.shape)


#iteratively merge each file from useful_files list into df. Outer merge on both subjectkey and eventname

for filename in useful_files[1:]:
    df = pd.read_csv(filename, delimiter='\t', low_memory = False)
    merged_df = pd.merge(merged_df, df, on = ['subjectkey', 'eventname'], how = 'outer')
    print ("Merged with " + filename.split("/")[-1], merged_df.shape)
merged_df.to_csv(f'{save_dir}/merged_df.csv', index = False)



#clean up duplicated columns from merge
#abcd_merged = merged_df.copy()
abcd_merged = pd.read_csv(f'{save_dir}/merged_df.csv')
#to load the merged df
#keep only the first occurence of a particular subjectkey/eventname combination, drop any subsequent occurences of that same subject for the same event

abcd_merged_unique_subjects = abcd_merged.drop_duplicates(subset=['subjectkey', 'eventname'])
#Writing regex attempt to isolate repeated columns due to merge 
#list all columns that end with _x. or _y. followed by 0 or more integers
import re

r1 = re.compile(r'\_x\.[0-9]*$|\_y\.[0-9]*$')
#loop through all column names in dataframe to find those that match, if match add name to series
#each one that is repeated will show up 152 times (initial occurence is not being matched)
duplicate_columns1 = []

for i in range(0, len(abcd_merged_unique_subjects.columns)):
    if r1.search(abcd_merged_unique_subjects.columns[i]):
        #print (abcd_merged_unique_subjects.columns[i])
        duplicate_columns1.append(abcd_merged_unique_subjects.columns[i])
duplicate_columns1 = np.array(duplicate_columns1)

#code to view base column name of duplicated columns
#regex to print string up to point I found previously

from ast import NotIn


shortened_duplicates1 = []

for i in range(0, len(duplicate_columns1)):

    shortened1 = re.sub(r'\_x\.[0-9]*$|\_y\.[0-9]*$', "", duplicate_columns1[i])
    if shortened1 not in shortened_duplicates1:
        shortened_duplicates1.append(shortened1)
shortened_duplicates1 = np.array(shortened_duplicates1)
len(shortened_duplicates1)
#all columns that match my regex expression, there are 7 as expected this time
#these are the columns for which I need to prune duplicates in my original dataset

shortened_duplicates1

#collection_id
#print all column names that start with 'collection_id' (how many times was it duplicated in merge)

filter_col_collection_id = [col for col in abcd_merged_unique_subjects if col.startswith('collection_id')]
#retreive indices of all columns that were duplicated

collection_ids =[]

for i in range(0,len(filter_col_collection_id)):

    #print(abcd_merged_unique_subjects.columns.get_loc(filter_col_collection_id[i]))
    collection_ids.append(abcd_merged_unique_subjects.columns.get_loc(filter_col_collection_id[i]))
#create sliced version of df with only the columns corresponding to "collection_id"

collection_id_cols = abcd_merged_unique_subjects.iloc[:, collection_ids]
#check unique values across all columns for each row, not including nan's

nunique_vals_per_row = []
for idx in range(0, len(collection_id_cols.index)):
    
    nunique_vals_per_row.append(len(collection_id_cols.iloc[idx,:][collection_id_cols.iloc[idx,:].notna()].unique()))

#verify that each row only has one unique value
#print('number of unique values per row: ', np.unique(nunique_vals_per_row))
#create new column using unique value across rows in duplicate columns

collection_id = []

for i in range(0, len(abcd_merged_unique_subjects)):

    for j in range(0, len(collection_id_cols.columns)):
        if pd.notna(collection_id_cols.iloc[i,j]):
            collection_id.append(collection_id_cols.iloc[i,j])
            break

#sex
#print all column names that start with 'sex' (how many times was it duplicated in merge)

filter_col_sex = [col for col in abcd_merged_unique_subjects if col.startswith('sex')]
#retreive indices of all columns that were duplicated

sexs =[]

for i in range(0,len(filter_col_sex)):

    #print(abcd_merged_unique_subjects.columns.get_loc(filter_col_sex[i]))
    sexs.append(abcd_merged_unique_subjects.columns.get_loc(filter_col_sex[i]))
#create sliced version of df with only the columns corresponding to "sex"

sex_cols = abcd_merged_unique_subjects.iloc[:, sexs]
#check unique values across all columns for each row, not including nan's

nunique_vals_per_row = []
for idx in range(0, len(sex_cols.index)):
    
    nunique_vals_per_row.append(len(sex_cols.iloc[idx,:][sex_cols.iloc[idx,:].notna()].unique()))

#verify that each row only has one unique value
#print('number of unique values per row: ', np.unique(nunique_vals_per_row))
#create new column using unique value across rows in duplicate columns

sex = []

for i in range(0, len(abcd_merged_unique_subjects)):

    for j in range(0, len(sex_cols.columns)):
        if pd.notna(sex_cols.iloc[i,j]):
            sex.append(sex_cols.iloc[i,j])
            break
np.unique(sex)

#collection_title
#print all column names that start with 'collection_title' (how many times was it duplicated in merge)

filter_col_collection_title = [col for col in abcd_merged_unique_subjects if col.startswith('collection_title')]
#retreive indices of all columns that were duplicated

collection_titles =[]

for i in range(0,len(filter_col_collection_title)):

    #print(abcd_merged_unique_subjects.columns.get_loc(filter_col_collection_title[i]))
    collection_titles.append(abcd_merged_unique_subjects.columns.get_loc(filter_col_collection_title[i]))
#create sliced version of df with only the columns corresponding to "collection_title"

collection_title_cols = abcd_merged_unique_subjects.iloc[:, collection_titles]
#check unique values across all columns for each row, not including nan's

nunique_vals_per_row = []
for idx in range(0, len(collection_title_cols.index)):
    
    nunique_vals_per_row.append(len(collection_title_cols.iloc[idx,:][collection_title_cols.iloc[idx,:].notna()].unique()))

#verify that each row only has one unique value
#print('number of unique values per row: ', np.unique(nunique_vals_per_row))
#create new column using unique value across rows in duplicate columns

collection_title = []

for i in range(0, len(abcd_merged_unique_subjects)):

    for j in range(0, len(collection_title_cols.columns)):
        if pd.notna(collection_title_cols.iloc[i,j]):
            collection_title.append(collection_title_cols.iloc[i,j])
            break
np.unique(collection_title)

#src_subject_id
#print all column names that start with 'src_subject_id' (how many times was it duplicated in merge)

filter_col_src_subject_id = [col for col in abcd_merged_unique_subjects if col.startswith('src_subject_id')]
#retreive indices of all columns that were duplicated

src_subject_ids =[]

for i in range(0,len(filter_col_src_subject_id)):

    #print(abcd_merged_unique_subjects.columns.get_loc(filter_col_src_subject_id[i]))
    src_subject_ids.append(abcd_merged_unique_subjects.columns.get_loc(filter_col_src_subject_id[i]))
#create sliced version of df with only the columns corresponding to "src_subject_id"

src_subject_id_cols = abcd_merged_unique_subjects.iloc[:, src_subject_ids]

#check unique values across all columns for each row, not including nan's

nunique_vals_per_row = []
for idx in range(0, len(src_subject_id_cols.index)):
    #unique_vals_per_row.extend(src_subject_id_cols.iloc[idx,:][src_subject_id_cols.iloc[idx,:].notna()].unique())
    nunique_vals_per_row.append(len(src_subject_id_cols.iloc[idx,:][src_subject_id_cols.iloc[idx,:].notna()].unique()))

#verify that each row only has one unique value
#print('number of unique values per row: ', np.unique(nunique_vals_per_row))
#create new column using unique value across rows in duplicate columns

src_subject_id = []

for i in range(0, len(abcd_merged_unique_subjects)):

    for j in range(0, len(src_subject_id_cols.columns)):
        if pd.notna(src_subject_id_cols.iloc[i,j]):
            src_subject_id.append(src_subject_id_cols.iloc[i,j])
            break

#interview_date
#print all column names that start with 'interview_date' (how many times was it duplicated in merge)

filter_col_interview_date = [col for col in abcd_merged_unique_subjects if col.startswith('interview_date')]

#retreive indices of all columns that were duplicated

interview_dates =[]

for i in range(0,len(filter_col_interview_date)):

    #print(abcd_merged_unique_subjects.columns.get_loc(filter_col_interview_date[i]))
    interview_dates.append(abcd_merged_unique_subjects.columns.get_loc(filter_col_interview_date[i]))
#create sliced version of df with only the columns corresponding to "interview_date"

interview_date_cols = abcd_merged_unique_subjects.iloc[:, interview_dates]
#check unique values across all columns for each row, not including nan's

nunique_vals_per_row = []
for idx in range(0, len(interview_date_cols.index)):
    
    nunique_vals_per_row.append(len(interview_date_cols.iloc[idx,:][interview_date_cols.iloc[idx,:].notna()].unique()))

#verify that each row only has one unique value
#print('number of unique values per row: ', np.unique(nunique_vals_per_row))
#create new column using unique value across rows in duplicate columns

interview_date = []

for i in range(0, len(abcd_merged_unique_subjects)):

    for j in range(0, len(interview_date_cols.columns)):
        if pd.notna(interview_date_cols.iloc[i,j]):
            interview_date.append(interview_date_cols.iloc[i,j])
            break

#interview_age
#print all column names that start with 'interview_age' (how many times was it duplicated in merge)

filter_col_interview_age = [col for col in abcd_merged_unique_subjects if col.startswith('interview_age')]
#retreive indices of all columns that were duplicated

interview_ages =[]

for i in range(0,len(filter_col_interview_age)):

    #print(abcd_merged_unique_subjects.columns.get_loc(filter_col_interview_age[i]))
    interview_ages.append(abcd_merged_unique_subjects.columns.get_loc(filter_col_interview_age[i]))
#create sliced version of df with only the columns corresponding to "interview_age"

interview_age_cols = abcd_merged_unique_subjects.iloc[:, interview_ages]
#check unique values across all columns for each row, not including nan's

nunique_vals_per_row = []
for idx in range(0, len(interview_age_cols.index)):
    
    nunique_vals_per_row.append(len(interview_age_cols.iloc[idx,:][interview_age_cols.iloc[idx,:].notna()].unique()))

#verify that each row only has one unique value
#print('number of unique values per row: ', np.unique(nunique_vals_per_row))
#create new column using unique value across rows in duplicate columns

interview_age = []

for i in range(0, len(abcd_merged_unique_subjects)):

    for j in range(0, len(interview_age_cols.columns)):
        if pd.notna(interview_age_cols.iloc[i,j]):
            interview_age.append(interview_age_cols.iloc[i,j])
            break


#print all column names that start with 'dataset_id' (how many times was it duplicated in merge)

filter_col_dataset_id = [col for col in abcd_merged_unique_subjects if col.startswith('dataset_id')]

#retreive indices of all columns that were duplicated

dataset_ids =[]

for i in range(0,len(filter_col_dataset_id)):

    #print(abcd_merged_unique_subjects.columns.get_loc(filter_col_dataset_id[i]))
    dataset_ids.append(abcd_merged_unique_subjects.columns.get_loc(filter_col_dataset_id[i]))
#create sliced version of df with only the columns corresponding to "dataset_id"

dataset_id_cols = abcd_merged_unique_subjects.iloc[:, dataset_ids]
#check unique values across all columns for each row, not including nan's

nunique_vals_per_row = []
for idx in range(0, len(dataset_id_cols.index)):
    
    nunique_vals_per_row.append(len(dataset_id_cols.iloc[idx,:][dataset_id_cols.iloc[idx,:].notna()].unique()))

#verify that each row only has one unique value
#print('number of unique values per row: ', np.unique(nunique_vals_per_row))

# in this case the dataset id will be different in each col bc they come from different studies
#will not carry this col forward

#Remove all duplicate columns from df
#concatenating all indices into one list and sorting from greatest to least

columns_to_drop = (collection_ids + collection_titles + src_subject_ids + dataset_ids + sexs + interview_dates + interview_ages) 
sorted_columns_to_drop = sorted(columns_to_drop, reverse = True)

#check columns being dropped

abcd_merged_unique_subjects.iloc[:,sorted_columns_to_drop]
abcd_nonimaging = abcd_merged_unique_subjects.drop(abcd_merged_unique_subjects.columns[sorted_columns_to_drop], axis=1)

#Add new column names into final df abcd_nonimaging
abcd_nonimaging.insert(loc=0,column='src_subject_id',value=src_subject_id)
abcd_nonimaging.insert(loc=1,column='sex',value=sex)
abcd_nonimaging.insert(loc=2,column='interview_date',value=interview_date)
abcd_nonimaging.insert(loc=3,column='interview_age',value=interview_age)
abcd_nonimaging['collection_id'] = collection_id
abcd_nonimaging['collection_title'] = collection_title

#csv file to save output of cleaned data merge
abcd_nonimaging.to_csv(f'{save_dir}/abcd_nonimaging.csv', index = False)



#create version of dataframe with multiindex and unstacked on 'eventname'

#load in abcd_nonimaging
#abcd_df = pd.read_csv(f'{save_dir}/abcd_nonimaging.csv', skiprows=[1])
abcd_df = abcd_nonimaging[1:].reset_index(drop=True)

subject_drop = []

#check where subjectkey and src_subject_id arent the same (only 9 out of 92053 (one particular subject))
for i in range(0,len(abcd_df)):
    if abcd_df.loc[i, 'subjectkey'] != abcd_df.loc[i, 'src_subject_id']:
        print("subject key:", abcd_df.loc[i, 'subjectkey'], "src_subject_id:", abcd_df.loc[i, 'src_subject_id'])
        subject_drop.append(i)

#drop rows where subjectkey != src_subject_id
abcd_df = abcd_df.drop(subject_drop, axis =0)

#Create MultiIndex and unstack on 'eventname'
abcd_sorted = abcd_df.sort_values('subjectkey', axis=0)

tuples = list(
    zip(
        abcd_sorted.subjectkey,
        abcd_sorted.eventname,
    )
)
index = pd.MultiIndex.from_tuples(tuples, names=["subjectkey", "eventname"])
abcd_sorted = abcd_sorted.set_index(index)

#drop columns which are now indices as well as those which are always the same value
abcd_sorted = abcd_sorted.drop(['subjectkey', 'eventname', 'src_subject_id', 'collection_id', 'collection_title'], axis=1)

abcd_unstacked = abcd_sorted.unstack()

#are any columns entirely nan?
nans_per_col = abcd_unstacked.isnull().sum(axis=0).tolist()
nans_per_col = np.array(nans_per_col)
non_nan_columns = np.where(nans_per_col != 11875)[0]
len(non_nan_columns)
#create new df with only those cols that were not nan from abcd_unstacked
abcd_unstacked_populated = abcd_unstacked.iloc[:, non_nan_columns]

abcd_df_stacked = abcd_unstacked_populated.stack()

#save to csv
abcd_df_stacked.to_csv(f'{save_dir}/abcd_stacked.csv')