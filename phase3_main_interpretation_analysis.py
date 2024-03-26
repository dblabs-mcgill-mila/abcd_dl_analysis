#phase 3 interpretation notebook
"""
this script contains all analyses of trained CVAE model producing main results
    -main text figures, main text tables, supplementary tables, supplementary figures
    
    -compute 95th percentile among all 100 components to retain phenotypes only in the 5 components where they exhibit the highest weight strength
    -Grouping the phenotypes in each of the top 10 components (A-J) into 23 predefined categories
    -Identifying driving catgeogries per component, and driving phenotypes per component
    -Determining overlap of SES variables across components A, B, D, E
    -predicting US state residency using participant SES variable scores in components A, B, C, D
    
Table 1: Mean weight strength and proportion of phenotypes from the Socioeconomic (SES) category across top 10 components (manually created)
    -created based on SES rows from Table S1 and Table S2
    
Figure 1: Study Pipeline
	-CVAE scree plot elbow
	-horizontal heatmap 95th percentile
Figure 2: weight strength bar plots + radial plots
Figure 3: SES venn diagram
Figure 4: Comp A phenotypes 
Figure 5:Comp B phenotypes 
Figure 6:Comp D phenotypes
Figure 7:Comp E phenotypes
Figure 8: US State plot
	-geo plot
	-density plots


Supplement figures+ tables:

Table S1: Mean weight strength across categories for top 10 components (95th %ile)
Table S2: Proportion of phenotypes across categories for top 10 components (95th %ile)
Table S3: unique SES measures in each 4 SES centric components
Table S4: original and updated ABCD category names
Table S5: full list of phenotypes with category and description
Table S6: Phenotype rankings per component (all 100)

Figure S1: PCA scree plots
Figure S2: Comparison of CVAE architectures and PCA MSE loss obtained
Figure S3: CVAE scree plot all 100 components

Figure S5: logReg confusion matrix
Figure S6: logReg coefficients per state

"""

#import libraries

import torch
import trvaep
from trvaep.utils import cvae_pca_cross_corr_test_v2, one_hot_encoder, new_reverse_one_hot_encoder
import pandas as pd
import numpy as np
from scipy import stats
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import math
from math import pi
from venn import venn

#plotting
from matplotlib import pyplot as plt
import seaborn as sns 
from mpl_toolkits.axisartist.axislines import AxesZero
import geopandas

from config import SAVE_DIRECTORY_PHASE1, SAVE_DIRECTORY_PHASE2, SAVE_DIRECTORY_PHASE3, GEO_DATA
#directories to save and load files
save_dir = SAVE_DIRECTORY_PHASE3
os.makedirs(save_dir, exist_ok = True)

load_dir_phase1 = SAVE_DIRECTORY_PHASE1
load_dir_phase2 = SAVE_DIRECTORY_PHASE2

#usa-states-census-2014.shp from https://github.com/joncutrer/geopandas-tutorial.git
load_us_geo_template = f'{GEO_DATA}/geopandas-tutorial/data'

#load data
data = pd.read_csv(f'{load_dir_phase1}/baseline_screen_6_1yr_z_4_cleaned.csv', index_col= 'subjectkey')

#arrange data discrete then continuous variables

numeric_df = pd.read_csv(f'{load_dir_phase1}/numeric_df', index_col= 'subjectkey')

#isolate all columns ids in data that are from numeric dataframe
numeric_data_col_idx = data.columns.isin(numeric_df.columns)

#split into numeric data using numeric column ids
numeric_data = data.iloc[:,numeric_data_col_idx]

#split into categorical data using non numeric column ids
categorical_data = data.iloc[:,~numeric_data_col_idx]


data = pd.concat([categorical_data, numeric_data], axis = 1)


#import trained model
#pre-pca input data
#arch5
latent_dim = 100
enc_dim = [150]
dec_dim = [150]
arch_number = '5'
pca_dim = 200
split = None
seed = 0

#number of abcd sites
n_conditions = 21

model = trvaep.CVAE(input_dim = 200, num_classes= n_conditions, #use 'input_d' for non-PCA, 'pca_d' for PCA input
                    encoder_layer_sizes=enc_dim, decoder_layer_sizes=dec_dim, latent_dim=100, alpha=0.001, use_batch_norm=False,
                    dr_rate=0, use_mmd=False, beta=1, output_activation='linear', var_type_split= split, pca_input = True, pca_dim = pca_dim, \
                        top_std_cols = False, splitter = 'strat_shuffle', decoder_choice = 'base') 

#import state dictionary
model.load_state_dict(torch.load(f'{load_dir_phase2}/cvae_model_final/arch5_seed_0/model.pt'))

#import trainer to replicate dataset splits used during model training
trainer = trvaep.Trainer_v2(model, data, seed = seed)

if model.pca_input == True:
    #return test, validation and train data splits used
    #original data in form of numpy arrays (before PCA)
    dataset_train, dataset_valid, dataset_test, orig_train_data, orig_validate_data, orig_test_data, pca_model = trainer.make_dataset_abcd()
    

#dataframe linking each phenotype to ABCD provided category
loadings_base_screen_6_1yr_category = pd.read_csv(f'{load_dir_phase1}/loadings_base_screen_6_1yr_category_updated.csv', index_col=0)

#update domain names to our more appropriate names
#13 changes out of 23
#basis of supplementary table 4: updated category names and original names

loadings_base_screen_6_1yr_category.loc[loadings_base_screen_6_1yr_category['category'] == 'Activity', 'category'] = 'Extracurricular'

loadings_base_screen_6_1yr_category.loc[loadings_base_screen_6_1yr_category['category'] == 'Food', 'category'] = 'Nutrition'

loadings_base_screen_6_1yr_category.loc[loadings_base_screen_6_1yr_category['category'] == 'Cognitive', 'category'] = 'Cognitive Capacity'

loadings_base_screen_6_1yr_category.loc[loadings_base_screen_6_1yr_category['category'] == 'Personality', 'category'] = 'Impulsivity'

loadings_base_screen_6_1yr_category.loc[loadings_base_screen_6_1yr_category['category'] == 'Phys Characteristics', 'category'] = 'Pubertal Development'

loadings_base_screen_6_1yr_category.loc[loadings_base_screen_6_1yr_category['category'] == 'Diagnostic', 'category'] = 'Diagnostic (KSADS)'

loadings_base_screen_6_1yr_category.loc[loadings_base_screen_6_1yr_category['category'] == 'Med History', 'category'] = 'Medical History'

loadings_base_screen_6_1yr_category.loc[loadings_base_screen_6_1yr_category['category'] == 'Phys Exam', 'category'] = 'Physical Exam'

loadings_base_screen_6_1yr_category.loc[loadings_base_screen_6_1yr_category['category'] == 'Parenting', 'category'] = 'Parent Characteristics'

loadings_base_screen_6_1yr_category.loc[loadings_base_screen_6_1yr_category['category'] == 'Questionnaire', 'category'] = 'Deep Phenotyping Assessments'

loadings_base_screen_6_1yr_category.loc[loadings_base_screen_6_1yr_category['category'] == 'Treatment', 'category'] = 'Medications'

loadings_base_screen_6_1yr_category.loc[loadings_base_screen_6_1yr_category['category'] == 'Summary', 'category'] = 'Mental Health Summary'

loadings_base_screen_6_1yr_category.loc[loadings_base_screen_6_1yr_category['category'] == 'Task Based', 'category'] = 'Neuropsychological Tests'




#update order of manhattan plots to use our new category names
category_order = np.array(['Socioeconomic', 'Activity', 'Food', 'Behavior', 'Life Events', 'Cognitive', 'Personality', 'Demographics', 'Phys Characteristics', 'Diagnostic',
       'Mania', 'Med History', 'Phys Exam', 'Parenting',
       'Questionnaire','Trauma', 'Sleep', 'Treatment', 'Social Adjustment',
       'Social Responsiveness', 'Substance Use',
       'Summary', 'Task Based'], dtype=object)

category_order[category_order == 'Activity'] = 'Extracurricular'

category_order[category_order == 'Food'] = 'Nutrition'

#category_order[category_order == 'Behavior'] = 'Personality Traits'

category_order[category_order == 'Cognitive'] = 'Cognitive Capacity'

category_order[category_order == 'Personality'] = 'Impulsivity'

category_order[category_order == 'Phys Characteristics'] = 'Pubertal Development'

category_order[category_order == 'Diagnostic'] = 'Diagnostic (KSADS)'

category_order[category_order == 'Med History'] = 'Medical History'

category_order[category_order == 'Phys Exam'] = 'Physical Exam'

category_order[category_order == 'Parenting'] = 'Parent Characteristics'

category_order[category_order == 'Questionnaire'] = 'Deep Phenotyping Assessments'

category_order[category_order == 'Treatment'] = 'Medications'

category_order[category_order =='Summary'] = 'Mental Health Summary'

category_order[category_order == 'Task Based'] = 'Neuropsychological Tests'
#cross correlation using the test data set for projection and loadings
#this segment is to do cross correlation between cvae latent variable "loadings" and PCA component loadings
# also self correlates cvae latent variables and projection of data into PCA latent space

#use full data minus label columns
X = data.drop(data.iloc[:,7:28], axis = 1)

if model.pca_input == True:
    #add site columns back to their original column index position to be included in cvae loadings
    #need to one hot encode test set labels before inserting back into test input data
    one_hot_test_labels = one_hot_encoder(torch.tensor(dataset_test.data_labels), model.num_cls)

    #insert back at index position 7 where it is in input dataframe
    orig_test_data_with_site = np.concatenate((orig_test_data[:,:7],one_hot_test_labels.detach().numpy().astype('float64'), orig_test_data[:,7:]) , axis = 1)

    #need to one hot encode train set labels before inserting back into test input data
    #for training PCA with site labels
    one_hot_train_labels = one_hot_encoder(torch.tensor(dataset_train.data_labels), model.num_cls)

    #insert back at index position 7 where it is in input dataframe
    orig_train_data_with_site = np.concatenate((orig_train_data[:,:7],one_hot_train_labels.detach().numpy().astype('float64'), orig_train_data[:,7:]) , axis = 1)
    ambient_train_data = orig_train_data_with_site
else:
    ambient_train_data = dataset_train.data

#retrieve latent variables from CVAE latent space by passing set of observations through encoder and using output to sample from latent distribution
latent_variables = model.get_latent(torch.Tensor(dataset_test.data), torch.Tensor(dataset_test.data_labels), mean = True) #can try with mean true and false
#These are latent variables of CVAE

#function to compute cvae loadings and pca loadings, as well as cross correlation between cvae loadings and pca loadings, and cvae and pca projections
cvae_loadings_df_test, loadings_pca_df, corr_matrix_loadings_df, cvae_pca_proj_corr_df = cvae_pca_cross_corr_test_v2(orig_test_data_with_site, orig_train_data_with_site, latent_variables, latent_dim, data)
#rename columns
cvae_col_names = []
for i in range(0,100):
    cvae_col_names.append('Latent_Dim' + str(i+1))

cvae_loadings_df_test.columns = cvae_col_names
#merge cvae loadings with domain category names
cvae_loadings_df_test_category = pd.merge(cvae_loadings_df_test, loadings_base_screen_6_1yr_category.category, how='left', left_index=True, right_index=True)
#sort by category so plots have like colours grouped
cvae_loadings_df_test_category = cvae_loadings_df_test_category.sort_values(by=['category'])
#drop rows where category value is 'nan'
cvae_loadings_df_test_category = cvae_loadings_df_test_category[~cvae_loadings_df_test_category.category.isnull()]
cvae_loadings_df_test_category


#create mapping df with numbers for order of categories
df_mapping = pd.DataFrame({'category': category_order})

sort_mapping = df_mapping.reset_index().set_index('category')
#create new column in dataframe based on numbers for each category
cvae_loadings_df_test_category['cat_num'] = cvae_loadings_df_test_category['category'].map(sort_mapping['index'])
#sort dataframe by newly created cat_num column
cvae_loadings_df_test_category = cvae_loadings_df_test_category.sort_values(by=['cat_num'])
#add 'i' column to determine the order they are plotted in
cvae_loadings_df_test_category['i'] = range(0,len(cvae_loadings_df_test_category))
cvae_loadings_df_test_category


#add in descriptions and nda_instrument (study name)
cvae_loadings_df_test_category_desc = pd.merge(cvae_loadings_df_test_category, loadings_base_screen_6_1yr_category.description, how='left', left_index=True, right_index=True)
cvae_loadings_df_test_category_desc = pd.merge(cvae_loadings_df_test_category_desc, loadings_base_screen_6_1yr_category.nda_instrument, how='left', left_index=True, right_index=True)
#identify value of 95th percentile for each loading
#use absolute value of df

loadings_scoreatpercentile_list = []
for loading in cvae_loadings_df_test.index:

    loadings_scoreatpercentile_list.append(stats.scoreatpercentile(cvae_loadings_df_test.abs().loc[loading,:], 95))
#threshold loadings at 95th percentile
cvae_loadings_df_test_thresholded = cvae_loadings_df_test.copy()

for i in range(0, len(loadings_scoreatpercentile_list)):

    cvae_loadings_df_test_thresholded.iloc[i,:].where(cvae_loadings_df_test_thresholded.iloc[i,:].abs() > loadings_scoreatpercentile_list[i], 0, inplace = True)

#merge thresholded loadings with category and description
cvae_loadings_df_test_thresholded_category_desc = pd.concat([cvae_loadings_df_test_thresholded, cvae_loadings_df_test_category_desc.iloc[:,-5:]], axis = 1)
def get_colors(n_colors):

    #I used the following parameters
	#Colourblind palette
	# H 0 - 360
	# C 60 - 100
    # L 0 - 70
    # Hard makes more different colors

    if n_colors == 23:
        palette_23 = ["#ff3dad",
            "#01b125",
            "#011fc1",
            "#ff8b22",
            "#6437e2",
            "#b08200",
            "#8b5dff",
            "#f81f03",
            "#000683",
            "#cc4300",
            "#d30fd6",
            "#993500",
            "#e769ff",
            "#ff7a47",
            "#9b00af",
            "#fb002f",
            "#fc30e5",
            "#f5725e",
            "#6b1374",
            "#f70055",
            "#8e0077",
            "#d74675",
            "#d6549e"]
        
        colors = palette_23
        
    elif n_colors == 15:

        palette_15 = ["#01a966",
            "#3b0b7c",
            "#ebd54c",
            "#0061d1",
            "#00a342",
            "#f185f7",
            "#b97c00",
            "#2da8ff",
            "#ffa348",
            "#241c58",
            "#6c4400",
            "#ffb3f1",
            "#620022",
            "#ff78b8",
            "#730046"]
        colors = palette_15

    elif n_colors == 10:

        palette_10 = ["#e8a0ff",
            "#00adef",
            "#c3005f",
            "#99c20e",
            "#ffa866",
            "#6c4376",
            "#735200",
            "#ff7a8a",
            "#952c40",
            "#e399ac"
            ]

        colors = palette_10


    return colors
#plot legend with number of variables per cateogry
variables_per_category = cvae_loadings_df_test_category.groupby('category').count().iloc[:,0]

colors = get_colors(23)
fig, ax = plt.subplots(1, figsize=(4.5, 7))

# Plot the vertical bar chart
bars = ax.bar(variables_per_category.index, variables_per_category, color = colors)

# Add counts to the legend
legend_labels = [f"{category} ({count})" for category, count in zip(variables_per_category.index, variables_per_category)]
ax.legend(bars, legend_labels, loc='center')

params = {'legend.fontsize': 24}
plt.rcParams.update(params)

# Hide axes
ax.axis('off')

fig.tight_layout()

fig.savefig(f'{save_dir}/variable_count_legend.pdf', format='pdf', dpi = 300, bbox_inches='tight')



#pca explained variance scree plots
#supplementary Figure S1

#pca scores/projections pearson correlated with data variables
#ordered abcd df, pca fit on PCAd abcd data
#this is the PCA model, using random seed=0 for repeatability
latent_dim = 100

pca_100 = PCA(n_components=latent_dim, random_state=0)
#fit model with train data and apply dim reduction on train data
pca_100.fit(ambient_train_data)

#transform ambient test data into latent space
pca_test_projections = pca_100.transform(orig_test_data_with_site)

params = {'axes.labelsize': 28, 'legend.fontsize': 18, 'xtick.labelsize': 28, 'ytick.labelsize': 28}
plt.rcParams.update(params)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Component')
ax.set_ylabel('Percent of Variance Explained')
#ax.set_title('Explained Variance Ratio Spectrum across principal components')
plt.tight_layout()

plt.plot(range(1,101), pca_100.explained_variance_ratio_*100, 'o-', linewidth=2, color='blue')

#mark elbow with veritcal line
#plt.axvline(x = 10, color = 'r', label = 'Elbow')
#plt.text(10, -0.004, '10', color = 'r', ha='center', va='center')


plt.savefig(f'{save_dir}/pca_scree_explained_variance_percent_elbow.pdf', format='pdf', bbox_inches="tight")


pca_100_explained_variance_cumulative = np.cumsum(pca_100.explained_variance_ratio_)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Component')
ax.set_ylabel('Percent of Variance Explained')
#ax.set_title('Explained variance percentage across principal components')
plt.tight_layout()

plt.plot(range(1,101), pca_100_explained_variance_cumulative*100, 'o-', linewidth=2, color='black', label = 'Cumulative')
plt.plot(range(1,101), pca_100.explained_variance_ratio_*100, 'o-', linewidth=2, color='blue', label = 'Individual')

#mark elbow with veritcal line
#plt.axvline(x = 10, color = 'r', label = 'Elbow')
#plt.text(10, -4.2, '10', color = 'r', ha='center', va='center', size = 28)
plt.legend()


plt.savefig(f'{save_dir}/pca_scree_explained_variance_percent_elbow_cumulative.pdf', format='pdf', dpi = 300, bbox_inches="tight")


#reconstruction error all latent dimensions

#model in eval mode
model.eval()
#forward pass on test data
recon_test, _, _ = model.forward(torch.Tensor(dataset_test.data), torch.Tensor(dataset_test.data_labels))

#compute recon loss on CVAE input/output (in PCA latent space)
test_mse_loss_latent = torch.nn.functional.mse_loss(recon_test, torch.Tensor(dataset_test.data), reduction="sum")
test_mse_loss_latent = test_mse_loss_latent / recon_test.size(0)
print("test loss latent: ", test_mse_loss_latent.item())


#inverse transform recon data
test_set_reconstructed = pca_model.inverse_transform(recon_test.detach().numpy())

recon_test_data = torch.Tensor(test_set_reconstructed)
original_test_data = torch.Tensor(orig_test_data)

#compute validation mse loss
test_mse_loss = torch.nn.functional.mse_loss(recon_test_data, original_test_data, reduction="sum")
test_mse_loss = test_mse_loss / recon_test_data.size(0)
print("test loss: ", test_mse_loss.item())

recon_error_all_dims = test_mse_loss.item()
recon_error_all_dims_latent = test_mse_loss_latent.item()


#take mean latent sample for repeatability

deviation_mse_scores_list = []
deviation_mse_scores_list_latent = []

latent_sample = model.get_latent(torch.Tensor(dataset_test.data), torch.Tensor(dataset_test.data_labels), mean = True)

#calculate explained variance per component, iteratively zero out all latent dims except one of latent sample
for latent_d in range(0,latent_sample.shape[1]):

    latent_sample_zeroed = latent_sample.copy()

    #zero out all latent dims except one of latent sample
    latent_sample_zeroed[:,(latent_d+1):] = 0
    latent_sample_zeroed[:,:latent_d] = 0

    #feed into decoder and get reconstruction
    latent_sample_recon_latent = model.reconstruct(torch.Tensor(latent_sample_zeroed), torch.Tensor(dataset_test.data_labels), use_latent=True)

    latent_sample_recon_latent_tensor = torch.Tensor(latent_sample_recon_latent)

    #compute mse in latent space
    test_mse_loss_latent = torch.nn.functional.mse_loss(latent_sample_recon_latent_tensor, torch.Tensor(dataset_test.data), reduction="sum")
    test_mse_loss_latent = test_mse_loss_latent / latent_sample_recon_latent_tensor.size(0) #divide by number of obs
    deviation_mse_scores_list_latent.append(test_mse_loss_latent.item())

    #inverse pca transform back to data space to compare with original test data
    latent_sample_recon_ambient = pca_model.inverse_transform(latent_sample_recon_latent)

    latent_sample_recon_ambient_tensor = torch.Tensor(latent_sample_recon_ambient)
    original_test_data = torch.Tensor(orig_test_data)

    #compute test mse loss
    test_mse_loss_ambient = torch.nn.functional.mse_loss(latent_sample_recon_ambient_tensor, original_test_data, reduction="sum")
    test_mse_loss_ambient = test_mse_loss_ambient / latent_sample_recon_ambient_tensor.size(0) #divide by number of obs
    deviation_mse_scores_list.append(test_mse_loss_ambient.item())



#explained variance =  (Reconstruction Error without all but one dim) - (Reconstruction Error with all dimensions)
dev_mse_ambient = [x - recon_error_all_dims for x in deviation_mse_scores_list]

dev_mse_latent = [x - recon_error_all_dims_latent for x in deviation_mse_scores_list_latent]

#dataframe of explained variance per component
deviation_mse_scores_latent_df = pd.DataFrame(dev_mse_latent, index = range(1,len(deviation_mse_scores_list_latent)+1), columns = ['deviation_mse_score_latent'])
deviation_mse_scores_latent_df.index.name = 'Latent_Dim'
deviation_mse_scores_latent_df['dev_mse_score_latent_rank'] = deviation_mse_scores_latent_df['deviation_mse_score_latent'].rank(ascending = True)



#capture list of sorted latent dims to use
cvae_elbow_dims_list = deviation_mse_scores_latent_df.deviation_mse_score_latent.sort_values(ascending=True).index

#name columns
cvae_elbow_dims_names = []
for dim in cvae_elbow_dims_list:
    cvae_elbow_dims_names.append('Latent_Dim' + str(dim))

#elbow components from latent dim ranking
cvae_elbow_dims_names_series = pd.Series(cvae_elbow_dims_names)
#save component ranking order to csv
cvae_component_ranking_order = cvae_elbow_dims_names_series.rename("Component")
cvae_component_ranking_order.to_csv(f'{save_dir}/cvae_component_ranking_order.csv')
#CVAE scree plot for study pipeline diagram


fig = plt.figure(figsize = (8,8))
#ax = fig.add_subplot(1,1,1) 
ax = fig.add_subplot(axes_class=AxesZero)

for direction in ["yzero"]:
    # adds arrows at the ends of each axis
    #ax.axis[direction].set_axisline_style("-|>")

    # adds X and Y-axis from the origin
    ax.axis[direction].set_visible(True)

for direction in ["left", "right", "top"]:
    # hides borders
    ax.axis[direction].set_visible(False)

ax.set_xlabel('Component')
ax.set_ylabel('Explained Variance')


#ax.set_title('CVAE Components Sorted by Explained Variance')
#plt.yticks([])
plt.xticks(fontsize=12)
#ax.arrow(0.1, 0.2, 0, 0.1, transform=ax.transAxes)
plt.tight_layout()
#effect size for each latent dim 
#smaller is better

plt.plot(deviation_mse_scores_latent_df.deviation_mse_score_latent.index[:30], deviation_mse_scores_latent_df.deviation_mse_score_latent.sort_values(ascending=True)[:30], 'o-', linewidth=2, color='blue')
plt.gca().invert_yaxis()
#mark elbow with veritcal line
plt.axvline(x = 10, color = 'r', label = 'Elbow')

plt.legend(loc = 'upper right')

params = {'axes.labelsize': 34, 'legend.fontsize': 34, 'xtick.labelsize': 28, 'ytick.labelsize': 28}
plt.rcParams.update(params)

plt.savefig(f'{save_dir}/cvae_scree_elbow_workflow_diagram.pdf', format='pdf', dpi = 300, bbox_inches="tight")


#CVAE scree plot showing all 100 components (supplement)


fig = plt.figure(figsize = (8,8))
#ax = fig.add_subplot(1,1,1) 
ax = fig.add_subplot(axes_class=AxesZero)


for direction in ["yzero"]:
    # adds arrows at the ends of each axis
    #ax.axis[direction].set_axisline_style("-|>")

    # adds X and Y-axis from the origin
    ax.axis[direction].set_visible(True)

for direction in ["left", "right", "top"]:
    # hides borders
    ax.axis[direction].set_visible(False)

ax.set_xlabel('Component')
ax.set_ylabel('Explained Variance')


#ax.set_title('CVAE Components Sorted by Explained Variance')
#plt.yticks([])
plt.xticks(fontsize=12)
#ax.arrow(0.1, 0.2, 0, 0.1, transform=ax.transAxes)
plt.tight_layout()

#smaller is better

plt.plot(deviation_mse_scores_latent_df.deviation_mse_score_latent.index, deviation_mse_scores_latent_df.deviation_mse_score_latent.sort_values(ascending=True), 'o-', linewidth=2, color='blue')
plt.gca().invert_yaxis()
#mark elbow with veritcal line
plt.axvline(x = 10, color = 'r', label = 'Elbow')
plt.text(0.127, -0.032, '10', color = 'r', ha='center', va='center', transform=ax.transAxes, fontsize=28)

plt.legend(loc = 'upper right')

params = {'axes.labelsize': 34, 'legend.fontsize': 34, 'xtick.labelsize': 28, 'ytick.labelsize': 28}
plt.rcParams.update(params)

plt.savefig(f'{save_dir}/cvae_scree_method_4_supplement.pdf', format='pdf', dpi = 300, bbox_inches="tight")


#supplementary table 1: mean weight strength per category

#calculate mean effect size per category only considering non-zero values in mean calculation
cvae_effect_category_abs_mean_nonzero = cvae_loadings_df_test_thresholded_category_desc.iloc[:,:-2].groupby('category').apply(lambda c: c[c != 0].abs().mean())
#replace nan by 0
cvae_effect_category_abs_mean_nonzero.fillna(0, inplace = True)

#order by component ranking, 
cvae_effect_category_abs_mean_nonzero_ordered = cvae_effect_category_abs_mean_nonzero.loc[:,cvae_elbow_dims_names_series.values.reshape(-1)]

#only use top 10 dims of elbow
cvae_effect_category_abs_mean_nonzero_ordered_elbow = cvae_effect_category_abs_mean_nonzero_ordered.iloc[:,:10]
cvae_effect_category_abs_mean_nonzero_ordered_elbow.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

#save rounded table to csv
cvae_effect_category_abs_mean_nonzero_ordered_elbow.round(decimals=3).to_csv(f'{save_dir}/cvae_effect_category_abs_mean_nonzero_ordered_elbow.csv')
#heatmap plot of top 10 latent Dims
#95th percentile thresholding
#for study pipeline diagram

cvae_effect_category_abs_mean_nonzero_ordered_elbow = cvae_effect_category_abs_mean_nonzero_ordered.iloc[:,:10]
cvae_effect_category_abs_mean_nonzero_ordered_elbow.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

#default parameter for this plot
plt.rcParams.update(plt.rcParamsDefault)

f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
#cmap = sns.diverging_palette(230, 20, as_cmap=True)
cmap = 'coolwarm'
    

sns.heatmap(cvae_effect_category_abs_mean_nonzero_ordered_elbow.T, xticklabels=True, cmap=cmap, annot=False, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, vmax = 0.41).set(title='CVAE Thresholded Effect Size Per Category (mean nonzero)')

ax = plt.gca()
plt.xticks(fontsize=12, rotation=90, ha='center')
plt.yticks(fontsize=12, rotation=0, ha='right')
#ax.set_xticklabels("")
#ax.tick_params(left=False)
#plt.tight_layout()
plt.title("")
plt.xlabel("")
plt.ylabel("")
plt.savefig(f'{save_dir}/cvae_thresh5_effect_category_abs_mean_nonzero_ranked_dims_elbow.pdf', format='pdf', dpi = 300, bbox_inches='tight')


#format data for barplots
cvae_effect_category_abs_mean_nonzero_ordered_elbow_long = pd.melt(cvae_effect_category_abs_mean_nonzero_ordered_elbow, value_vars = cvae_effect_category_abs_mean_nonzero_ordered_elbow.columns, var_name='Component', value_name='Absolute mean effect size', ignore_index=False).reset_index()

#split into 2 separate dataframes (each with 5 components), better for plotting visual
cvae_effect_category_abs_mean_nonzero_ordered_elbow_long_AE = cvae_effect_category_abs_mean_nonzero_ordered_elbow_long[cvae_effect_category_abs_mean_nonzero_ordered_elbow_long.Component.isin(['A', 'B', 'C', 'D', 'E'])]
cvae_effect_category_abs_mean_nonzero_ordered_elbow_long_FJ = cvae_effect_category_abs_mean_nonzero_ordered_elbow_long[cvae_effect_category_abs_mean_nonzero_ordered_elbow_long.Component.isin(['F', 'G', 'H', 'I', 'J'])]
cat_names = cvae_effect_category_abs_mean_nonzero_ordered_elbow.columns[:5]
colors = get_colors(n_colors=23)

fig, ax = plt.subplots(1, figsize=(12, 30))
g = sns.barplot(data=cvae_effect_category_abs_mean_nonzero_ordered_elbow_long_AE, y='Component', x='Absolute mean effect size', hue = 'category', palette=colors, errorbar=None, orient='h')

ax.set_xlim(0, 0.41) 
ax.get_legend().remove()
#ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), ncol=1)
ax.set_title('') #Absolute component mean effect size per domain after 5% thresholding
ax.set_yticklabels(cat_names)
ax.set_xlabel('Mean Weight Strengths', fontsize = 28)
ax.set_ylabel('')
plt.yticks(fontsize=32, rotation=0, ha='right')
plt.xticks(fontsize=20, rotation=0, ha='center')
for i in range(len(cat_names)-1):
    plt.axhline(y=i+0.5, linestyle='--', linewidth=0.5, color='k')
ax.xaxis.grid(linestyle='-')
sns.despine(left=True)
plt.tight_layout()

plt.savefig(f'{save_dir}/cvae_5percent_thresholded_component_abs_mean_effectsize_per_domain_AE.pdf', format='pdf', dpi = 300, bbox_inches="tight")


cat_names = cvae_effect_category_abs_mean_nonzero_ordered_elbow.columns[5:]
colors = get_colors(n_colors=23)

fig, ax = plt.subplots(1, figsize=(12, 30))
g = sns.barplot(data=cvae_effect_category_abs_mean_nonzero_ordered_elbow_long_FJ, y='Component', x='Absolute mean effect size', hue = 'category', palette=colors, errorbar=None, orient='h')

ax.set_xlim(0, 0.41) 
ax.get_legend().remove()
#ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), ncol=1)
ax.set_title('') #Absolute component mean effect size per domain after 5% thresholding
ax.set_yticklabels(cat_names)
ax.set_xlabel('Mean Weight Strengths', fontsize = 28)
plt.yticks(fontsize=32, rotation=0, ha='right')
plt.xticks(fontsize=20, rotation=0, ha='center')
ax.set_ylabel('')
for i in range(len(cat_names)-1):
    plt.axhline(y=i+0.5, linestyle='--', linewidth=0.5, color='k')
ax.xaxis.grid(linestyle='-')
sns.despine(left=True)
plt.tight_layout()

plt.savefig(f'{save_dir}/cvae_5percent_thresholded_component_abs_mean_effectsize_per_domain_FJ.pdf', format='pdf', dpi = 300, bbox_inches="tight")


#supplementary table 2: proportion of phenotypes from each category per component

cvae_loadings_df_test_thresholded_binarized = cvae_loadings_df_test_thresholded.copy()

#create binarized dataframe based on presence of nonzero loadings
for i in range(0, cvae_loadings_df_test_thresholded.shape[0]):

    cvae_loadings_df_test_thresholded_binarized.iloc[i,:].where(cvae_loadings_df_test_thresholded.iloc[i,:] == 0, 1, inplace = True)

#add in descriptions and nda_instrument (study name)
cvae_loadings_df_test_thresholded_binarized_category_desc = pd.concat([cvae_loadings_df_test_thresholded_binarized, cvae_loadings_df_test_category_desc.iloc[:,-5:]], axis = 1)

#total number of loadings per domain
count_domainwise = cvae_loadings_df_test_thresholded_binarized_category_desc.groupby('category').count().iloc[:,:-4]

#number of nonzero loadings per domain
count_nonzero_5pc_domainwise = cvae_loadings_df_test_thresholded_binarized_category_desc.groupby('category').sum().iloc[:,:-2]
count_nonzero_5pc_domainwise

#percentage of nonzero loadings per domain
percent_nonzero_5pc_domainwise = count_nonzero_5pc_domainwise.div(count_domainwise).mul(100)
percent_nonzero_5pc_domainwise

#order by component ranking
percent_nonzero_5pc_domainwise_ordered = percent_nonzero_5pc_domainwise.loc[:,cvae_elbow_dims_names_series.values.reshape(-1)]

#only use top 10 dims of elbow
percent_nonzero_5pc_domainwise_ordered_elbow = percent_nonzero_5pc_domainwise_ordered.iloc[:,:10]
percent_nonzero_5pc_domainwise_ordered_elbow.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

percent_nonzero_5pc_domainwise_ordered_elbow

#save rounded table to csv
percent_nonzero_5pc_domainwise_ordered_elbow.round(decimals=1).to_csv(f'{save_dir}/percent_nonzero_5pc_domainwise_ordered_elbow.csv')
def my_cbar_plot_weights(data=None, title="", axis_range=None, label=None, color=None, width=0.25,
                 export_figures=False, fig_name=None):
    
    PATH_FIG = f'{save_dir}/'
    
    # Create dataframe
    class_mean = pd.DataFrame(data=data.T, index=label, columns=['mean']).T
    # Number of variable
    categories = list(class_mean)
    N = len(categories)
    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values = class_mean.values.flatten().tolist()
    # values += values[:1]
    # Angle? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    # angles += angles[:1]
    # Initialise the spider plot
    fig = plt.figure(figsize=(16, 8))
    ax = plt.subplot(111, polar=True)
    ax.spines["polar"].set_visible(False)
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles, label, color="black", size=30)
    #ax.tick_params(axis='x', which='major', pad=15)
    ax.xaxis.get_majorticklabels()[0].set_horizontalalignment("left")
    ax.xaxis.get_majorticklabels()[2].set_verticalalignment("bottom")
    ax.xaxis.get_majorticklabels()[3].set_verticalalignment("top")
    # Draw ylabels
    ax.set_rlabel_position(0)
    if axis_range is None:
        axis_range = (np.min(values), np.max(values))
    inc = (axis_range[1] - axis_range[0]) / 4
    newinc = [
        axis_range[0] + inc,
        axis_range[0] + (inc * 2),
        axis_range[0] + (inc * 3),
        0,
    ]
    plt.yticks(
        newinc, [str("{:.2f}".format(elem)) for elem in newinc],
        color="black", size=16
    )

    plt.ylim(axis_range)
    if title:
        plt.title(title, size=28)
    # Plot data
    ax.bar(angles, np.abs(values), alpha=1,
           width=width, linewidth=1, edgecolor='k',
           color=color,
           )
    # ax.yaxis.grid(color=‘white’, linestyle=‘dashed’)
    # ax.xaxis.grid(color=‘white’, linestyle=‘dashed’)
    ax.yaxis.zorder = 1
    if export_figures:
        fig.savefig(PATH_FIG + fig_name,
                    bbox_inches='tight', format='pdf', dpi = 300)
    

def my_cbar_plot_proportions(data=None, title="", axis_range=None, label=None, color=None, width=0.25,
                 export_figures=False, fig_name=None):
    
    PATH_FIG = f'{save_dir}/'
    
    # Create dataframe
    class_mean = pd.DataFrame(data=data.T, index=label, columns=['mean']).T
    # Number of variable
    categories = list(class_mean)
    N = len(categories)
    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values = class_mean.values.flatten().tolist()
    # values += values[:1]
    # Angle? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    # angles += angles[:1]
    # Initialise the spider plot
    fig = plt.figure(figsize=(16, 8))
    ax = plt.subplot(111, polar=True)
    ax.spines["polar"].set_visible(False)
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles, label, color="black", size=30)
    ax.xaxis.get_majorticklabels()[0].set_horizontalalignment("left")
    ax.xaxis.get_majorticklabels()[2].set_verticalalignment("bottom")
    ax.xaxis.get_majorticklabels()[3].set_verticalalignment("top")
    # Draw ylabels
    ax.set_rlabel_position(0)
    if axis_range is None:
        axis_range = (np.min(values), np.max(values))
    inc = (axis_range[1] - axis_range[0]) / 4
    newinc = [
        axis_range[0] + inc,
        axis_range[0] + (inc * 2),
        axis_range[0] + (inc * 3),
        0,
    ]
    plt.yticks(
        newinc, [str("{:.0f}%".format(elem)) for elem in newinc],
        color="black", size=16
    )


    plt.ylim(axis_range)
    if title:
        plt.title(title, size=28)
    # Plot data
    ax.bar(angles, np.abs(values), alpha=1,
           width=width, linewidth=1, edgecolor='k',
           color=color,
           )
    # ax.yaxis.grid(color=‘white’, linestyle=‘dashed’)
    # ax.xaxis.grid(color=‘white’, linestyle=‘dashed’)
    ax.yaxis.zorder = 1
    if export_figures:
        fig.savefig(PATH_FIG + fig_name,
                    bbox_inches='tight', format='pdf', dpi = 300)
    
    
#Radial plot SES weight strength
colors = get_colors(n_colors=10)

my_cbar_plot_weights(cvae_effect_category_abs_mean_nonzero_ordered_elbow.T.Socioeconomic.values, 
             title="Socioeconomic Weight Strength", axis_range=None, label=cvae_effect_category_abs_mean_nonzero_ordered_elbow.T.index, color=colors, width=0.25,
                 export_figures=True, fig_name='radial_bar_plot_ses_effect.pdf')

#Radial plot SES proportions
colors = get_colors(n_colors=10)

my_cbar_plot_proportions(percent_nonzero_5pc_domainwise_ordered_elbow.T.Socioeconomic.values, 
             title="Socioeconomic Proportion", axis_range=None, label=percent_nonzero_5pc_domainwise_ordered_elbow.T.index, color=colors, width=0.25,
                 export_figures=True, fig_name='radial_bar_plot_ses_proportion.pdf')
#supplementary table 3: unique SES measures per component (4 components) (post-thresholding)

#df with top 10 components and category
cvae_loadings_df_test_top10_cat = cvae_loadings_df_test_thresholded_category_desc.iloc[:,:-4].loc[:,list(cvae_elbow_dims_names_series[:10]) + ['category']]
cvae_loadings_df_test_top10_cat.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'Category']
cvae_loadings_df_test_top10_cat

#find intersection of SES variables in comp A, B, D, E

#find non-zero SES loadings in each component
compA_ses_5pc_nonzero = cvae_loadings_df_test_top10_cat.query("(Category=='Socioeconomic') and (A != 0)")['A']
compB_ses_5pc_nonzero = cvae_loadings_df_test_top10_cat.query("(Category=='Socioeconomic') and (B != 0)")['B']
compD_ses_5pc_nonzero = cvae_loadings_df_test_top10_cat.query("(Category=='Socioeconomic') and (D != 0)")['D']
compE_ses_5pc_nonzero = cvae_loadings_df_test_top10_cat.query("(Category=='Socioeconomic') and (E != 0)")['E']



ses_unique_to_compA = compA_ses_5pc_nonzero.index.difference(compB_ses_5pc_nonzero.index).difference(compD_ses_5pc_nonzero.index).difference(compE_ses_5pc_nonzero.index)
ses_unique_to_compB = compB_ses_5pc_nonzero.index.difference(compA_ses_5pc_nonzero.index).difference(compD_ses_5pc_nonzero.index).difference(compE_ses_5pc_nonzero.index)
ses_unique_to_compD = compD_ses_5pc_nonzero.index.difference(compA_ses_5pc_nonzero.index).difference(compB_ses_5pc_nonzero.index).difference(compE_ses_5pc_nonzero.index)
ses_unique_to_compE = compE_ses_5pc_nonzero.index.difference(compA_ses_5pc_nonzero.index).difference(compB_ses_5pc_nonzero.index).difference(compD_ses_5pc_nonzero.index)
#link in with dataframe that contains descriptions and nda_instrument (study name)


#component A
#return subset of dataframe with only socioeconomic variables, and convert to absolute value

socioeconomic_loadings_thresholded_dim30_abs = cvae_loadings_df_test_thresholded_category_desc[['Latent_Dim30','description']][cvae_loadings_df_test_thresholded_category_desc.category == 'Socioeconomic']
socioeconomic_loadings_thresholded_dim30_abs['Latent_Dim30'] = socioeconomic_loadings_thresholded_dim30_abs['Latent_Dim30'].abs()

#find indices where variable is nonzero and sort those loadings from greatest to least
nonzero_indices = np.nonzero(socioeconomic_loadings_thresholded_dim30_abs.Latent_Dim30.values)
socioeconomic_loadings_thresholded_dim30_abs_nonzero = socioeconomic_loadings_thresholded_dim30_abs.iloc[nonzero_indices].sort_values(by='Latent_Dim30', ascending=False)

#return subset of original df with effect values at indices discovered (nonzero loadings, category specific sorted by magnitude)
socioeconomic_loadings_thresholded_dim30_nonzero = cvae_loadings_df_test_thresholded_category_desc.loc[socioeconomic_loadings_thresholded_dim30_abs_nonzero.index][['description','Latent_Dim30',]]

#save to csv for supplementary table 4
socioeconomic_loadings_thresholded_dim30_nonzero.to_csv(f'{save_dir}/socioeconomic_compA_thresholded5_value_desc.csv')


#component B
#return subset of dataframe with only socioeconomic variables, and convert to absolute value

socioeconomic_loadings_thresholded_dim80_abs = cvae_loadings_df_test_thresholded_category_desc[['Latent_Dim80','description']][cvae_loadings_df_test_thresholded_category_desc.category == 'Socioeconomic']
socioeconomic_loadings_thresholded_dim80_abs['Latent_Dim80'] = socioeconomic_loadings_thresholded_dim80_abs['Latent_Dim80'].abs()

#find indices where variable is nonzero and sort those loadings from greatest to least

nonzero_indices = np.nonzero(socioeconomic_loadings_thresholded_dim80_abs.Latent_Dim80.values)
socioeconomic_loadings_thresholded_dim80_abs_nonzero = socioeconomic_loadings_thresholded_dim80_abs.iloc[nonzero_indices].sort_values(by='Latent_Dim80', ascending=False)

#return subset of original df with effect values at indices discovered (nonzero loadings, category specific sorted by magnitude)
socioeconomic_loadings_thresholded_dim80_nonzero = cvae_loadings_df_test_thresholded_category_desc.loc[socioeconomic_loadings_thresholded_dim80_abs_nonzero.index][['description','Latent_Dim80',]]

#save to csv 
socioeconomic_loadings_thresholded_dim80_nonzero.to_csv(f'{save_dir}/socioeconomic_compB_thresholded5_value_desc.csv')


#component D
#return subset of dataframe with only socioeconomic variables, and convert to absolute value

socioeconomic_loadings_thresholded_dim57_abs = cvae_loadings_df_test_thresholded_category_desc[['Latent_Dim57','description']][cvae_loadings_df_test_thresholded_category_desc.category == 'Socioeconomic']
socioeconomic_loadings_thresholded_dim57_abs['Latent_Dim57'] = socioeconomic_loadings_thresholded_dim57_abs['Latent_Dim57'].abs()

#find indices where variable is nonzero and sort those loadings from greatest to least

nonzero_indices = np.nonzero(socioeconomic_loadings_thresholded_dim57_abs.Latent_Dim57.values)
socioeconomic_loadings_thresholded_dim57_abs_nonzero = socioeconomic_loadings_thresholded_dim57_abs.iloc[nonzero_indices].sort_values(by='Latent_Dim57', ascending=False)

#return subset of original df with effect values at indices discovered (nonzero loadings, category specific sorted by magnitude)
socioeconomic_loadings_thresholded_dim57_nonzero = cvae_loadings_df_test_thresholded_category_desc.loc[socioeconomic_loadings_thresholded_dim57_abs_nonzero.index][['description','Latent_Dim57',]]

#save to csv 
socioeconomic_loadings_thresholded_dim57_nonzero.to_csv(f'{save_dir}/socioeconomic_compD_thresholded5_value_desc.csv')


#component E
#return subset of dataframe with only socioeconomic variables, and convert to absolute value

socioeconomic_loadings_thresholded_dim93_abs = cvae_loadings_df_test_thresholded_category_desc[['Latent_Dim93','description']][cvae_loadings_df_test_thresholded_category_desc.category == 'Socioeconomic']
socioeconomic_loadings_thresholded_dim93_abs['Latent_Dim93'] = socioeconomic_loadings_thresholded_dim93_abs['Latent_Dim93'].abs()

#find indices where variable is nonzero and sort those loadings from greatest to least

nonzero_indices = np.nonzero(socioeconomic_loadings_thresholded_dim93_abs.Latent_Dim93.values)
socioeconomic_loadings_thresholded_dim93_abs_nonzero = socioeconomic_loadings_thresholded_dim93_abs.iloc[nonzero_indices].sort_values(by='Latent_Dim93', ascending=False)

#return subset of original df with effect values at indices discovered (nonzero loadings, category specific sorted by magnitude)
socioeconomic_loadings_thresholded_dim93_nonzero = cvae_loadings_df_test_thresholded_category_desc.loc[socioeconomic_loadings_thresholded_dim93_abs_nonzero.index][['description','Latent_Dim93',]]

#save to csv 
socioeconomic_loadings_thresholded_dim93_nonzero.to_csv(f'{save_dir}/socioeconomic_compE_thresholded5_value_desc.csv')

#save ses indicators unique to each of the 4 SES focused components to csv 
socioeconomic_loadings_thresholded_dim30_nonzero.loc[ses_unique_to_compA.values,:].to_csv(f'{save_dir}/socioeconomic_compA_unique.csv')
socioeconomic_loadings_thresholded_dim80_nonzero.loc[ses_unique_to_compB.values,:].to_csv(f'{save_dir}/socioeconomic_compB_unique.csv')
socioeconomic_loadings_thresholded_dim57_nonzero.loc[ses_unique_to_compD.values,:].to_csv(f'{save_dir}/socioeconomic_compD_unique.csv')
socioeconomic_loadings_thresholded_dim93_nonzero.loc[ses_unique_to_compE.values,:].to_csv(f'{save_dir}/socioeconomic_compE_unique.csv')

#venn diagram figure
comp_ses_dict = {
    'A': set(compA_ses_5pc_nonzero.index),
    'B': set(compB_ses_5pc_nonzero.index),
    'D': set(compD_ses_5pc_nonzero.index),
    'E': set(compE_ses_5pc_nonzero.index)}

comp_keys = []
comp_vals = []

for key, value in comp_ses_dict.items():
    #print value
    #print(key, len([item for item in value if item]))
    comp_keys.append(key)
    comp_vals.append(len([item for item in value if item]))
    
#plot with count of SES measures retained after 5% thresholding

comp_colors = ["#e8a0ff", "#00adef", "#99c20e", "#ffa866"]

ses_venn = venn(comp_ses_dict, cmap = comp_colors)

plt.rcParams.update(plt.rcParamsDefault)
params = {'legend.fontsize': 14}
plt.rcParams.update(params)

#.set_title('Socioeconomic variable overlap across components A, B, D, E after top 5% thresholding')
plt.title('Components', loc='center', size = 16)
#plt.xlabel('Overlap in Driving SES Phenotypes (Total = 202)', loc='center', size = 20)
plt.xlabel(r'$\mathrm{Limited\ Overlap\ in\ Driving\ SES\ Phenotypes}$' + '\n' + r'$\mathrm{(Total\ count = 202)}$', loc='center', size = 20) 
# Add counts to the legend
legend_labels = [f"{component} ({count})" for component, count in zip(comp_keys, comp_vals)]
plt.legend(legend_labels, loc='upper center')


plt.savefig(f'{save_dir}/venn_ses_5pc_labelled.pdf', format='pdf', dpi = 300, bbox_inches='tight')


#for plotting loadings of phenotypes in components A, B, D, E
#preserve alphabetical color assignment

#keep same colors for each category across all plots

#alphabetical list of categories
alphabetical_cat_order = np.sort(category_order)

#create dictionary of colors for each category
#assign colors to each category alphabetically
preset_colors_dict = dict(zip(alphabetical_cat_order, get_colors(n_colors=23)))

#reorder dictionary of colors to match desired order of categories for manhattan plot
preset_colors_dict_ordered = {k : preset_colors_dict[k] for k in category_order}

manhattan_plot_colors = list(preset_colors_dict_ordered.values())


def manhattan_plot_CVAE_loadings_abs(df, latent_dim, new_cat_name, 
                   thres=0.05, ylim=None, plot_height=7):
    """
    Create a manhattan plot based on info in df
    ylim changes the bounds of the plot
    ylim should be a tuple
    """
    # ylim should be a tuple
    # Find FDR Thres
    
    colors = manhattan_plot_colors #get_colors(n_colors=len(new_cat_name))

    t_df = df.groupby('cat_num')['i'].median()
    t_dfm = df.groupby('cat_num')['i'].max()[:-1]

    plot = sns.relplot(data=df, x='i', y=f'{latent_dim}', size= 1, alpha=1, edgecolor=None, #'k'
                        aspect=1.5, height=plot_height, hue='cat_num',
                        palette=colors, legend=None)

    plot.ax.set_ylabel('')
    plot.ax.set_xlabel('')

    plot.ax.set_xticks(t_df) #plot x ticks at median of category values
    plot.ax.set_xticklabels(new_cat_name, rotation=90, ha='center', fontsize = 12) #use category names as xaxis labels

    for xtick, color in zip(plot.ax.get_xticklabels(), colors): #color category names according to scatter color
        xtick.set_color(color)

    #remove y axis
    plot.ax.spines['left'].set_visible(False)

    [plt.axvline(x=xc, color='k', linestyle='--', linewidth=1, alpha = 0.4) for xc in t_dfm] #plot vertical lines between categories


    #plot.suptitle('')


    if ylim:
        plot.tight_layout()
        plot.set(ylim=ylim)
        plot.tight_layout()
        locs, labels = plt.yticks()
        plt.yticks([*locs],
                [*labels])

    else:
        locs, labels = plt.yticks()
        plt.yticks([*locs],
                [*labels])

   # Hide grid lines
    plot.ax.grid(False)

    plot.tight_layout()
    plt.savefig(f'{save_dir}/cvae_manhattan_thresh5_{latent_dim}_abs.pdf', format='pdf', dpi = 300, bbox_inches='tight')
    
    
#convert 0 entries to nan so they are not plotted
cvae_loadings_df_test_thresholded_nan = cvae_loadings_df_test_thresholded_category_desc.iloc[:,:100].replace(0, np.nan)
#include category information
cvae_loadings_df_test_thresholded_nan_category_desc = pd.concat((cvae_loadings_df_test_thresholded_nan, cvae_loadings_df_test_thresholded_category_desc.iloc[:,-5:]), axis = 1)
#convert to absolute value
cvae_loadings_df_test_thresholded_nan_category_desc_abs = pd.concat((cvae_loadings_df_test_thresholded_nan_category_desc.iloc[:,:-5].abs(), cvae_loadings_df_test_thresholded_nan_category_desc.iloc[:,-5:]), axis = 1)

#component A loadings
manhattan_plot_CVAE_loadings_abs(cvae_loadings_df_test_thresholded_nan_category_desc_abs, 'Latent_Dim30', category_order, ylim = (0,0.6))
#component B loadings
manhattan_plot_CVAE_loadings_abs(cvae_loadings_df_test_thresholded_nan_category_desc_abs, 'Latent_Dim80', category_order, ylim = (0,0.6))
#component D loadings
manhattan_plot_CVAE_loadings_abs(cvae_loadings_df_test_thresholded_nan_category_desc_abs, 'Latent_Dim57', category_order, ylim = (0,0.6))
#component E loadings
manhattan_plot_CVAE_loadings_abs(cvae_loadings_df_test_thresholded_nan_category_desc_abs, 'Latent_Dim93', category_order, ylim = (0,0.6))
#supplementary table 5: full list of phenotypes with category and description 
loadings_base_screen_6_1yr_category.iloc[:,5:-1].to_csv(f'{save_dir}/phenotypes_category_description.csv')
#supplementary table 6: overall phenotype rankings per component
#post 95th percentile thresholding
#just one table per component, with category info
#100 components
os.makedirs(f'{save_dir}/phenotypes_ranking_per_component', exist_ok = True)

for i in range(len(cvae_elbow_dims_names_series)):
    phenotype_ranking_comp_df = cvae_loadings_df_test_thresholded_category_desc.loc[cvae_loadings_df_test_thresholded_category_desc[cvae_elbow_dims_names_series.values[i]].abs().sort_values(ascending = False).index ,(cvae_elbow_dims_names_series.values[i], 'category', 'description')]
    
    phenotype_ranking_comp_df.to_csv(f'{save_dir}/phenotypes_ranking_per_component/phenotypes_ranking_{i+1}_{cvae_elbow_dims_names_series.values[i]}.csv')
#State prediction code

#link state with subject by site
reverse_encoded_site_column, site_labels_dict = new_reverse_one_hot_encoder(data.iloc[:,7:28])
abcd_site = reverse_encoded_site_column + 1

site_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] 
state = ['california', 'colorado', 'florida', 'oklahoma', 'southcarolina', 'oregon', 'newyork', 'california', 'california', 'california', 'florida', 'maryland', 'michigan', 'minnesota', 'pennsylvania' , 'utah' ,'vermont', 'wisconsin', 'virginia', 'missouri', 'connecticut', 'maine']

site_state_mapping = pd.DataFrame(list(zip(site_id, state)), columns =['site_id', 'state'])

data_with_site = data.copy()
data_with_site['site'] = abcd_site
data_with_site_state = pd.merge(data_with_site, site_state_mapping, how='left', left_on='site', right_on='site_id').set_index(data_with_site.index)

#calculate particpant scores using only SES variables

#filter to top 10 ranked components
cvae_loadings_df_test_top10 = cvae_loadings_df_test.loc[:,cvae_elbow_dims_names_series[:10]]
cvae_loadings_df_test_top10.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
#add category info
cvae_loadings_df_test_top10_category = pd.merge(cvae_loadings_df_test_top10, loadings_base_screen_6_1yr_category.category, how='left', left_index=True, right_index=True)
#keep only SES category variables and 4 key components
cvae_loadings_test_ses_a_b_d_e = cvae_loadings_df_test_top10_category[cvae_loadings_df_test_top10_category.category == 'Socioeconomic'].loc[:, ['A','B','D','E']]
data_ses = data.loc[:,cvae_loadings_test_ses_a_b_d_e.index]
#compute scores
cvae_scores_ses_a_b_d_e = data_ses.dot(cvae_loadings_test_ses_a_b_d_e)
#merge with state data
cvae_scores_ses_a_b_d_e_state = pd.merge(cvae_scores_ses_a_b_d_e, data_with_site_state.state, how='left', left_index=True, right_index=True)
#save subject scores with state info to csv
cvae_scores_ses_a_b_d_e_state.to_csv(f'{save_dir}/cvae_scores_ses_a_b_d_e_state.csv')

cvae_scores_ses_a_b_d_e_state.reset_index(inplace=True)

#with kfold
#train multiclass logistic regression to distinguish states by SES component scores

kf = KFold(n_splits=10)

X = cvae_scores_ses_a_b_d_e_state.iloc[:,1:-1]
Y = cvae_scores_ses_a_b_d_e_state.iloc[:,-1:]

#track classification accuracy
predicted_y = []
true_y = []
coef_matrices = []

#k-1 splits used as train in set, last split used as test set. Repeat k times with different splits used as test set
for train_index, test_index in kf.split(X): #Do this prediction problem based on using 10-fold (KFold) cross-validation
    train_ses_scores, test_ses_scores = X.iloc[train_index,:], X.iloc[test_index,:]
    train_state_labels, test_state_labels = Y.iloc[train_index,:], Y.iloc[test_index,:]


    #train model on train set scores
    logreg_clf_state_by_ses = LogisticRegression(random_state=0, penalty='none', multi_class='ovr').fit(train_ses_scores, train_state_labels)

    #determine which state corresponds to which row in coefficients matrix
    #order of these coefficients corresponds to the order of unique class labels encountered during the fitting process

    coef_matrix = logreg_clf_state_by_ses.coef_
    coef_matrix_df = pd.DataFrame(coef_matrix, columns = logreg_clf_state_by_ses.feature_names_in_, index= logreg_clf_state_by_ses.classes_)
    coef_matrices.append(coef_matrix_df)


    #test model on test set scores
    state_pred = logreg_clf_state_by_ses.predict(test_ses_scores)

    #accuracy (out of sample)
    test_set_prediction_accuracy = logreg_clf_state_by_ses.score(test_ses_scores, test_state_labels.state)

    #probability of assigned label
    prediction_prob = logreg_clf_state_by_ses.predict_proba(test_ses_scores)

    #train set accuracy (in sample)
    train_set_prediction_accuracy = logreg_clf_state_by_ses.score(train_ses_scores, train_state_labels.state)

    #retrieve predicted and true Y values
    predicted_y.append(state_pred)
    true_y.append(test_state_labels.state)

#concatenate all predicted and true y values into one array for each

predicted_y_flat = np.concatenate(predicted_y, axis=0)
true_y_flat = np.concatenate(true_y, axis=0)


#combine all coef matrices into one
coef_matrix_concat = pd.concat([coef_matrices[0], coef_matrices[1], coef_matrices[2], coef_matrices[3], coef_matrices[4], \
          coef_matrices[5], coef_matrices[6], coef_matrices[7], coef_matrices[8], coef_matrices[9]], axis = 0) 

coef_matrix_mean_df = coef_matrix_concat.groupby(coef_matrix_concat.index).mean()
coef_matrix_mean_df

#which component has highest absolute value coefficient for each state
#interpret as which component which most identifies that state

state_component_repr = coef_matrix_mean_df.abs().idxmax(axis=1)

#return largest magnitude value for each state
maxCol=lambda x: max(x.min(), x.max(), key=abs)
max_coef_per_state = coef_matrix_mean_df.apply(maxCol,axis=1)

#save state, component, and max coef to csv
state_component_repr_df = pd.DataFrame([state_component_repr, max_coef_per_state], index = ['component', 'coef']).T
state_component_repr_df.to_csv(f'{save_dir}/state_component_repr.csv')

#reset plotting parameters
plt.rcParams.update(plt.rcParamsDefault)

#plot component coefficients (for supplement)
plt.figure(figsize=(8, 8))
sns.heatmap(coef_matrix_mean_df, annot=True, fmt='.2f', cmap = 'coolwarm', center=0)
plt.xlabel('Component')
plt.ylabel('State')
plt.title('Component Coefficients')
plt.savefig(f'{save_dir}/logreg_ovr_ses_state_10kf_coef_matrix_mean_heatmap.pdf', format='pdf', dpi=300, bbox_inches='tight')



#plot confusion matrix of out of sample state prediction (for supplement)
states_alphabetical = ['california', 'colorado', 'connecticut', 'florida', 'maryland', 'michigan', 'minnesota', 'missouri', 'newyork', 'oklahoma', 'oregon', 'pennsylvania', 'southcarolina', 'utah', 'vermont', 'virginia', 'wisconsin']

# Create a confusion matrix
conf_matrix = confusion_matrix(true_y_flat, predicted_y_flat, labels=states_alphabetical)

conf_matrix_percent = conf_matrix/conf_matrix.sum(axis=1)[:, np.newaxis]
# Visualize the confusion matrix as a heatmap
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix_percent, annot=True, fmt='.2f', cmap='Blues', xticklabels=states_alphabetical, yticklabels=states_alphabetical, vmin = 0, vmax = 1)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(f'{save_dir}/logreg_ovr_ses_state_10kf_conf_matrix_percent.pdf', format='pdf', dpi=300, bbox_inches='tight')


#Density plots
import matplotlib.transforms as transforms
#density plots by state with no background
states = cvae_scores_ses_a_b_d_e_state.state.unique()

for us_state in states:
    # Create a density plot using Seaborn
    sns.set(style="white")
    sns.kdeplot(cvae_scores_ses_a_b_d_e_state[cvae_scores_ses_a_b_d_e_state.state == us_state], common_norm = True, fill=True, palette = comp_colors, legend = False)
    #common_norm = True scale each conditional density by the number of observations such that the total area under all densities sums to 1
    sns.despine(bottom=True, left = True)
    # Add labels and a title
    #plt.xlabel("Score")
    plt.ylabel("")
    #plt.title(f"SES Scores Density Plot {us_state}")

    plt.gca().set_xticklabels([]) 
    plt.gca().set_yticklabels([]) 

    plt.axvline(x = 0,      # Line on x = 0
           ymin = 0, # Bottom of the plot
           ymax = 1,
           color = 'black',
           alpha = 0.5) # Top of the plot
 
    plt.text(0, -0.0005, '0', color = 'black', ha='center', va='center')
    #mytrans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    """
    plt.arrow(x = 0,
           y = 0.000001,      # Line on x = 0
           dx = 0, # Bottom of the plot
           dy = 0.00000002,
           color = 'black',
           alpha = 0.5,head_width=0.1)
    """
    os.makedirs(f'{save_dir}/density_plots', exist_ok = True)
    # Show the plot
    plt.savefig(f'{save_dir}/density_plots/ses_scores_density_plot_state_{us_state}_transparent.pdf', format='pdf', dpi=300, transparent=True)
    
    
#US geoplotting figure
#import geopandas state plotting data
states = geopandas.read_file(load_us_geo_template)

#To make the map look a little more familiar lets reproject it's coordinates to **Mercator**.
states = states.to_crs("EPSG:3395")

#calculate mean projection value per state
cvae_scores_ses_a_b_d_e_state_mean = cvae_scores_ses_a_b_d_e_state.groupby('state').mean()

#only retain rows with ABCD populated states
abcd_states = states[states['NAME'].isin(['California', 'Florida','Maryland', 'Michigan','Minnesota', 'Missouri', 'New York', 'Oregon','Virginia', 'Wisconsin',  'Colorado','Connecticut', 'Oklahoma','South Carolina','Utah', 'Pennsylvania', 'Vermont'])]

#Alphabetize states, to match order of cvae_scores_ses_a_b_d_e_state_mean dataframe
states_sort=['California', 'Colorado', 'Connecticut', 'Florida',
       'Maryland', 'Michigan', 'Minnesota', 'Missouri', 'New York', 'Oklahoma', 'Oregon', 'Pennsylvania',
       'South Carolina', 'Utah', 'Vermont', 'Virginia','Wisconsin']

#initialize new column as state name column to map appropriate values

#link component A mean state projections
abcd_states['A_ses_overall'] = abcd_states['NAME']
abcd_states['A_ses_overall'].replace(states_sort, cvae_scores_ses_a_b_d_e_state_mean.A.values, inplace=True)
#link component B mean state projections
abcd_states['B_ses_overall'] = abcd_states['NAME']
abcd_states['B_ses_overall'].replace(states_sort, cvae_scores_ses_a_b_d_e_state_mean.B.values, inplace=True)
#link component D mean state projections
abcd_states['D_ses_overall'] = abcd_states['NAME']
abcd_states['D_ses_overall'].replace(states_sort, cvae_scores_ses_a_b_d_e_state_mean.D.values, inplace=True)
#link component E mean state projections
abcd_states['E_ses_overall'] = abcd_states['NAME']
abcd_states['E_ses_overall'].replace(states_sort, cvae_scores_ses_a_b_d_e_state_mean.E.values, inplace=True)

#link each state to component with largest magnitude coefficient
abcd_states['logreg_repr_state'] = abcd_states['NAME']
abcd_states['logreg_repr_state'].replace(states_sort, state_component_repr_df.component.values, inplace=True)

#create column with coefficient
abcd_states['logreg_repr_coef'] = abcd_states['NAME']
abcd_states['logreg_repr_coef'].replace(states_sort, state_component_repr_df.coef.values, inplace=True)

#absolute value col to just look at magnitude of driving component per state
abcd_states['logreg_repr_coef_abs'] = abcd_states['logreg_repr_coef'].abs()


#plot the US state figure colored by most representative component

#reset plotting parameters
plt.rcParams.update(plt.rcParamsDefault)
params = {'ytick.labelsize': 38}
plt.rcParams.update(params)

fig = plt.figure(figsize=(25,15)) 
ax = fig.add_subplot()

#label states
abcd_states.apply(lambda x: ax.annotate(text=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', fontsize=14),axis=1)

us_boundary_map = states.boundary.plot(ax=ax, color='Gray', linewidth=.4)

abcd_states[abcd_states.logreg_repr_state == 'A'].plot(ax=us_boundary_map, cmap='Purples', column='logreg_repr_coef_abs', vmin = 0, vmax = abcd_states.logreg_repr_coef_abs.max(), figsize=(12, 12), legend=False) #change legernd to 'True' for colorbars
abcd_states[abcd_states.logreg_repr_state == 'B'].plot(ax=us_boundary_map, cmap='Blues', column='logreg_repr_coef_abs', vmin = 0, vmax = abcd_states.logreg_repr_coef_abs.max(), figsize=(12, 12), legend=False)
abcd_states[abcd_states.logreg_repr_state == 'D'].plot(ax=us_boundary_map, cmap='Greens', column='logreg_repr_coef_abs', vmin = 0, vmax = abcd_states.logreg_repr_coef_abs.max(), figsize=(12, 12), legend=False)
abcd_states[abcd_states.logreg_repr_state == 'E'].plot(ax=us_boundary_map, cmap='Oranges', column='logreg_repr_coef_abs', vmin = 0, vmax = abcd_states.logreg_repr_coef_abs.max(), figsize=(12, 12), legend=False)

plt.title("Most representative component per state",size=15, weight='bold')

plt.axis("off")

plt.savefig(f'{save_dir}/us_geoplot_logreg_state_ses.pdf', format='pdf', transparent=True)