from random import shuffle

import numpy as np
import scanpy as sc
import torch
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split #remove their train_test_split function to use this
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit


def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True,
              n_top_genes=2000):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    adata_count = adata.copy()

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if n_top_genes > 0 and adata.shape[1] > n_top_genes:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        genes = adata.var['highly_variable']
        adata = adata[:, genes]
        adata_count = adata_count[:, genes]

    if normalize_input:
        sc.pp.scale(adata)

    if sparse.issparse(adata_count.X):
        adata_count.X = adata_count.X.A

    if sparse.issparse(adata.X):
        adata.X = adata.X.A

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata_count.copy()
    else:
        adata.raw = adata_count

    return adata


def train_test_split_source(adata, train_frac=0.85):
    train_size = int(adata.shape[0] * train_frac)
    indices = np.arange(adata.shape[0])
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    train_data = adata[train_idx, :]
    valid_data = adata[test_idx, :]

    return train_data, valid_data


def train_val_test_split_updated(data):
    #split df into groups by site
    #will get 21 splits
    site_df_dict = {}

    #col 7-29 are site1-21 one hot encoded
    for i in range(7,28):
        site_df_dict[i-6] = data[data.iloc[:,i] == 1]
    #split each site into train validate and test sets
    train_df_dict = {}
    validate_df_dict = {}
    test_df_dict = {}

    #first split into train and test
    for i in range(1,22):
        train_df_dict[i], test_df_dict[i] = train_test_split(site_df_dict[i], test_size=0.1, shuffle=True, random_state=0)

    for i in range(1,22):
        train_df_dict[i], validate_df_dict[i] = train_test_split(train_df_dict[i], test_size=0.11, shuffle=True, random_state=0)
    #concatentate all train sets back together and all test sets back together
    train_df = train_df_dict[1]

    for i in range(1, 21):
        train_df = pd.concat([train_df, train_df_dict[i+1]],axis=0)
    #concatentate all validation sets back together and all test sets back together
    validate_df = validate_df_dict[1]

    for i in range(1, 21):
        validate_df = pd.concat([validate_df, validate_df_dict[i+1]],axis=0)
    #concatentate all test sets back together and all test sets back together
    #now have training, validation,  and test sets with equal count of subjects per site
    test_df = test_df_dict[1]

    for i in range(1, 21):
        test_df = pd.concat([test_df, test_df_dict[i+1]],axis=0)


    #following is to set up data splits to feed into CVAE (remove columns acting as labels)
    #choose columns that act as labels in train set
    #site columns (col 7-28 are site1-21)
    train_data_labels = train_df.iloc[:,7:28]  #.to_numpy()

    #drop columns from data that are serving as labels
    train_data = train_df.drop(train_df.iloc[:,7:28], axis = 1)
    numpy_train_data = train_data.to_numpy()


    #choose columns that act as labels in validate set
    validate_data_labels = validate_df.iloc[:,7:28]  #.to_numpy()

    #drop columns from data that are serving as labels
    validate_data = validate_df.drop(validate_df.iloc[:,7:28], axis = 1)
    numpy_validate_data = validate_data.to_numpy()


    #choose columns that act as labels in test set
    test_data_labels = test_df.iloc[:,7:28]  #.to_numpy()

    #drop columns from data that are serving as labels
    test_data = test_df.drop(test_df.iloc[:,7:28], axis = 1)
    numpy_test_data = test_data.to_numpy()

    return numpy_train_data, numpy_validate_data, numpy_test_data, train_data_labels, validate_data_labels, test_data_labels,\
    train_data, validate_data, test_data

def train_val_test_split(data):
    #split df into groups by site
    #will get 22 splits
    site_df_dict = {}

    #col 7-29 are site1-22 one hot encoded
    for i in range(7,29):
        site_df_dict[i-6] = data[data.iloc[:,i] == 1]
    #split each site into train validate and test sets
    train_df_dict = {}
    validate_df_dict = {}
    test_df_dict = {}

    #first split into train and test
    for i in range(1,23):
        train_df_dict[i], test_df_dict[i] = train_test_split(site_df_dict[i], test_size=0.1, shuffle=True, random_state=0)

    for i in range(1,23):
        train_df_dict[i], validate_df_dict[i] = train_test_split(train_df_dict[i], test_size=0.11, shuffle=True, random_state=0)
    #concatentate all train sets back together and all test sets back together
    train_df = train_df_dict[1]

    for i in range(1, 22):
        train_df = pd.concat([train_df, train_df_dict[i+1]],axis=0)
    #concatentate all validation sets back together and all test sets back together
    validate_df = validate_df_dict[1]

    for i in range(1, 22):
        validate_df = pd.concat([validate_df, validate_df_dict[i+1]],axis=0)
    #concatentate all test sets back together and all test sets back together
    #now have training, validation,  and test sets with equal count of subjects per site
    test_df = test_df_dict[1]

    for i in range(1, 22):
        test_df = pd.concat([test_df, test_df_dict[i+1]],axis=0)


    #following is to set up data splits to feed into CVAE (remove columns acting as labels)
    #choose columns that act as labels in train set
    #site columns (col 7-29 are site1-22)
    train_data_labels = train_df.iloc[:,7:29]  #.to_numpy()

    #drop columns from data that are serving as labels
    train_data = train_df.drop(train_df.iloc[:,7:29], axis = 1)
    numpy_train_data = train_data.to_numpy()


    #choose columns that act as labels in validate set
    validate_data_labels = validate_df.iloc[:,7:29]  #.to_numpy()

    #drop columns from data that are serving as labels
    validate_data = validate_df.drop(validate_df.iloc[:,7:29], axis = 1)
    numpy_validate_data = validate_data.to_numpy()


    #choose columns that act as labels in test set
    test_data_labels = test_df.iloc[:,7:29]  #.to_numpy()

    #drop columns from data that are serving as labels
    test_data = test_df.drop(test_df.iloc[:,7:29], axis = 1)
    numpy_test_data = test_data.to_numpy()

    return numpy_train_data, numpy_validate_data, numpy_test_data, train_data_labels, validate_data_labels, test_data_labels,\
    train_data, validate_data, test_data

def sklearn_train_val_test_split_separate(data, one_hot_labels):

    #labels should be one hot encoded
    X = data

    #reverse one hot encode labels
    y, labels_dict = new_reverse_one_hot_encoder(one_hot_labels)

    #first split into train and test
    sss_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    sss_1.get_n_splits(X, y)
    for i, (interim_train_index, test_index) in enumerate(sss_1.split(X, y)):

        #interim train split data to feed into second split
        X_2 = X.iloc[interim_train_index]
        #interim train split labels to feed into second split
        y_2 = y.iloc[interim_train_index]
        #set test split as data indices produced from first split
        test_data = X.iloc[test_index]
        test_data_labels = y.iloc[test_index]
        
        numpy_test_data = test_data.to_numpy()

    #second split train split from split 1 into train and validate splits
    sss_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.11, random_state=0)
    sss_2.get_n_splits(X_2, y_2)
    for i, (train_index, val_index) in enumerate(sss_2.split(X_2, y_2)):

        #train and validate data splits with their labels
        train_data = X_2.iloc[train_index]
        train_data_labels = y_2.iloc[train_index]

        validate_data = X_2.iloc[val_index]
        validate_data_labels = y_2.iloc[val_index]

        numpy_train_data = train_data.to_numpy()
        numpy_validate_data = validate_data.to_numpy()

    return numpy_train_data, numpy_validate_data, numpy_test_data, train_data_labels, validate_data_labels, test_data_labels,\
    train_data, validate_data, test_data, labels_dict

def sklearn_train_val_test_split(data):

    #separate data from labels
    X = data.drop(data.iloc[:,7:28], axis = 1)

    #create labels one hot encoded
    one_hot_labels = data.iloc[:,7:28]

    #reverse one hot encode labels
    y, labels_dict = new_reverse_one_hot_encoder(one_hot_labels)

    #first split into train and test
    sss_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    sss_1.get_n_splits(X, y)
    for i, (interim_train_index, test_index) in enumerate(sss_1.split(X, y)):

        #interim train split data to feed into second split
        X_2 = X.iloc[interim_train_index]
        #interim train split labels to feed into second split
        y_2 = y.iloc[interim_train_index]
        #set test split as data indices produced from first split
        test_data = X.iloc[test_index]
        test_data_labels = y.iloc[test_index]
        
        numpy_test_data = test_data.to_numpy()

    #second split train split from split 1 into train and validate splits
    sss_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.11, random_state=0)
    sss_2.get_n_splits(X_2, y_2)
    for i, (train_index, val_index) in enumerate(sss_2.split(X_2, y_2)):

        #train and validate data splits with their labels
        train_data = X_2.iloc[train_index]
        train_data_labels = y_2.iloc[train_index]

        validate_data = X_2.iloc[val_index]
        validate_data_labels = y_2.iloc[val_index]

        numpy_train_data = train_data.to_numpy()
        numpy_validate_data = validate_data.to_numpy()

    return numpy_train_data, numpy_validate_data, numpy_test_data, train_data_labels, validate_data_labels, test_data_labels,\
    train_data, validate_data, test_data, labels_dict

def new_reverse_one_hot_encoder(one_hot_data):
    single_column = one_hot_data.idxmax(1) 

    #create dicctionary linking each column name to site index number
    labels_dict = {}
    for i in range(0,len(one_hot_data.columns)):
        labels_dict[i] = one_hot_data.columns[i]

    #replace variable name in each row with associated numeric index from dict
    #replaces any row that has certain site string name with associated index number
    for key, value in labels_dict.items():
        single_column.replace([value],
                        [key], inplace=True)
    
    reverse_encoded_column = single_column

    return reverse_encoded_column, labels_dict

def reverse_one_hot_encoder(one_hot_data):

    #one_hot_data should be in form of pandas df

    single_column = one_hot_data.idxmax(1)

    #find unique values of single column
    sites = single_column.unique()

    #create indices from 0 to n-1 of classes
    #normalize labels
    indices = range(0, len(sites))

    #replace variable name in each row with associated numeric index
    #replaces any row that has certain site string name with associated index number
    for site, i in zip(sites,indices):

        single_column.replace([site],
                            [i], inplace=True)
        
    reverse_encoded_column = single_column
    
    return reverse_encoded_column

def pca_input_data(pca_dim, train_data, val_data, test_data, seed):

    """
    data input before creating CustomDataset class, no labels

    pca_dim = integer, number of components in PCA
    train_data = numpy_train_data output from data split
    val_data = numpy_validate_data output from data split
    test_data = numpy_test_data output from data split
    """

    #PCA the data
    pca_dim = pca_dim #change for different architectures


    pca_model = PCA(n_components=pca_dim, random_state=seed)
    #fit model with training data and apply dim reduction on training data
    pc_train = pca_model.fit_transform(train_data)


    #apply dim reduction on validate data
    pc_validate = pca_model.transform(val_data)


    #apply dim reduction on test data
    pc_test = pca_model.transform(test_data)

    return pc_train, pc_validate, pc_test, pca_model

def separate_pca_input_data(pca_dim, train_data_df, val_data_df, test_data_df, seed):

    #performs separate PCA of the continuous and discrete variables in data
    #concatenates the two parts back together before returning pca'd train/validate/test splits and pca_model

    #split train/val/test data into continuous and discrete parts
    numpy_train_data_split_num, numpy_train_data_split_cat = num_cat_data_split(train_data_df)
    numpy_val_data_split_num, numpy_val_data_split_cat = num_cat_data_split(val_data_df)
    numpy_test_data_split_num, numpy_test_data_split_cat = num_cat_data_split(test_data_df)

    #use half the number of components for each split
    dim = int(pca_dim/2)

    #create PCA model for use with continuous data and transform the continuous data 
    pc_train_num, pc_validate_num, pc_test_num, pca_model_num = pca_input_data(dim, numpy_train_data_split_num, numpy_val_data_split_num, numpy_test_data_split_num, seed)
    #create PCA model for use with discrete data  and transform the discrete data 
    pc_train_cat, pc_validate_cat, pc_test_cat, pca_model_cat = pca_input_data(dim, numpy_train_data_split_cat, numpy_val_data_split_cat, numpy_test_data_split_cat, seed)

    print('pca num train split shape',pc_train_num.shape)
    print('pca cat train split shape',pc_train_cat.shape)

    #concatentate the discrete+continuous PCAd data back together

    #train data
    pca_train_data = np.concatenate((pc_train_cat, pc_train_num),axis=1)

    #validate data
    pca_validate_data = np.concatenate((pc_validate_cat, pc_validate_num),axis=1)

    #test data
    pca_test_data = np.concatenate((pc_test_cat, pc_test_num),axis=1)

    return pca_train_data, pca_validate_data, pca_test_data, pca_model_num, pca_model_cat

    

def num_cat_data_split(data_split_df):
    #function to split pd data dataframe into two numpy arrays by numeric and categorical columns

    #Import numeric dataframe from heurstic split

    numeric_df = pd.read_csv('/Users/justin/Documents/BBME/ABCD/ABCD_Code/numeric_df', index_col= 'subjectkey')

    #isolate all columns ids in data that are from numeric dataframe
    numeric_data_col_idx = data_split_df.columns.isin(numeric_df.columns)

    #split into numeric data using numeric column ids
    numeric_data = data_split_df.iloc[:,numeric_data_col_idx]

    #split into categorical data using non numeric column ids
    categorical_data = data_split_df.iloc[:,~numeric_data_col_idx]

    #confirm addition of two splits equals original data (# of cols matches)
    assert categorical_data.shape[1]+numeric_data.shape[1] == data_split_df.shape[1]

    #convert from pandas df to numpy array as required for training
    numpy_data_split_num = numeric_data.to_numpy()
    numpy_data_split_cat = categorical_data.to_numpy()

    return numpy_data_split_num, numpy_data_split_cat

def high_var_cols(train_split_df, val_split_df, test_split_df):
    #function to retrieve only the 4746 (half training split # of observations) most highly varying columns
    half_num_samples = train_split_df.shape[0] // 2 #floor division, will be 4746

    #filter top 4747 highest std columns 
    high_std_cols = train_split_df.std().sort_values(ascending=False).index[:half_num_samples]
    
    #isolate all corresponding columns ids in data
    data_high_std_cols_idx = train_split_df.columns.isin(high_std_cols)

    #filter to keep only these high std columns in data
    train_high_std = train_split_df.iloc[:,data_high_std_cols_idx]

    val_high_std = val_split_df.iloc[:,data_high_std_cols_idx]

    test_high_std = test_split_df.iloc[:,data_high_std_cols_idx]

    #convert from pandas df to numpy array as required for training
    numpy_train_high_std = train_high_std.to_numpy()

    numpy_val_high_std = val_high_std.to_numpy()

    numpy_test_high_std = test_high_std.to_numpy()

    return numpy_train_high_std, numpy_val_high_std, numpy_test_high_std

def label_encoder(adata, label_encoder=None, condition_key='condition'):
    """
        Encode labels of Annotated `adata` matrix using sklearn.preprocessing.LabelEncoder class.
        Parameters
        ----------
        adata: `~anndata.AnnData`
            Annotated data matrix.
        Returns
        -------
        labels: numpy nd-array
            Array of encoded labels
        Example
        --------
        >>> import scanpy as sc
        >>> train_data = sc.read("./data/train.h5ad")
        >>> train_labels, label_encoder = label_encoder(train_data)
    """
    if label_encoder is None:
        le = LabelEncoder()
        labels = le.fit_transform(adata.obs[condition_key].tolist())
    else:
        le = label_encoder
        labels = np.zeros(adata.shape[0])
        for condition, label in label_encoder.items():
            labels[adata.obs[condition_key] == condition] = label
    return labels.reshape(-1, 1), le


def one_hot_encoder(idx, n_cls):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert torch.max(idx).item() < n_cls
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n_cls)
    onehot = onehot.to(device)
    onehot.scatter_(1, idx.long(), 1)
    return onehot


def remove_sparsity(adata):
    if sparse.issparse(adata.X):
        adata = sc.AnnData(adata.X.todense(), obs=adata.obs, varm=adata.varm,
                           var=adata.var)
    return adata


def create_dictionary(conditions, target_conditions=[]):
    if isinstance(target_conditions, list):
        target_conditions = [target_conditions]

    dictionary = {}
    conditions = [e for e in conditions if e not in target_conditions]
    for idx, condition in enumerate(conditions):
        dictionary[condition] = idx
    return dictionary


def shuffle_adata(adata):
    """
        Shuffles the `adata`.
        # Parameters
        adata: `~anndata.AnnData`
            Annotated data matrix.
        labels: numpy nd-array
            list of encoded labels
        # Returns
            adata: `~anndata.AnnData`
                Shuffled annotated data matrix.
            labels: numpy nd-array
                Array of shuffled labels if `labels` is not None.
        # Example
        ```python
        import scgen
        import anndata
        import pandas as pd
        train_data = anndata.read("./data/train.h5ad")
        train_labels = pd.read_csv("./data/train_labels.csv", header=None)
        train_data, train_labels = shuffle_data(train_data, train_labels)
        ```
    """
    adata = remove_sparsity(adata)

    ind_list = [i for i in range(adata.shape[0])]
    shuffle(ind_list)
    new_adata = adata[ind_list, :]
    return new_adata


def partition(data, partitions, num_partitions):
    res = []
    partitions = partitions.flatten()
    for i in range(num_partitions):
        res += [data[(partitions == i).nonzero().squeeze(1)]]
    return res

def cvae_pca_cross_corr(ambient_train_data, latent_variables, latent, X):
    #when PCA input to cvae model need to use orig_train_data, otherwise use dataset_train.data
   


    #pearson correlate each latent dim of CVAE with each input variable raw data
    #take the n values I have for a latent variable and correlate that with the n values for a specific input feature to get a representative "loading" for that feature in that latent variable

    # Compute the correlation matrix
    loadings_corr_matrix = np.corrcoef(ambient_train_data.T, latent_variables.T)

    # Extract the correlations between the latent variables and the input features
    #extracts bottom left quadrant
    latent_var_corrs = loadings_corr_matrix[-latent_variables.shape[1]:, :-latent_variables.shape[1]]


    cvae_loadings_df = pd.DataFrame(latent_var_corrs.T, columns = range(1,latent+1), index= X.columns)
    


    #PCA fit on training data

    pca_ = PCA(n_components=latent, random_state=0)
    #fit model with train data and apply dim reduction on train data
    pca_.fit(ambient_train_data)

    #transform
    pca_train_projections = pca_.transform(ambient_train_data)

    #pca component loadings df
    loadings_pca_df = pd.DataFrame(pca_.components_.T, columns=range(1,latent+1), index= X.columns)
    # Calculate the cross correlation coefficients between each PCA component and each CVAE latent variable
    #rows are PCA component loadings, cols are CVAE latent dim loadings


    corr_matrix_loadings = np.corrcoef(loadings_pca_df, cvae_loadings_df, rowvar=False)[:loadings_pca_df.shape[1], loadings_pca_df.shape[1]:]

    pc_list = []
    for i in range(1,latent+1):
        pc_list.append(f'PC{i}')

    corr_matrix_loadings_df = pd.DataFrame(corr_matrix_loadings, columns = range(1,latent+1), index= pc_list)

  

    #pearson correlate each latent variable of CVAE with data projected into PCA latent space
    #take the n values I have for a latent variable and correlate that with the n values for a PCA component

    # Compute the correlation matrix
    proj_corr_matrix = np.corrcoef(pca_train_projections.T, latent_variables.T)

    # Extract the correlations between the latent variables and the pca projections
    proj_corrs = proj_corr_matrix[:loadings_pca_df.shape[1], loadings_pca_df.shape[1]:] #[-latent_variables.shape[1]:, :-latent_variables.shape[1]]


    pc_list = []
    for i in range(1,latent+1):
        pc_list.append(f'PC{i}')

    #rows are PCA components, cols are cvae latent dims
    cvae_pca_proj_corr_df = pd.DataFrame(proj_corrs, columns = range(1,latent+1), index= pc_list)

    return cvae_loadings_df, loadings_pca_df, corr_matrix_loadings_df, cvae_pca_proj_corr_df


def cvae_pca_cross_corr_test(ambient_test_set, ambient_train_data, latent_variables, latent, X):
    #to be used on test set
    #when PCA input to model need to use orig_train_data, otherwise use dataset_train.data
   


    #pearson correlate each latent dim of CVAE with each input variable raw data
    #take the n values I have for a latent variable and correlate that with the n values for a specific input feature to get a representative "loading" for that feature in that latent variable

    # Compute the correlation matrix
    loadings_corr_matrix = np.corrcoef(ambient_test_set.T, latent_variables.T)

    # Extract the correlations between the latent variables and the input features
    #extracts bottom left quadrant
    latent_var_corrs = loadings_corr_matrix[-latent_variables.shape[1]:, :-latent_variables.shape[1]]


    cvae_loadings_df = pd.DataFrame(latent_var_corrs.T, columns = range(1,latent+1), index= X.columns)
    


    #PCA fit on training data

    pca_ = PCA(n_components=latent, random_state=0)
    #fit model with train data and apply dim reduction on train data
    pca_.fit(ambient_train_data)

    #transform
    pca_train_projections = pca_.transform(ambient_test_set)

    #pca component loadings df
    loadings_pca_df = pd.DataFrame(pca_.components_.T, columns=range(1,latent+1), index= X.columns)
    # Calculate the cross correlation coefficients between each PCA component and each CVAE latent variable
    #rows are PCA component loadings, cols are CVAE latent dim loadings


    corr_matrix_loadings = np.corrcoef(loadings_pca_df, cvae_loadings_df, rowvar=False)[:loadings_pca_df.shape[1], loadings_pca_df.shape[1]:]

    pc_list = []
    for i in range(1,latent+1):
        pc_list.append(f'PC{i}')

    corr_matrix_loadings_df = pd.DataFrame(corr_matrix_loadings, columns = range(1,latent+1), index= pc_list)

  

    #pearson correlate each latent variable of CVAE with data projected into PCA latent space
    #take the n values I have for a latent variable and correlate that with the n values for a PCA component

    # Compute the correlation matrix
    proj_corr_matrix = np.corrcoef(pca_train_projections.T, latent_variables.T)

    # Extract the correlations between the latent variables and the pca projections
    proj_corrs = proj_corr_matrix[:loadings_pca_df.shape[1], loadings_pca_df.shape[1]:] #[-latent_variables.shape[1]:, :-latent_variables.shape[1]]


    pc_list = []
    for i in range(1,latent+1):
        pc_list.append(f'PC{i}')

    #rows are PCA components, cols are cvae latent dims
    cvae_pca_proj_corr_df = pd.DataFrame(proj_corrs, columns = range(1,latent+1), index= pc_list)

    return cvae_loadings_df, loadings_pca_df, corr_matrix_loadings_df, cvae_pca_proj_corr_df

def cvae_pca_cross_corr_test_v2(ambient_test_set_with_site, ambient_train_data_with_site, latent_variables, latent, data):
    #to be used on test set
    #when PCA input to model need to use orig_train_data, otherwise use dataset_train.data
   


    #pearson correlate each latent dim of CVAE with each input variable raw data
    #take the n values I have for a latent variable and correlate that with the n values for a specific input feature to get a representative "loading" for that feature in that latent variable

    # Compute the correlation matrix
    loadings_corr_matrix = np.corrcoef(ambient_test_set_with_site.T, latent_variables.T)

    # Extract the correlations between the latent variables and the input features
    #extracts bottom left quadrant
    latent_var_corrs = loadings_corr_matrix[-latent_variables.shape[1]:, :-latent_variables.shape[1]]


    cvae_loadings_df = pd.DataFrame(latent_var_corrs.T, columns = range(1,latent+1), index= data.columns)
    


    #PCA fit on training data

    pca_ = PCA(n_components=latent, random_state=0)
    #fit model with train data and apply dim reduction on train data
    pca_.fit(ambient_train_data_with_site)

    #transform
    pca_train_projections = pca_.transform(ambient_test_set_with_site)

    #pca component loadings df
    loadings_pca_df = pd.DataFrame(pca_.components_.T, columns=range(1,latent+1), index= data.columns)
    # Calculate the cross correlation coefficients between each PCA component and each CVAE latent variable
    #rows are PCA component loadings, cols are CVAE latent dim loadings


    corr_matrix_loadings = np.corrcoef(loadings_pca_df, cvae_loadings_df, rowvar=False)[:loadings_pca_df.shape[1], loadings_pca_df.shape[1]:]

    pc_list = []
    for i in range(1,latent+1):
        pc_list.append(f'PC{i}')

    corr_matrix_loadings_df = pd.DataFrame(corr_matrix_loadings, columns = range(1,latent+1), index= pc_list)

  

    #pearson correlate each latent variable of CVAE with data projected into PCA latent space
    #take the n values I have for a latent variable and correlate that with the n values for a PCA component

    # Compute the correlation matrix
    proj_corr_matrix = np.corrcoef(pca_train_projections.T, latent_variables.T)

    # Extract the correlations between the latent variables and the pca projections
    proj_corrs = proj_corr_matrix[:loadings_pca_df.shape[1], loadings_pca_df.shape[1]:] #[-latent_variables.shape[1]:, :-latent_variables.shape[1]]


    pc_list = []
    for i in range(1,latent+1):
        pc_list.append(f'PC{i}')

    #rows are PCA components, cols are cvae latent dims
    cvae_pca_proj_corr_df = pd.DataFrame(proj_corrs, columns = range(1,latent+1), index= pc_list)

    return cvae_loadings_df, loadings_pca_df, corr_matrix_loadings_df, cvae_pca_proj_corr_df


def cvae_loadings(ambient_test_set_with_site, latent_variables, latent, data):
    #to be used on test set
    #when PCA input to model need to use orig_train_data, otherwise use dataset_train.data
   


    #pearson correlate each latent dim of CVAE with each input variable raw data
    #take the n values I have for a latent variable and correlate that with the n values for a specific input feature to get a representative "loading" for that feature in that latent variable

    # Compute the correlation matrix
    loadings_corr_matrix = np.corrcoef(ambient_test_set_with_site.T, latent_variables.T)

    # Extract the correlations between the latent variables and the input features
    #extracts bottom left quadrant
    latent_var_corrs = loadings_corr_matrix[-latent_variables.shape[1]:, :-latent_variables.shape[1]]


    cvae_loadings_df = pd.DataFrame(latent_var_corrs.T, columns = range(1,latent+1), index= data.columns)

    return cvae_loadings_df
