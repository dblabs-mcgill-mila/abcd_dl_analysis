import numpy as np
import torch
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from trvaep.utils import remove_sparsity


def label_encoder(adata, label_encoder=None, condition_key='condition'):
    if label_encoder is None:
        le = LabelEncoder()
        labels = le.fit_transform(adata.obs[condition_key].tolist())
    else:
        le = label_encoder
        labels = np.zeros(adata.shape[0])
        for condition, label in label_encoder.items():
            labels[adata.obs[condition_key] == condition] = label
    return labels.reshape(-1, 1), le


class CustomDatasetFromAdata(Dataset):
    def __init__(self, adata, condition_key=None):
        self.condtion_key = condition_key
        self.adata = adata
        if sparse.issparse(self.adata.X):
            self.adata = remove_sparsity(self.adata)
        self.data = np.array(self.adata.X)
        if self.condtion_key is not None:
            self.labels, self.le = label_encoder(self.adata, condition_key=condition_key)
            self.labels = np.array(self.labels)

    def __getitem__(self, index):
        if self.condtion_key is not None:
            single_cell_label = self.labels[index]
            label_as_tensor = torch.Tensor(single_cell_label)
        single_cell_expression = self.data[index, :]
        cell_as_tensor = torch.Tensor(single_cell_expression)
        if self.condtion_key is not None:
            return cell_as_tensor, label_as_tensor
        else:
            return cell_as_tensor, None

    def __len__(self):
        return len(self.adata)

    def get_label_ecnoder(self):
        return self.le
    
"""
def label_encoder_abcd(adata, label_encoder=None, condition_key='condition'):
    if label_encoder is None:
        le = LabelEncoder()
        labels = le.fit_transform(adata.obs[condition_key].tolist())
    else:
        le = label_encoder
        labels = np.zeros(adata.shape[0])
        for condition, label in label_encoder.items():
            labels[adata.obs[condition_key] == condition] = label
    return labels.reshape(-1, 1), le
"""

#data_labels should not be 1H encoded before inputing here
class CustomDataset(Dataset):
    def __init__(self, data_labels, data, use_labels = True, normalize_labels = True):

        #data_labels should not be single column of values (use reverse_one_hot_encoder in utils.py
        #use_labels set to True for CVAE and False for VAE
        #normalize_labels should be true if labels are not already normalized to be values from 0 to n-1 classes


        
        self.data = np.array(data)
        self.use_labels = use_labels
        if self.use_labels is not False:
            self.data_labels = np.array(data_labels)
        

        if self.use_labels is not False:
                self.le = LabelEncoder()
                if normalize_labels is True: #must set to true in order to use predict function
                    self.data_labels = self.le.fit_transform(self.data_labels) #normalize labels  (not sure if desired)
                    self.data_labels = self.data_labels.reshape(-1, 1)
                    self.data_labels = self.data_labels
                
                else:
                    self.data_labels = self.data_labels.reshape(-1, 1) #do not normalize

    def __len__(self):
        return len(self.data)
    #idx is the particular image id in this case, in mine it would be a specific subjectkey
    def __getitem__(self, idx):
        if self.use_labels is not False:
            label = self.data_labels[idx] #label corresponding to subject (numpy array)
            label_tensor = torch.Tensor(label)
        subject = self.data[idx, :] #subject is specific row of data
        subject_tensor = torch.Tensor(subject)
        indices = idx
        if self.use_labels is not False:
            return subject_tensor, label_tensor, indices
        else:
            return subject_tensor, None, indices
    
    
    def get_label_ecnoder(self):
        return self.le
    
