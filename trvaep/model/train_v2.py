from collections import defaultdict
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from trvaep.model._losses import MSE_kl, mmd
from trvaep.data_loader import CustomDatasetFromAdata, CustomDataset
from trvaep.utils import train_val_test_split_updated, new_reverse_one_hot_encoder, pca_input_data, num_cat_data_split, high_var_cols , \
sklearn_train_val_test_split, separate_pca_input_data, sklearn_train_val_test_split_separate #train_test_split,


class Trainer_v2:
    def __init__(self, model, adata, one_hot_labels = None,
                 condition_key="condition", seed=0, print_every=2000,
                 learning_rate=0.001, validation_itr=20, train_frac=0.85, n_workers=0):
        """
                trVAE Network class. This class contains the implementation of Regularized Conditional
                Variational Auto-encoder network.
                # Parameters
                    model: CVAE
                        a CVAE model object.
                    adata: `~anndata.AnnData`
                    `AnnData` object for training the model.

                    condition_key: str
                       The observation key in which data conditions are stored
                    seed: integer
                        Random seed for training initialization.

                    print_every= integer
                        How often print the loss values after, by default after every 1000 iterations.

                    learning_rate: float
                        Learning rate for the optimizer.

                    validation_itr: integer
                        How often print validation error, by default after every 5 epochs.

                    train_frac= float
                        Train-test split fraction. the model will be trained with train_frac for training
                        and 1-train_frac for validation.
                    n_workers= int
                        num of subsprocess for loading more batches for GPU. value bigger than 1 will require
                        more RAM and may increase the speed.


            """

        self.model = model
        self.adata = adata
        self.one_hot_labels = one_hot_labels
        self.condition_key = condition_key
        self.seed = seed
        self.print_loss = print_every
        self.lr = learning_rate
        self.val_check = validation_itr
        self.train_frac = train_frac
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.device = self.device
        self.logs = defaultdict(list)
        self.model.to(self.device)
        self.n_workers = n_workers

    """
    def make_dataset(self):
        train_adata, validation_adata = train_test_split(self.adata, self.train_frac)
        data_set_train = CustomDatasetFromAdata(train_adata, self.condition_key)
        data_set_valid = CustomDatasetFromAdata(validation_adata, self.condition_key)
        self.model.label_encoder = data_set_train.get_label_ecnoder()
        return data_set_train, data_set_valid
    """

    #to use abcd dataset
    def make_dataset_abcd(self):

        if self.model.splitter == 'manual':
            print('splitting data with manual function')
            train_data, validate_data, test_data, train_labels, validate_labels, test_labels,\
            train_data_df, validate_data_df, test_data_df  = train_val_test_split_updated(self.adata) #split data, and get 1hot labels

            #call reverse one hot encoder function here if using my manual train/val/test/split, done in splitter for sklearn StratifiedShuffleSplit
            train_labels_column = new_reverse_one_hot_encoder(train_labels) #convert 1 hot encoded labels to single column of values from 0 to n-1 classes
            validate_labels_column = new_reverse_one_hot_encoder(validate_labels) #repeat for validate set
        
        elif self.model.splitter == 'strat_shuffle':
            print('splitting data sklearn StratifiedShuffleSplit')
            train_data, validate_data, test_data, train_labels_column, validate_labels_column, test_labels_column,\
            train_data_df, validate_data_df, test_data_df, labels_dict  = sklearn_train_val_test_split(self.adata) #split data, and get 1hot labels
        
        elif self.model.splitter == 'strat_shuffle_separate':
            print('splitting data sklearn StratifiedShuffleSplit')
            train_data, validate_data, test_data, train_labels_column, validate_labels_column, test_labels_column,\
            train_data_df, validate_data_df, test_data_df, labels_dict  = sklearn_train_val_test_split_separate(self.adata, self.one_hot_labels) #split data, and get 1hot labels
            """
            if self.model.num_cls is None:
                train_labels_column = pd.DataFrame(np.full((train_labels_column.shape[0],1),None))
                validate_labels_column = pd.DataFrame(np.full((train_labels_column.shape[0],1),None))
                test_labels_column = pd.DataFrame(np.full((train_labels_column.shape[0],1),None))
            """
        #to use only highest variance columns (std) in data
        if self.model.top_std_cols is True:
            train_data, validate_data, test_data = high_var_cols(train_data_df, validate_data_df, test_data_df)
            print("train split shape: ",train_data.shape)
            print("val split shape: ",validate_data.shape)
            print("test split shape: ",test_data.shape)

        #to use only numeric or categorical variables in model
        if self.model.var_type_split == 'num' or self.model.var_type_split == 'cat':
            #split train/val/test splits into numeric and categorical components
            train_data_num, train_data_cat = num_cat_data_split(train_data_df)
            validate_data_num, validate_data_cat = num_cat_data_split(validate_data_df)
            test_data_num, test_data_cat = num_cat_data_split(test_data_df)

            #select whether to use numeric or categorical data split in model
            if self.model.var_type_split == 'num':
                #use only numeric variables in model
                train_data = train_data_num
                validate_data = validate_data_num
                test_data = test_data_num

            elif self.model.var_type_split == 'cat':
                #use only categorical variables in model
                train_data = train_data_cat
                validate_data = validate_data_cat
                test_data = test_data_cat

        
        #If pca_input flag is turned on, PCA the data before feeding into CVAE
        if self.model.pca_input is True:
            print("PCA fitting trian data and transforming train/val/test data")
            #save pre-PCA input data
            orig_train_data = train_data
            orig_validate_data = validate_data
            orig_test_data = test_data
            #convert input data from ambient space to PCA latent space
            if self.model.separate_pca is True:
                train_data, validate_data, test_data, pca_model_num, pca_model_cat = separate_pca_input_data(self.model.pca_dim, train_data_df, validate_data_df, test_data_df, self.seed)
            else:     
                train_data, validate_data, test_data, pca_model = pca_input_data(self.model.pca_dim, train_data, validate_data, test_data, self.seed)
            print("PCA train split shape: ",train_data.shape)
            print("PCA val split shape: ",validate_data.shape)
            print("PCA test split shape: ",test_data.shape)

        data_set_train = CustomDataset(train_labels_column, train_data)
        data_set_valid = CustomDataset(validate_labels_column, validate_data)
        data_set_test = CustomDataset(test_labels_column, test_data)

        #Feed PCA'd val data in as train set, pca'd train as valid
        #data_set_valid = CustomDataset(train_labels_column, train_data)
        #data_set_train = CustomDataset(validate_labels_column, validate_data)
        
        #if using VAE, don't create label encoder attribute for model
        if self.model.num_cls is not None:
            self.model.label_encoder = data_set_train.get_label_ecnoder() 

        #if PCA input, must return extra variables such as pre-PCAd train/val data, fit PCA model
        if self.model.pca_input is True:
            if self.model.separate_pca is True:
                return data_set_train, data_set_valid, data_set_test, orig_train_data, orig_validate_data, orig_test_data, pca_model_num, pca_model_cat
            else:
                return data_set_train, data_set_valid, data_set_test, orig_train_data, orig_validate_data, orig_test_data, pca_model
        #otherwise just return CustomDataset classes of data
        return data_set_train, data_set_valid, data_set_test
    


    def train_trvae(self, n_epochs=300, batch_size=512, early_patience=50):

        """
                    Trains a CVAE model `n_epochs` times with given `batch_size`. This function is using `early stopping`
                    technique to prevent overfitting.
                    # Parameters
                        n_epochs: int
                            number of epochs to iterate and optimize network weights
                        batch_size: int
                            number of samples to be used in each batch for network weights optimization
                        early_patience: int
                            number of consecutive epochs in which network loss is not going lower.
                            After this limit, the network will stop training.
                    # Returns
                        Nothing will be returned
                    # Example
                    ```python
                    adata = sc.read("./data/kang_seurat.h5ad")
                    n_conditions = adata.obs["condition"].unique().shape[0]
                    adata_train = adata[~((adata.obs["cell_type"] == "pDC")
                                          & (adata.obs["condition"] == "CTRL"))]
                    model = CVAE(adata_train.n_vars, num_classes=n_conditions,
                                 encoder_layer_sizes=[64], decoder_layer_sizes=[64], latent_dim=10, alpha=0.0001,
                                 use_mmd=True, beta=10)
                    trainer = Trainer(model, adata_train)
                    trainer.train_trvae(100, 64)
                    ```
        """
        es = EarlyStopping(patience=early_patience)
        dataset_train, dataset_valid = self.make_dataset_abcd()
        data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=self.n_workers)
        data_loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=self.n_workers)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.logs = defaultdict(list)
        self.model.train()
        for epoch in range(n_epochs):
            train_loss = 0
            train_rec = 0
            train_kl = 0
            train_mmd = 0
            for iteration, (x, y, idx) in enumerate(data_loader_train):
                if y is not None:
                    x, y = x.to(self.device), y.to(self.device)
                else:
                    x = x.to(self.device)

                recon_x, mean, log_var, y_mmd = self.model(x, y)
                vae_loss, reconstruction_loss, kl_loss = MSE_kl(recon_x, x, mean, log_var, self.model.alpha)
                mmd_calculator = mmd(self.model.num_cls, self.model.beta)
                mdd_loss = mmd_calculator(y_mmd, y)
                loss = vae_loss + mdd_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_rec += reconstruction_loss.item()
                train_kl += kl_loss.item()
                train_mmd += mdd_loss.item()
                if iteration % self.print_loss == 0 or iteration == len(data_loader_train) - 1:
                    print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss: {:9.4f}, "
                          "rec_loss: {:9.4f}, KL_loss: {:9.4f}, MMD_loss:  {:9.4f}".format(
                        epoch, n_epochs, iteration, len(data_loader_train) - 1,
                        loss.item(), reconstruction_loss.item(), kl_loss.item(), mdd_loss.item()))
            self.logs['loss_train'].append(train_loss / (iteration+1))
            self.logs["rec_loss_train"].append(train_rec / (iteration+1))
            self.logs["KL_loss_train"].append(train_kl / (iteration+1))
            self.logs["mmd_loss_train"].append(train_mmd / (iteration+1))
            valid_loss, valid_rec, valid_kl, valid_mmd = self.validate(data_loader_valid, use_mmd=True)
            self.logs['loss_valid'].append(valid_loss)
            self.logs["rec_loss_valid"].append(valid_rec)
            self.logs["KL_loss_valid"].append(valid_kl)
            self.logs["MMD_loss_valid"].append(valid_mmd)

            if es.step(valid_loss):
                print("Training stopped with early stopping")
                self.logs["earlystop_epoch"].append(epoch+1-early_patience)
                break

            if epoch % self.val_check == 0 and epoch != 0:
                print("Epoch {:02d}, Loss_valid: {:9.4f}, rec_loss_valid: {:9.4f},"
                      " KL_loss_valid: {:9.4f}, MMD_loss:  {:9.4f} ".format(
                    epoch, valid_loss, valid_rec, valid_kl, valid_mmd))
        self.model.eval()

    def train(self, n_epochs=100, batch_size=256, early_patience=15):

        """
                    Trains a CVAE model `n_epochs` times with given `batch_size`. This function is using `early stopping`
                    technique to prevent overfitting.
                    # Parameters
                        n_epochs: int
                            number of epochs to iterate and optimize network weights
                        batch_size: int
                            number of samples to be used in each batch for network weights optimization
                        early_patience: int
                            number of consecutive epochs in which network loss is not going lower.
                            After this limit, the network will stop training.
                    # Returns
                        Nothing will be returned
                    # Example
                    ```python
                    adata = sc.read("./data/kang_seurat.h5ad")
                    n_conditions = adata.obs["condition"].unique().shape[0]
                    adata_train = adata[~((adata.obs["cell_type"] == "pDC")
                                          & (adata.obs["condition"] == "CTRL"))]
                    model = CVAE(adata_train.n_vars, num_classes=n_conditions,
                                 encoder_layer_sizes=[64], decoder_layer_sizes=[64], latent_dim=10, alpha=0.0001,
                                 use_mmd=True, beta=10)
                    trainer = Trainer(model, adata_train)
                    trainer.train(100, 64)
                    ```
        """

        es = EarlyStopping(patience=early_patience)
        if self.model.pca_input is True:
            if self.model.separate_pca is True:
                dataset_train, dataset_valid, dataset_test, _, _, _, _, _ = self.make_dataset_abcd()
            else:
                dataset_train, dataset_valid, dataset_test, _, _, _, _ = self.make_dataset_abcd()
        else:
            dataset_train, dataset_valid, dataset_test = self.make_dataset_abcd()
        data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                        batch_size=batch_size,
                                                        shuffle=True)
        data_loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                        batch_size=batch_size,
                                                        shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.logs = defaultdict(list)

        self.model.train()
        for epoch in range(n_epochs):

            batch_loss_list_train = []
            batch_rec_loss_list_train = []

            train_loss = 0
            train_rec = 0
            train_kl = 0
            for iteration, (x, y, idx) in enumerate(data_loader_train):
                if y is not None:
                    x, y = x.to(self.device), y.to(self.device)
                else:
                    x = x.to(self.device)
                if self.model.num_cls is not None:
                    recon_x, mean, log_var = self.model(x, y)
                else:
                    recon_x, mean, log_var = self.model(x)
                loss, reconstruction_loss, kl_loss = MSE_kl(recon_x, x, mean, log_var, self.model.alpha)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                #log overall train losses per batch
                batch_loss_list_train.append(loss.item())
                batch_rec_loss_list_train.append(reconstruction_loss.item())

                train_rec += reconstruction_loss.item()
                train_kl += kl_loss.item()
                if iteration % self.print_loss == 0 or iteration == len(data_loader_train) - 1:
                    print(
                        "Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss: {:9.4f}, rec_loss: {:9.4f}, KL_loss: {:9.4f}".format(
                            epoch, n_epochs, iteration, len(data_loader_train) - 1,
                            loss.item(), reconstruction_loss.item(), kl_loss.item()))
   
            #epoch level losses
            self.logs['loss_train'].append(train_loss / (iteration+1))
            self.logs["rec_loss_train"].append(train_rec / (iteration+1))
            self.logs["KL_loss_train"].append(train_kl / (iteration+1))
            valid_loss, valid_rec, valid_kl , batch_loss_list_val, batch_rec_loss_list_val  = self.validate(data_loader_valid)
            self.logs['loss_valid'].append(valid_loss)
            self.logs["rec_loss_valid"].append(valid_rec)
            self.logs["KL_loss_valid"].append(valid_kl)

            #batch logs
            self.logs['loss_train_batch'].append(batch_loss_list_train)
            self.logs['rec_loss_train_batch'].append(batch_rec_loss_list_train)
            self.logs['loss_valid_batch'].append(batch_loss_list_val)
            self.logs['rec_loss_valid_batch'].append(batch_rec_loss_list_val)

            if es.step(valid_loss):
                print("Training stopped with early stopping")
                self.logs["earlystop_epoch"].append(epoch+1-early_patience)
                break

            if epoch % self.val_check == 0 and epoch != 0:
                print("Epoch {:02d}, Loss_valid: {:9.4f}, rec_loss_valid: {:9.4f}, KL_loss_valid: {:9.4f}".format(
                    epoch, valid_loss, valid_rec, valid_kl))
        self.model.eval()

    def validate(self, validation_data, use_mmd=False):
        """
                            Validat a CVAE model using  `validation_data`.
                            # Parameters
                                validation_data: `~anndata.AnnData`
                                    `AnnData` object for validating the model.
                                use_mmd: boolean
                                    If `True` the mmd loss wil be returned
                            # Returns
                                if `use_mmd` is `True` return following four `float`

                                valid_loss: float
                                    sum of all the losses

                                valid_rec: float
                                    reconstruction loss for the validation data

                                valid_kl: float
                                    KL loss for the validation data

                                valid_mmd: float
                                    MMD loss for validation data

                """
        self.model.eval()
        with torch.no_grad():
            valid_loss = 0
            valid_rec = 0
            valid_kl = 0
            valid_mmd = 0

            batch_loss_list_val = []
            batch_rec_loss_list_val = []

            for iteration, (x, y, idx) in enumerate(validation_data):
                if y is not None:
                    x, y = x.to(self.device), y.to(self.device)
                else:
                    x = x.to(self.device)
                if self.model.num_cls is not None:
                    if self.model.use_mmd:
                        recon_x, mean, log_var, y_mmd = self.model(x, y)
                    else:
                        recon_x, mean, log_var = self.model(x, y)
                else:
                    recon_x, mean, log_var = self.model(x)
                valid_vae_loss, reconstruction_loss, kl_loss = MSE_kl(recon_x, x, mean, log_var, self.model.alpha)
                if self.model.use_mmd:
                    mms_calculator = mmd(self.model.num_cls, 10)
                    valid_mmd = mms_calculator(y_mmd, y)
                if use_mmd:
                    valid_loss += valid_vae_loss.item() + valid_mmd.item()
                else:
                    valid_loss += valid_vae_loss.item()
                valid_rec += reconstruction_loss.item()
                valid_kl += kl_loss.item()
                if use_mmd:
                    valid_mmd += valid_mmd.item()

                #batch losses
                batch_loss_list_val.append(valid_vae_loss.item())
                batch_rec_loss_list_val.append(reconstruction_loss.item())
        self.model.train()
        #if iteration < 1:
        #    iteration = 1
        if use_mmd:
                return valid_loss / (iteration+1), valid_rec / (iteration+1), valid_kl / (iteration+1), valid_mmd / (iteration+1)
        else:
            return valid_loss / (iteration+1), valid_rec / (iteration+1), valid_kl / (iteration+1), batch_loss_list_val, \
                batch_rec_loss_list_val


# taken from https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                        best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                        best * min_delta / 100)
