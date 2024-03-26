#Phase 2 Model tuning and validation
"""
This script carries out steps outlined in Methods section 'Design of deep learning model architecture'

-CVAE model adapted from https://github.com/theislab/trvaep (as utilized in Lotfollahi et al., 2020)
-training (80%), validation (10%), and test (10%) split datasets using StratifiedShuffleSplit from the scikit-learn package
-PCA fit for 25 iterations of solver random state to identify the range of possible MSE reconstruction performance values.
-Train 6 CVAE architectures across 25 different weight initialization random seed values
-Mean and standard deviation of MSE reconstruction performance were computed across iterations and compared to that of our six candidate CVAE models

Key output is best performing archtecture 5 state dictionary used for interpretation of model

Results used to create Table 2 Evaluated CVAE architectures (mean MSE reconstruction loss and standard deviation per architecture)

"""

import trvaep
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import os
import json
import matplotlib.pyplot as plt
import random
from trvaep.utils import sklearn_train_val_test_split, pca_input_data
from config import SAVE_DIRECTORY_PHASE2, SAVE_DIRECTORY_PHASE1

#directories to save and load files
save_dir = SAVE_DIRECTORY_PHASE2
os.makedirs(save_dir, exist_ok = True)

load_dir = SAVE_DIRECTORY_PHASE1
#load data
data = pd.read_csv(f'{load_dir}/baseline_screen_6_1yr_z_4_cleaned.csv', index_col= 'subjectkey')
numeric_df = pd.read_csv(f'{load_dir}/numeric_df', index_col= 'subjectkey')

#isolate all columns ids in data that are from numeric dataframe
numeric_data_col_idx = data.columns.isin(numeric_df.columns)

#split into numeric data using numeric column ids
numeric_data = data.iloc[:,numeric_data_col_idx]

#split into categorical data using non numeric column ids
categorical_data = data.iloc[:,~numeric_data_col_idx]

data = pd.concat([categorical_data, numeric_data], axis = 1)


#Split data into train, validation, and test split
train_data, validate_data, test_data, train_labels_column, validate_labels_column, test_labels_column,\
            train_data_df, validate_data_df, test_data_df, labels_dict  = sklearn_train_val_test_split(data) #split data, and get 1hot labels
#directory to log pca run data
logdir = f'{save_dir}/pca_seed_runs/'
os.makedirs(logdir, exist_ok = True)


#save pre-PCA input data
orig_train_data = train_data
orig_validate_data = validate_data


pca_100_train_loss_list = []
pca_100_val_loss_list = []
pca_100_test_loss_list = []

#PCA 25 random seeds
#loop through 25 random seed values
#generate unique seeds for use in PCA random_seed

#seeds = random.sample(range(0, 100), 25)
#pca seeds for repeatability
seeds = [32, 59, 75, 63, 40, 20, 25, 85, 98, 78, 29, 88, 65, 41, 90, 21, 14, 16, 11, 53, 99, 26, 94, 66, 68]

pca_dim = 100

#convert input data from ambient space to PCA latent space
for seed in seeds:
    train_data_one_pca, validate_data_one_pca, test_data_one_pca, pca_model_one_pca = pca_input_data(pca_dim, train_data, validate_data, test_data, seed)

    #reconstruction error on train set
    train_set_reconstructed = pca_model_one_pca.inverse_transform(train_data_one_pca)

    train_set_reconstructed_tensor = torch.Tensor(train_set_reconstructed)
    input_data_tensor = torch.Tensor(orig_train_data)

    loss = F.mse_loss(train_set_reconstructed_tensor, input_data_tensor, reduction="sum")
    loss = loss / input_data_tensor.shape[0]
    #print(loss)

    #append to list
    pca_100_train_loss_list.append(loss.item())


    #reconstruction error on val set
    val_set_reconstructed = pca_model_one_pca.inverse_transform(validate_data_one_pca)

    val_set_reconstructed_tensor = torch.Tensor(val_set_reconstructed)
    input_data_tensor = torch.Tensor(orig_validate_data)

    loss = F.mse_loss(val_set_reconstructed_tensor, input_data_tensor, reduction="sum")
    loss = loss / input_data_tensor.shape[0]
    #print(loss)

    #append to list
    pca_100_val_loss_list.append(loss.item())

    #reconstruction error on test set
    test_set_reconstructed = pca_model_one_pca.inverse_transform(test_data_one_pca)

    test_set_reconstructed_tensor = torch.Tensor(test_set_reconstructed)
    input_data_tensor = torch.Tensor(test_data)

    loss = F.mse_loss(test_set_reconstructed_tensor, input_data_tensor, reduction="sum")
    loss = loss / input_data_tensor.shape[0]
    #print(loss)

    #append to list
    pca_100_test_loss_list.append(loss.item())

#convert to numpy
pca_100_train_loss_list_np = np.array(pca_100_train_loss_list)
pca_100_val_loss_list_np = np.array(pca_100_val_loss_list)
pca_100_test_loss_list_np = np.array(pca_100_test_loss_list)

#compute mean
pca_100_train_loss_list_mean = np.mean(pca_100_train_loss_list_np)
pca_100_val_loss_list_mean = np.mean(pca_100_val_loss_list_np)
pca_100_test_loss_list_mean = np.mean(pca_100_test_loss_list_np)

#compute std
pca_100_train_loss_std = np.std(pca_100_train_loss_list_np)
pca_100_val_loss_std = np.std(pca_100_val_loss_list_np)
pca_100_test_loss_std = np.std(pca_100_test_loss_list_np)


#save lists of losses for all seeds

pca_cv_losses = pd.DataFrame([seeds, pca_100_train_loss_list, pca_100_val_loss_list, pca_100_test_loss_list],\
                            index = ['Seed', 'pca_100_train_loss_list', 'pca_100_val_loss_list', 'pca_100_test_loss_list'])
final_path = os.path.join(logdir, "pca_recon_loss.csv")
pca_cv_losses.T.to_csv(final_path)




#experiments for CVAE architectures 1, 2 and 3

#seeds = random.sample(range(0, 100), 25)

#seeds for standarch archs for repeatability
seeds = [82, 72, 84, 92, 97, 6, 83, 76, 26, 37, 20, 39, 32, 73, 95, 91, 50, 46, 30, 69, 56, 85, 48, 53, 27]

#abcd sites as labels
n_conditions = 21

#standard arch 1, 2, and 3
latent_dim = [100, 100, 100]
enc_dec_dim = [200, 400, 800]
arch_number = ['1','2', '3']

#number of columns without site labels as input shape
input_dim = data.shape[1] - n_conditions


#make a list for each arch to track train and val loss in latent/ambient space during model eval


eval_train_loss_arch_1 = []
eval_val_loss_arch_1 = []
eval_test_loss_arch_1 = []


eval_train_loss_arch_2 = []
eval_val_loss_arch_2 = []
eval_test_loss_arch_2 = []


eval_train_loss_arch_3 = []
eval_val_loss_arch_3 = []
eval_test_loss_arch_3 = []

#Save model statedict to log folder for cvae standard architecture runs
logdir = f'{save_dir}/cvae_standard_archs/'
os.makedirs(logdir, exist_ok=True)

i=1
for seed in seeds:
    #print("run#: ",i)
    #loop through each architecture
    for enc_dec, latent, arch_num in zip(enc_dec_dim, latent_dim, arch_number):
        
            
        torch.manual_seed(seed) #initialize weights of model
        #init model
        model = trvaep.CVAE(input_dim = input_dim, num_classes= n_conditions, #use 'input_d' for non-PCA, 'pca_d' for PCA input
                    encoder_layer_sizes=[enc_dec], decoder_layer_sizes=[enc_dec], latent_dim=latent, alpha=0.001, use_batch_norm=False,
                    dr_rate=0, use_mmd=False, beta=1, output_activation='linear', var_type_split= None, pca_input = False, pca_dim = None, \
                        top_std_cols = False, splitter = 'strat_shuffle', decoder_choice = 'split_op', cols_split_idx = 7971) 
    
        #print("arch: ", arch_num)
        #make trainer object

        trainer = trvaep.Trainer_v2(model, data, seed = seed)
        
        #batch size 256, early stopping patience set at 20
        trainer.train(n_epochs = 1000, batch_size = 256, early_patience=20)
        
    
        #save to unique directory for each run
        run_name = f'arch{arch_num}_seed_{seed}'

        #create subfolder for each run of this experiment
        os.makedirs(logdir+run_name, exist_ok=True)

        #save model parameters
        path_to_model = os.path.join(logdir, run_name+'/model.pt')
        torch.save(model.state_dict(), path_to_model)

        #save validation and training losses
        with open(os.path.join(logdir, run_name+'/losses.json'), 'w') as f:
            f.write(json.dumps(
                {
                    list(trainer.logs)[0]: list(trainer.logs.values())[0],
                    list(trainer.logs)[1]: list(trainer.logs.values())[1],
                    list(trainer.logs)[2]: list(trainer.logs.values())[2],
                    list(trainer.logs)[3]: list(trainer.logs.values())[3],
                    list(trainer.logs)[4]: list(trainer.logs.values())[4],
                    list(trainer.logs)[5]: list(trainer.logs.values())[5],
                    list(trainer.logs)[6]: list(trainer.logs.values())[6], #loss_train_batch
                    list(trainer.logs)[7]: list(trainer.logs.values())[7], #rec_loss_train_batch
                    list(trainer.logs)[8]: list(trainer.logs.values())[8], #loss_valid_batch
                    list(trainer.logs)[9]: list(trainer.logs.values())[9], #rec_loss_valid_batch
                    list(trainer.logs)[10]: list(trainer.logs.values())[10] #earlystop epoch number
                },
                indent=4,
            ))
        
        #loss plot code
        """
        path_to_losses = os.path.join(logdir, run_name+'/losses.json')

        with open(path_to_losses, 'r') as f:
            loss_data = json.load(f)

        fig = plt.figure(figsize=(10,8))
        plt.plot(range(1,len(loss_data['rec_loss_train'])+1),loss_data['rec_loss_train'], label='Training Loss')
        plt.plot(range(1,len(loss_data['rec_loss_valid'])+1),loss_data['rec_loss_valid'],label='Validation Loss')

        # find position of lowest validation loss
        minposs = loss_data['rec_loss_valid'].index(min(loss_data['rec_loss_valid']))+1 
        plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('MSE Reconstruction loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        #save fig
        path_to_loss_plot = os.path.join(logdir, run_name+'/loss_plot.pdf')

        plt.savefig(path_to_loss_plot, format='pdf')

        #flatten list os lists from batch logs
        loss_train_batch = [item for sublist in loss_data['loss_train_batch'] for item in sublist]
        rec_loss_train_batch = [item for sublist in loss_data['rec_loss_train_batch'] for item in sublist]

        loss_valid_batch = [item for sublist in loss_data['loss_valid_batch'] for item in sublist]
        rec_loss_valid_batch = [item for sublist in loss_data['rec_loss_valid_batch'] for item in sublist]

        #plot train reconstruction losses per batch

        fig = plt.figure(figsize=(10,8))
        plt.plot(range(1,len(rec_loss_train_batch)+1),rec_loss_train_batch, label='Reconstruction Training Loss')
        #plt.plot(range(1,len(loss_train_batch)+1),loss_train_batch, label='Overall Training Loss')

        plt.xlabel('batches')
        plt.ylabel('MSE Reconstruction loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        #save fig
        path_to_loss_plot = os.path.join(logdir, run_name+'/train_batch_loss_plot.pdf')

        plt.savefig(path_to_loss_plot, format='pdf')

        #plot validation reconstruction losses per batch

        fig = plt.figure(figsize=(10,8))
        #plt.plot(range(1,len(rec_loss_valid_batch)+1),rec_loss_valid_batch, label='Reconstruction Validation Loss')
        plt.plot(range(1,len(loss_valid_batch)+1),loss_valid_batch, label='Validation Loss')

        plt.xlabel('batches')
        plt.ylabel('MSE Reconstruction loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        #save fig
        path_to_loss_plot = os.path.join(logdir, run_name+'/validate_batch_loss_plot.pdf')

        plt.savefig(path_to_loss_plot, format='pdf')
        """

        

        #evaluation reconstruction loss

        #return validation and train data splits used
        #data in form of numpy arrays 
        dataset_train, dataset_valid, dataset_test = trainer.make_dataset_abcd()

        #need to pass whole validation split through model, get reconstructed output
        model.eval()
        recon_test, _, _ = model.forward(torch.Tensor(dataset_test.data), torch.Tensor(dataset_test.data_labels))

        recon_validation, _, _ = model.forward(torch.Tensor(dataset_valid.data), torch.Tensor(dataset_valid.data_labels))

        recon_train, _, _ = model.forward(torch.Tensor(dataset_train.data), torch.Tensor(dataset_train.data_labels))
        
        #compute test mse loss

        test_mse_loss = torch.nn.functional.mse_loss(recon_test, torch.Tensor(dataset_test.data), reduction="sum")
        test_mse_loss = test_mse_loss / recon_test.size(0)
        #print("test loss: ", test_mse_loss)

        #compute validate mse loss

        valid_mse_loss = torch.nn.functional.mse_loss(recon_validation, torch.Tensor(dataset_valid.data), reduction="sum")
        valid_mse_loss = valid_mse_loss / recon_validation.size(0)
        #print("validation loss: ", valid_mse_loss)

        #compute train mse loss 

        train_mse_loss = torch.nn.functional.mse_loss(recon_train, torch.Tensor(dataset_train.data), reduction="sum")
        train_mse_loss = train_mse_loss / recon_train.size(0)
        #print("train loss: ", train_mse_loss)

        #append to appropriate lists

        if arch_num == '1':
            eval_train_loss_arch_1.append(train_mse_loss.item())
            eval_val_loss_arch_1.append(valid_mse_loss.item())
            eval_test_loss_arch_1.append(test_mse_loss.item())

        if arch_num == '2':
            eval_train_loss_arch_2.append(train_mse_loss.item())
            eval_val_loss_arch_2.append(valid_mse_loss.item())
            eval_test_loss_arch_2.append(test_mse_loss.item())

        if arch_num == '3':
            eval_train_loss_arch_3.append(train_mse_loss.item())
            eval_val_loss_arch_3.append(valid_mse_loss.item())
            eval_test_loss_arch_3.append(test_mse_loss.item())

        #iterate run / seed number
        i+=1

#compute mean loss for each arch across seeds
#first convert to numpy

eval_train_loss_arch_1_np = np.array(eval_train_loss_arch_1)
eval_val_loss_arch_1_np = np.array(eval_val_loss_arch_1)
eval_test_loss_arch_1_np = np.array(eval_test_loss_arch_1)

eval_train_loss_arch_2_np = np.array(eval_train_loss_arch_2)
eval_val_loss_arch_2_np = np.array(eval_val_loss_arch_2)
eval_test_loss_arch_2_np = np.array(eval_test_loss_arch_2)

eval_train_loss_arch_3_np = np.array(eval_train_loss_arch_3)
eval_val_loss_arch_3_np = np.array(eval_val_loss_arch_3)
eval_test_loss_arch_3_np = np.array(eval_test_loss_arch_3)

#compute std 

eval_train_loss_arch_1_std = np.std(eval_train_loss_arch_1_np)
eval_val_loss_arch_1_std = np.std(eval_val_loss_arch_1_np)
eval_test_loss_arch_1_std = np.std(eval_test_loss_arch_1_np)

eval_train_loss_arch_2_std = np.std(eval_train_loss_arch_2_np)
eval_val_loss_arch_2_std = np.std(eval_val_loss_arch_2_np)
eval_test_loss_arch_2_std = np.std(eval_test_loss_arch_2_np)

eval_train_loss_arch_3_std = np.std(eval_train_loss_arch_3_np)
eval_val_loss_arch_3_std = np.std(eval_val_loss_arch_3_np)
eval_test_loss_arch_3_std = np.std(eval_test_loss_arch_3_np)

#compute mean

eval_train_loss_arch_1_mean = np.mean(eval_train_loss_arch_1_np)
eval_val_loss_arch_1_mean = np.mean(eval_val_loss_arch_1_np)
eval_test_loss_arch_1_mean = np.mean(eval_test_loss_arch_1_np)

eval_train_loss_arch_2_mean = np.mean(eval_train_loss_arch_2_np)
eval_val_loss_arch_2_mean = np.mean(eval_val_loss_arch_2_np)
eval_test_loss_arch_2_mean = np.mean(eval_test_loss_arch_2_np)

eval_train_loss_arch_3_mean = np.mean(eval_train_loss_arch_3_np)
eval_val_loss_arch_3_mean = np.mean(eval_val_loss_arch_3_np)
eval_test_loss_arch_3_mean = np.mean(eval_test_loss_arch_3_np)



#save csvs of eval losses for all archs and seeds

cvae_standard_weight_seed_losses_df = pd.DataFrame([seeds, eval_train_loss_arch_1, eval_val_loss_arch_1, eval_test_loss_arch_1, \
                                eval_train_loss_arch_2, eval_val_loss_arch_2, eval_test_loss_arch_2, \
                                eval_train_loss_arch_3, eval_val_loss_arch_3, eval_test_loss_arch_3],\
                            index = ['Seed', 'eval_train_loss_arch_1', 'eval_val_loss_arch_1', 'eval_test_loss_arch_1',\
                                'eval_train_loss_arch_2', 'eval_val_loss_arch_2', 'eval_test_loss_arch_2',\
                                'eval_train_loss_arch_3', 'eval_val_loss_arch_3', 'eval_test_loss_arch_3'])
final_path = os.path.join(logdir, "cvae_standard_weight_seed_losses.csv")
cvae_standard_weight_seed_losses_df.T.to_csv(final_path)




#experiments for CVAE architectures 4, 5, and 6

#seeds = random.sample(range(0, 100), 25)
#seeds for pca archs for repeatability
seeds =[23, 99, 95, 45, 21, 83, 8, 18, 14, 75, 85, 27, 4, 29, 94, 63, 93, 39, 7, 56, 66, 57, 76, 20, 38]


n_conditions = 21

#corresponds to arch 4 through 6 (PCA)
latent_dim = [100, 100, 100]
enc_dec_dim = [200, 150, 125]
arch_number = ['4', '5', '6']
pca_dim = [400, 200, 150]


#make a list for each arch to track train and val loss in latent/ambient space during model eval



eval_train_loss_arch_4 = []
eval_val_loss_arch_4 = []
eval_test_loss_arch_4 = []


eval_train_loss_arch_5 = []
eval_val_loss_arch_5 = []
eval_test_loss_arch_5 = []


eval_train_loss_arch_6 = []
eval_val_loss_arch_6 = []
eval_test_loss_arch_6 = []

#Save model statedict to log folder for cvae pre-pca architecture runs
logdir = f'{save_dir}/cvae_pca_archs/'
os.makedirs(logdir, exist_ok=True)
        
i=1
for seed in seeds:
    #print("run#: ",i)
    #loop through each architecture
    for enc_dec, latent, arch_num, pca_d in zip(enc_dec_dim, latent_dim, arch_number, pca_dim):

        torch.manual_seed(seed) #initialize weights of model
        #init model
        model = trvaep.CVAE(input_dim = pca_d, num_classes= n_conditions, #use 'input_d' for non-PCA, 'pca_d' for PCA input
                    encoder_layer_sizes=[enc_dec], decoder_layer_sizes=[enc_dec], latent_dim=latent, alpha=0.001, use_batch_norm=False,
                    dr_rate=0, use_mmd=False, beta=1, output_activation='linear', var_type_split= None, pca_input = True, pca_dim = pca_d, \
                        top_std_cols = False, splitter = 'strat_shuffle', decoder_choice = 'base', cols_split_idx = 7971)
    
        #print("arch: ", arch_num)
        #make trainer object

        trainer = trvaep.Trainer_v2(model, data, seed = seed)
        #batch size 256, early stopping patience set at 20
        trainer.train(n_epochs = 1000, batch_size = 256, early_patience=20)
        
        #save to unique directory for each run
        run_name = f'arch{arch_num}_seed_{seed}'

        #create subfolder for each run of this experiment
        os.makedirs(logdir+run_name, exist_ok=True)

        #save model parameters
        path_to_model = os.path.join(logdir, run_name+'/model.pt')
        torch.save(model.state_dict(), path_to_model)

        #save validation and training losses
        with open(os.path.join(logdir, run_name+'/losses.json'), 'w') as f:
            f.write(json.dumps(
                {
                    list(trainer.logs)[0]: list(trainer.logs.values())[0],
                    list(trainer.logs)[1]: list(trainer.logs.values())[1],
                    list(trainer.logs)[2]: list(trainer.logs.values())[2],
                    list(trainer.logs)[3]: list(trainer.logs.values())[3],
                    list(trainer.logs)[4]: list(trainer.logs.values())[4],
                    list(trainer.logs)[5]: list(trainer.logs.values())[5],
                    list(trainer.logs)[6]: list(trainer.logs.values())[6], #loss_train_batch
                    list(trainer.logs)[7]: list(trainer.logs.values())[7], #rec_loss_train_batch
                    list(trainer.logs)[8]: list(trainer.logs.values())[8], #loss_valid_batch
                    list(trainer.logs)[9]: list(trainer.logs.values())[9], #rec_loss_valid_batch
                    list(trainer.logs)[10]: list(trainer.logs.values())[10] #earlystop epoch number
                },
                indent=4,
            ))
        """
        #loss plot code
        path_to_losses = os.path.join(logdir, run_name+'/losses.json')

        with open(path_to_losses, 'r') as f:
            loss_data = json.load(f)

        fig = plt.figure(figsize=(10,8))
        plt.plot(range(1,len(loss_data['rec_loss_train'])+1),loss_data['rec_loss_train'], label='Training Loss')
        plt.plot(range(1,len(loss_data['rec_loss_valid'])+1),loss_data['rec_loss_valid'],label='Validation Loss')

        # find position of lowest validation loss
        minposs = loss_data['rec_loss_valid'].index(min(loss_data['rec_loss_valid']))+1 
        plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('MSE Reconstruction loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        #save fig
        path_to_loss_plot = os.path.join(logdir, run_name+'/loss_plot.pdf')

        plt.savefig(path_to_loss_plot, format='pdf')

        #flatten list os lists from batch logs
        loss_train_batch = [item for sublist in loss_data['loss_train_batch'] for item in sublist]
        rec_loss_train_batch = [item for sublist in loss_data['rec_loss_train_batch'] for item in sublist]

        loss_valid_batch = [item for sublist in loss_data['loss_valid_batch'] for item in sublist]
        rec_loss_valid_batch = [item for sublist in loss_data['rec_loss_valid_batch'] for item in sublist]

        #plot train reconstruction losses per batch

        fig = plt.figure(figsize=(10,8))
        plt.plot(range(1,len(rec_loss_train_batch)+1),rec_loss_train_batch, label='Reconstruction Training Loss')
        #plt.plot(range(1,len(loss_train_batch)+1),loss_train_batch, label='Overall Training Loss')

        plt.xlabel('batches')
        plt.ylabel('MSE Reconstruction loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        #save fig
        path_to_loss_plot = os.path.join(logdir, run_name+'/train_batch_loss_plot.pdf')

        plt.savefig(path_to_loss_plot, format='pdf')

        #plot validation reconstruction losses per batch

        fig = plt.figure(figsize=(10,8))
        #plt.plot(range(1,len(rec_loss_valid_batch)+1),rec_loss_valid_batch, label='Reconstruction Validation Loss')
        plt.plot(range(1,len(loss_valid_batch)+1),loss_valid_batch, label='Validation Loss')

        plt.xlabel('batches')
        plt.ylabel('MSE Reconstruction loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        #save fig
        path_to_loss_plot = os.path.join(logdir, run_name+'/validate_batch_loss_plot.pdf')

        plt.savefig(path_to_loss_plot, format='pdf')
        """
        
        #evaluation reconstruction loss for PCA input data

        #return validation and train data splits used
        #data in form of numpy arrays 
        dataset_train, dataset_valid, dataset_test, orig_train_data, orig_validate_data, orig_test_data, pca_model = trainer.make_dataset_abcd()

        #need to pass whole validation split through model, get reconstructed output
        model.eval()
        recon_test, _, _ = model.forward(torch.Tensor(dataset_test.data), torch.Tensor(dataset_test.data_labels))

        recon_validation, _, _ = model.forward(torch.Tensor(dataset_valid.data), torch.Tensor(dataset_valid.data_labels))

        recon_train, _, _ = model.forward(torch.Tensor(dataset_train.data), torch.Tensor(dataset_train.data_labels))


        #compute recon loss on CVAE input/output (in PCA latent space)
        test_mse_loss_latent = torch.nn.functional.mse_loss(recon_test, torch.Tensor(dataset_test.data), reduction="sum")
        test_mse_loss_latent = test_mse_loss_latent / recon_test.size(0)
        #print("test loss latent: ", test_mse_loss_latent)

        valid_mse_loss_latent = torch.nn.functional.mse_loss(recon_validation, torch.Tensor(dataset_valid.data), reduction="sum")
        valid_mse_loss_latent = valid_mse_loss_latent / recon_validation.size(0)
        #print("validation loss latent: ", valid_mse_loss_latent)

        train_mse_loss_latent = torch.nn.functional.mse_loss(recon_train, torch.Tensor(dataset_train.data), reduction="sum")
        train_mse_loss_latent = train_mse_loss_latent / recon_train.size(0)
        #print("train loss latent: ", train_mse_loss_latent)


        #inverse transform recon data
        train_set_reconstructed = pca_model.inverse_transform(recon_train.detach().numpy())
        validate_set_reconstructed = pca_model.inverse_transform(recon_validation.detach().numpy())
        test_set_reconstructed = pca_model.inverse_transform(recon_test.detach().numpy())

        #convert original and train data to tensors
        recon_train_data = torch.Tensor(train_set_reconstructed)
        original_train_data = torch.Tensor(orig_train_data)

        recon_validate_data = torch.Tensor(validate_set_reconstructed)
        original_validate_data = torch.Tensor(orig_validate_data)

        recon_test_data = torch.Tensor(test_set_reconstructed)
        original_test_data = torch.Tensor(orig_test_data)

        #compute test mse loss
        test_mse_loss = torch.nn.functional.mse_loss(recon_test_data, original_test_data, reduction="sum")
        test_mse_loss = test_mse_loss / recon_test_data.size(0)
        #print("test loss: ", test_mse_loss)

        #compute vaidate mse loss
        valid_mse_loss = torch.nn.functional.mse_loss(recon_validate_data, original_validate_data, reduction="sum")
        valid_mse_loss = valid_mse_loss / recon_validate_data.size(0)
        #print("validation loss: ", valid_mse_loss)

        #compute train mse loss 

        train_mse_loss = torch.nn.functional.mse_loss(recon_train_data, original_train_data, reduction="sum")
        train_mse_loss = train_mse_loss / recon_train_data.size(0)
        #print("train loss: ", train_mse_loss)
    

        #append to appropriate lists

        if arch_num == '4':
            eval_train_loss_arch_4.append(train_mse_loss.item())
            eval_val_loss_arch_4.append(valid_mse_loss.item())
            eval_test_loss_arch_4.append(test_mse_loss.item())

        if arch_num == '5':
            eval_train_loss_arch_5.append(train_mse_loss.item())
            eval_val_loss_arch_5.append(valid_mse_loss.item())
            eval_test_loss_arch_5.append(test_mse_loss.item())

        if arch_num == '6':
            eval_train_loss_arch_6.append(train_mse_loss.item())
            eval_val_loss_arch_6.append(valid_mse_loss.item())
            eval_test_loss_arch_6.append(test_mse_loss.item())

        #iterate run / seed
        i+=1

#compute mean loss for each arch across seeds
#first convert to numpy

eval_train_loss_arch_4_np = np.array(eval_train_loss_arch_4)
eval_val_loss_arch_4_np = np.array(eval_val_loss_arch_4)
eval_test_loss_arch_4_np = np.array(eval_test_loss_arch_4)

eval_train_loss_arch_5_np = np.array(eval_train_loss_arch_5)
eval_val_loss_arch_5_np = np.array(eval_val_loss_arch_5)
eval_test_loss_arch_5_np = np.array(eval_test_loss_arch_5)

eval_train_loss_arch_6_np = np.array(eval_train_loss_arch_6)
eval_val_loss_arch_6_np = np.array(eval_val_loss_arch_6)
eval_test_loss_arch_6_np = np.array(eval_test_loss_arch_6)

#compute std 

eval_train_loss_arch_4_std = np.std(eval_train_loss_arch_4_np)
eval_val_loss_arch_4_std = np.std(eval_val_loss_arch_4_np)
eval_test_loss_arch_4_std = np.std(eval_test_loss_arch_4_np)

eval_train_loss_arch_5_std = np.std(eval_train_loss_arch_5_np)
eval_val_loss_arch_5_std = np.std(eval_val_loss_arch_5_np)
eval_test_loss_arch_5_std = np.std(eval_test_loss_arch_5_np)

eval_train_loss_arch_6_std = np.std(eval_train_loss_arch_6_np)
eval_val_loss_arch_6_std = np.std(eval_val_loss_arch_6_np)
eval_test_loss_arch_6_std = np.std(eval_test_loss_arch_6_np)

#compute mean

eval_train_loss_arch_4_mean = np.mean(eval_train_loss_arch_4_np)
eval_val_loss_arch_4_mean = np.mean(eval_val_loss_arch_4_np)
eval_test_loss_arch_4_mean = np.mean(eval_test_loss_arch_4_np)

eval_train_loss_arch_5_mean = np.mean(eval_train_loss_arch_5_np)
eval_val_loss_arch_5_mean = np.mean(eval_val_loss_arch_5_np)
eval_test_loss_arch_5_mean = np.mean(eval_test_loss_arch_5_np)

eval_train_loss_arch_6_mean = np.mean(eval_train_loss_arch_6_np)
eval_val_loss_arch_6_mean = np.mean(eval_val_loss_arch_6_np)
eval_test_loss_arch_6_mean = np.mean(eval_test_loss_arch_6_np)


#save lists of ambient/latent eval losses for all archs and seeds

cvae_pca_weight_seed_losses_df = pd.DataFrame([seeds, eval_train_loss_arch_4, eval_val_loss_arch_4, eval_test_loss_arch_4, \
                                eval_train_loss_arch_5, eval_val_loss_arch_5, eval_test_loss_arch_5, \
                                eval_train_loss_arch_6, eval_val_loss_arch_6, eval_test_loss_arch_6],\
                            index = ['Seed', 'eval_train_loss_arch_4', 'eval_val_loss_arch_4', 'eval_test_loss_arch_4',\
                                'eval_train_loss_arch_5', 'eval_val_loss_arch_5', 'eval_test_loss_arch_5',\
                                'eval_train_loss_arch_6', 'eval_val_loss_arch_6', 'eval_test_loss_arch_6'])
final_path = os.path.join(logdir, "cvae_pca_weight_seed_losses.csv")
cvae_pca_weight_seed_losses_df.T.to_csv(final_path)
#import cvae architecture cross validation losses csvs

#standard archs
cvae_standard_weight_seed_losses = pd.read_csv(f'{save_dir}/cvae_standard_archs/cvae_standard_weight_seed_losses.csv', index_col = 0)

#pre-PCAd archs
cvae_pca_weight_seed_losses = pd.read_csv(f'{save_dir}/cvae_pca_archs/cvae_pca_weight_seed_losses.csv', index_col = 0)

#pca random seed archs
pca_cv_losses = pd.read_csv(f'{save_dir}/pca_seed_runs/pca_recon_loss.csv', index_col = 0)

pca_mean = pca_cv_losses.loc[:, 'pca_100_test_loss_list'].mean()
pca_std = pca_cv_losses.loc[:, 'pca_100_test_loss_list'].std()

cvae_arch1_mean = cvae_standard_weight_seed_losses.loc[:, 'eval_test_loss_arch_1'].mean()
cvae_arch1_std = cvae_standard_weight_seed_losses.loc[:, 'eval_test_loss_arch_1'].std()

cvae_arch2_mean = cvae_standard_weight_seed_losses.loc[:, 'eval_test_loss_arch_2'].mean()
cvae_arch2_std = cvae_standard_weight_seed_losses.loc[:, 'eval_test_loss_arch_2'].std()

cvae_arch3_mean = cvae_standard_weight_seed_losses.loc[:, 'eval_test_loss_arch_3'].mean()
cvae_arch3_std = cvae_standard_weight_seed_losses.loc[:, 'eval_test_loss_arch_3'].std()

cvae_pre_pca_arch4_mean = cvae_pca_weight_seed_losses.loc[:, 'eval_test_loss_arch_4'].mean()
cvae_pre_pca_arch4_std = cvae_pca_weight_seed_losses.loc[:, 'eval_test_loss_arch_4'].std()

cvae_pre_pca_arch5_mean = cvae_pca_weight_seed_losses.loc[:, 'eval_test_loss_arch_5'].mean()
cvae_pre_pca_arch5_std = cvae_pca_weight_seed_losses.loc[:, 'eval_test_loss_arch_5'].std()

cvae_pre_pca_arch6_mean = cvae_pca_weight_seed_losses.loc[:, 'eval_test_loss_arch_6'].mean()
cvae_pre_pca_arch6_std = cvae_pca_weight_seed_losses.loc[:, 'eval_test_loss_arch_6'].std()



#plot PCA, CVAE standard, CVAE PCA with error bars
#arch 5 out of sample (test) MSE across 25 random seeds
import matplotlib.pyplot as plt


# Create lists for the plot
methods = ['PCA', 'CVAE-1', 'CVAE-2', 'CVAE-3', 'CVAE-4', 'CVAE-5', 'CVAE-6']
x_pos = np.arange(len(methods))
MSE_mean = [pca_mean, cvae_arch1_mean, cvae_arch2_mean, cvae_arch3_mean, cvae_pre_pca_arch4_mean, cvae_pre_pca_arch5_mean, cvae_pre_pca_arch6_mean]
error = [pca_std, cvae_arch1_std, cvae_arch2_std, cvae_arch3_std, cvae_pre_pca_arch4_std, cvae_pre_pca_arch5_std, cvae_pre_pca_arch6_std]
error_2sd = [2*pca_std, 2*cvae_arch1_std, 2*cvae_arch2_std, 2*cvae_arch3_std, 2*cvae_pre_pca_arch4_std, 2*cvae_pre_pca_arch5_std, 2*cvae_pre_pca_arch6_std]

# Build the plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x_pos, MSE_mean, alpha=0.5)
ax.errorbar(x_pos, MSE_mean, yerr=error_2sd, fmt="o", alpha=0.5, ecolor='black', capsize=10, label = '2 SD')
ax.set_ylabel('MSE')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods)
ax.set_title('')
ax.legend(loc='upper right')
ax.yaxis.grid(True)
#plt.xlim(-0.5, 1.5)

# Save the figure and show
plt.tight_layout()

plt.savefig(f'{save_dir}/mse_arch_randomseed_scatterplot.pdf', bbox_inches="tight")

#plot PCA, CVAE standard, CVAE PCA with error bars
#arch 5 out of sample (test) MSE across 25 random seeds
import matplotlib.pyplot as plt
import os

# Create lists for the plot
methods = ['PCA', 'CVAE']
x_pos = np.arange(len(methods))
MSE_mean = [pca_mean, cvae_pre_pca_arch5_mean]
error = [pca_std, cvae_pre_pca_arch5_std]
error_2sd = [2*pca_std, 2*cvae_pre_pca_arch5_std]

# Build the plot
fig, ax = plt.subplots(figsize=(4, 3))
ax.scatter(x_pos, MSE_mean, alpha=0.5,label='_nolegend_')
ax.errorbar(x_pos, MSE_mean, yerr=error_2sd, fmt="o", alpha=0.5, ecolor='black', capsize=10, label = '2 SD')
ax.set_ylabel('MSE')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods)
ax.set_title('')
#ax.legend('2 std')
ax.yaxis.grid(True)
plt.xlim(-0.5, 1.5)
plt.legend(loc = 'upper right')
# Save the figure and show
plt.tight_layout()

plt.savefig(f'{save_dir}/mse_arch_randomseed_scatterplot_study_pipeline.pdf', bbox_inches="tight")


#final target model for interpretation, using best architecture

n_conditions = 21


#corresponds to arch 5
latent_dim = 100
enc_dec_dim = 150
arch_number = '5'
pca_dim = 200
seed = 0

#Save model statedict to log folder for cvae pre-pca architecture runs
logdir = f'{save_dir}/cvae_model_final/'
os.makedirs(logdir, exist_ok=True)
        

torch.manual_seed(seed) #initialize weights of model
#init model
model = trvaep.CVAE(input_dim = pca_dim, num_classes= n_conditions, #use 'input_d' for non-PCA, 'pca_d' for PCA input
            encoder_layer_sizes=[enc_dec_dim], decoder_layer_sizes=[enc_dec_dim], latent_dim=latent_dim, alpha=0.001, use_batch_norm=False,
            dr_rate=0, use_mmd=False, beta=1, output_activation='linear', var_type_split= None, pca_input = True, pca_dim = pca_dim, \
                top_std_cols = False, splitter = 'strat_shuffle', decoder_choice = 'base', cols_split_idx = 7971)

#print("arch: ", arch_num)
#make trainer object

trainer = trvaep.Trainer_v2(model, data, seed = seed)
#batch size 256, early stopping patience set at 20
trainer.train(n_epochs = 1000, batch_size = 256, early_patience=20)

#save to unique directory for each run
run_name = f'arch{arch_number}_seed_{seed}'

#create subfolder for each run of this experiment
os.makedirs(logdir+run_name, exist_ok=True)

#save model parameters
path_to_model = os.path.join(logdir, run_name+'/model.pt')
torch.save(model.state_dict(), path_to_model)

#save validation and training losses
with open(os.path.join(logdir, run_name+'/losses.json'), 'w') as f:
    f.write(json.dumps(
        {
            list(trainer.logs)[0]: list(trainer.logs.values())[0],
            list(trainer.logs)[1]: list(trainer.logs.values())[1],
            list(trainer.logs)[2]: list(trainer.logs.values())[2],
            list(trainer.logs)[3]: list(trainer.logs.values())[3],
            list(trainer.logs)[4]: list(trainer.logs.values())[4],
            list(trainer.logs)[5]: list(trainer.logs.values())[5],
            list(trainer.logs)[6]: list(trainer.logs.values())[6], #loss_train_batch
            list(trainer.logs)[7]: list(trainer.logs.values())[7], #rec_loss_train_batch
            list(trainer.logs)[8]: list(trainer.logs.values())[8], #loss_valid_batch
            list(trainer.logs)[9]: list(trainer.logs.values())[9], #rec_loss_valid_batch
            list(trainer.logs)[10]: list(trainer.logs.values())[10] #earlystop epoch number
        },
        indent=4,
    ))


path_to_losses = os.path.join(logdir, run_name+'/losses.json')

with open(path_to_losses, 'r') as f:
    loss_data = json.load(f)
"""
#loss plot code
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(loss_data['rec_loss_train'])+1),loss_data['rec_loss_train'], label='Training Loss')
plt.plot(range(1,len(loss_data['rec_loss_valid'])+1),loss_data['rec_loss_valid'],label='Validation Loss')

# find position of lowest validation loss
minposs = loss_data['rec_loss_valid'].index(min(loss_data['rec_loss_valid']))+1 
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('MSE Reconstruction loss')
plt.grid(True)
plt.legend()
plt.tight_layout()

#save fig
path_to_loss_plot = os.path.join(logdir, run_name+'/loss_plot.pdf')

plt.savefig(path_to_loss_plot, format='pdf')

#flatten list os lists from batch logs
loss_train_batch = [item for sublist in loss_data['loss_train_batch'] for item in sublist]
rec_loss_train_batch = [item for sublist in loss_data['rec_loss_train_batch'] for item in sublist]

loss_valid_batch = [item for sublist in loss_data['loss_valid_batch'] for item in sublist]
rec_loss_valid_batch = [item for sublist in loss_data['rec_loss_valid_batch'] for item in sublist]

#plot train reconstruction losses per batch

fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(rec_loss_train_batch)+1),rec_loss_train_batch, label='Reconstruction Training Loss')
#plt.plot(range(1,len(loss_train_batch)+1),loss_train_batch, label='Overall Training Loss')

plt.xlabel('batches')
plt.ylabel('MSE Reconstruction loss')
plt.grid(True)
plt.legend()
plt.tight_layout()

#save fig
path_to_loss_plot = os.path.join(logdir, run_name+'/train_batch_loss_plot.pdf')

plt.savefig(path_to_loss_plot, format='pdf')

#plot validation reconstruction losses per batch

fig = plt.figure(figsize=(10,8))
#plt.plot(range(1,len(rec_loss_valid_batch)+1),rec_loss_valid_batch, label='Reconstruction Validation Loss')
plt.plot(range(1,len(loss_valid_batch)+1),loss_valid_batch, label='Validation Loss')

plt.xlabel('batches')
plt.ylabel('MSE Reconstruction loss')
plt.grid(True)
plt.legend()
plt.tight_layout()

#save fig
path_to_loss_plot = os.path.join(logdir, run_name+'/validate_batch_loss_plot.pdf')

plt.savefig(path_to_loss_plot, format='pdf')
"""

#evaluation reconstruction loss for PCA input data

#return validation and train data splits used
#data in form of numpy arrays 
dataset_train, dataset_valid, dataset_test, orig_train_data, orig_validate_data, orig_test_data, pca_model = trainer.make_dataset_abcd()

#need to pass whole validation split through model, get reconstructed output
model.eval()
recon_test, _, _ = model.forward(torch.Tensor(dataset_test.data), torch.Tensor(dataset_test.data_labels))

recon_validation, _, _ = model.forward(torch.Tensor(dataset_valid.data), torch.Tensor(dataset_valid.data_labels))

recon_train, _, _ = model.forward(torch.Tensor(dataset_train.data), torch.Tensor(dataset_train.data_labels))


#compute recon loss on CVAE input/output (in PCA latent space)
test_mse_loss_latent = torch.nn.functional.mse_loss(recon_test, torch.Tensor(dataset_test.data), reduction="sum")
test_mse_loss_latent = test_mse_loss_latent / recon_test.size(0)
#print("test loss latent: ", test_mse_loss_latent)

valid_mse_loss_latent = torch.nn.functional.mse_loss(recon_validation, torch.Tensor(dataset_valid.data), reduction="sum")
valid_mse_loss_latent = valid_mse_loss_latent / recon_validation.size(0)
#print("validation loss latent: ", valid_mse_loss_latent)

train_mse_loss_latent = torch.nn.functional.mse_loss(recon_train, torch.Tensor(dataset_train.data), reduction="sum")
train_mse_loss_latent = train_mse_loss_latent / recon_train.size(0)
#print("train loss latent: ", train_mse_loss_latent)


#inverse transform recon data
train_set_reconstructed = pca_model.inverse_transform(recon_train.detach().numpy())
validate_set_reconstructed = pca_model.inverse_transform(recon_validation.detach().numpy())
test_set_reconstructed = pca_model.inverse_transform(recon_test.detach().numpy())

#convert original and train data to tensors
recon_train_data = torch.Tensor(train_set_reconstructed)
original_train_data = torch.Tensor(orig_train_data)

recon_validate_data = torch.Tensor(validate_set_reconstructed)
original_validate_data = torch.Tensor(orig_validate_data)

recon_test_data = torch.Tensor(test_set_reconstructed)
original_test_data = torch.Tensor(orig_test_data)

#compute test mse loss
test_mse_loss = torch.nn.functional.mse_loss(recon_test_data, original_test_data, reduction="sum")
test_mse_loss = test_mse_loss / recon_test_data.size(0)
#print("test loss: ", test_mse_loss)

#compute vaidate mse loss
valid_mse_loss = torch.nn.functional.mse_loss(recon_validate_data, original_validate_data, reduction="sum")
valid_mse_loss = valid_mse_loss / recon_validate_data.size(0)
#print("validation loss: ", valid_mse_loss)

#compute train mse loss 

train_mse_loss = torch.nn.functional.mse_loss(recon_train_data, original_train_data, reduction="sum")
train_mse_loss = train_mse_loss / recon_train_data.size(0)
#print("train loss: ", train_mse_loss)


#save evaluation metrics to text file
path_to_evaluation_metrics = os.path.join(logdir, run_name+'/eval_metrics.txt')

recons_loss = open(path_to_evaluation_metrics,"w")
recons_loss.write("training reconstruction loss (eval): "+ str(train_mse_loss.item())+"\n")
recons_loss.write("validation reconstruction loss (eval): "+ str(valid_mse_loss.item())+"\n")
recons_loss.write("test reconstruction loss (eval): "+ str(test_mse_loss.item())+"\n")
recons_loss.write("number of epochs: "+ str(loss_data['earlystop_epoch'])+"\n")

recons_loss.write("\n")
recons_loss.close()
