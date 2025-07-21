#!/usr/bin/env python
# coding: utf-8
import csv
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn

import torch.optim as optim
import torch.nn.functional as F


from cyp_model_transformercpi_batch import *

import sys
sys.path.append('/home/yaganapu/CYP/cyp_update/benchmarks/')


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    
    device = torch.device('cpu')
    print('The code uses CPU!!!')
    
    
    
torch.cuda.empty_cache()

# Defining Constants
ATTEMPT = 4
iteration = 4
BATCH_SIZE = 64


TRAIN_PERCENTAGE = 0.70

print("\n"*2)
print("##########################################################")
print("PERCENTAGE OF POSITIVE SAMPLES USED FOR TRAINING: ", TRAIN_PERCENTAGE*100)
print("##########################################################")
print("\n"*2)

NUM_SAMPLES = 400
NUM_REPEATS = 20

NUM_EPOCHS = 1000

BASE_PATH = "/home/yaganapu/CYP/cyp_update/benchmarks/new_training_setting/iteration_training_for_bins/"
COMPOUND_DATA_PATH = "/home/yaganapu/CYP/cyp_update/benchmarks/bacteria_data/transformercpi/new_padded_compounds.npy"
ADJACENCY_DATA_PATH = "/home/yaganapu/CYP/cyp_update/benchmarks/bacteria_data/transformercpi/new_padded_adjacencies.npy"
PROTEIN_DATA_PATH = "/home/yaganapu/CYP/cyp_update/benchmarks/bacteria_data/transformercpi/new_padded_proteins.npy"
TARGET_DATA_PATH = "/home/yaganapu/CYP/cyp_update/benchmarks/bacteria_data/transformercpi/interactions.npy"


#--------------------------------Attempt_* FOLDER_CREATION--------------------------------#
DIR_NAME = BASE_PATH + '/phase1_new/iteration_' + str(iteration) +'/transformercpi/' + 'Attempt_' + str(ATTEMPT) + '/'
if(os.path.exists(DIR_NAME)):
    None
else:
    os.mkdir(DIR_NAME)
#--------------------------------Attempt_* FOLDER_CREATION--------------------------------#

#ATTEMPT = 30

# Loading the Preprocessed Datasets
smile_compound_data = np.load(COMPOUND_DATA_PATH, allow_pickle = True)
smile_adjacency_data = np.load(ADJACENCY_DATA_PATH, allow_pickle = True)
protein_data = np.load(PROTEIN_DATA_PATH, allow_pickle = True)
targets = torch.tensor(np.load(TARGET_DATA_PATH, allow_pickle = True)).float()


#--------------------------------SMILE_DATA_PREP--------------------------------#
print("*"*50)
print("Shape of Smile Data Before Reshaping: ", smile_compound_data.shape)


# Reshaping 3D data to 4D - Performing this step such that the input channels are 1.
tt_smile_compound_data = torch.tensor(smile_compound_data.reshape(smile_compound_data.shape[0],
                                                smile_compound_data.shape[1],
                                                smile_compound_data.shape[2])).float()

print("Shape of smile data after aligining with first CNN layer: ", tt_smile_compound_data.shape)


tt_smile_adjacency_data = torch.tensor(smile_adjacency_data.reshape(smile_adjacency_data.shape[0], 
                                                smile_adjacency_data.shape[1],
                                                smile_adjacency_data.shape[2])).float()
print("Shape of smile data after aligining with first CNN layer: ", tt_smile_adjacency_data.shape)


#--------------------------------PROTEIN_DATA_PREP--------------------------------#
print("*"*50)
print("Shape of protein data before reshaping: ", protein_data.shape)

tt_protein_data = torch.tensor(protein_data.reshape(protein_data.shape[0],  
                                                    protein_data.shape[1],
                                                    protein_data.shape[2])).float()

print("Shape of protein data after aligining with first CNN layer: ", tt_protein_data.shape)


NUM_TRAIN_SAMPLES = 400
#NUM_VAL_SAMPLES = 100
NUM_TEST_SAMPLES = 100

#--------------------------------TARGETS_DATA_PREP--------------------------------#
print("*"*50)

# Replace -1s in targets with 0s
targets[targets==-1]=0
print("Length of targets data: ", len(targets))
print("*"*50)
print("\n")
#--------------------------------TARGETS_DATA_PREP--------------------------------#



#--------------------------------POSITIVE_SAMPLES_SPLIT--------------------------------#
##-------------------------------TRAIN_AND_SPIES_SPLIT--------------------------------##
print("*"*50)

# Extract Positive Indices
#positive_indices = list(torch.where(targets==1)[0])

#NUM_SAMPLES = int(len(positive_indices)*TRAIN_PERCENTAGE)
# print("Total number of positive class samples: ", len(positive_indices))
print("-"*50)
print("Splitting the positive samples into training and spies sets")
print("Number of positive class samples in training set: ", NUM_TRAIN_SAMPLES)

# RANDOM SEED TAKES THE VALUE OF CURRENT ATTEMPT


# RANDOMLY SELECT - NUM_SAMPLES positives as training samples.
#train_pos_indices = np.random.choice(positive_indices, size=NUM_SAMPLES, replace=False)

train_randomized_data_path = '/home/yaganapu/CYP/cyp_update/benchmarks/phase1_data/iteration_' + str(iteration)+'/train_indices.npy'
val_randomized_data_path = '/home/yaganapu/CYP/cyp_update/benchmarks/phase1_data/iteration_' + str(iteration)+'/val_indices.npy'
test_randomized_data_path = '/home/yaganapu/CYP/cyp_update/benchmarks/phase1_data/iteration_' + str(iteration)+'/test_indices.npy'

train_pos_indices_1 = np.load(train_randomized_data_path, allow_pickle = True)
val_pos_indices_1 = np.load(val_randomized_data_path, allow_pickle = True)
train_pos_indices = np.concatenate((train_pos_indices_1, val_pos_indices_1))

print("Number of positive class samples as spies: ", len(train_pos_indices))
print("-"*50)


#val_pos_indices = np.load(val_randomized_data_path, allow_pickle = True)


train_pos_smiles_compound = tt_smile_compound_data[train_pos_indices]
train_pos_smiles_adjacency = tt_smile_adjacency_data[train_pos_indices]
train_pos_proteins = tt_protein_data[train_pos_indices]
train_pos_targets = targets[train_pos_indices]


# REMANING WILL BE spy samples
#spy_samples = np.setdiff1d(positive_indices, train_pos_indices)
spy_indices = np.load(test_randomized_data_path, allow_pickle = True)
#print(spy_indices)
#sys.exit()

print("Number of positive class samples as spies: ", len(spy_indices))
print("-"*50)

spy_smiles_compound = tt_smile_compound_data[spy_indices]
spy_smiles_adjacency = tt_smile_adjacency_data[spy_indices]
spy_proteins = tt_protein_data[spy_indices]
spy_targets = targets[spy_indices]



# # ###_________________THIS LOGIC WILL HELP IN IDENTIFYING SPIES IN TRAINING SET___________###

num_zeros_dim2 = spy_proteins.shape[1] - spy_smiles_compound.shape[1]
num_zeros_dim3 = spy_proteins.shape[2] - spy_smiles_compound.shape[2]
padded_spy_smiles = torch.nn.functional.pad(spy_smiles_compound, (0, num_zeros_dim3, 0, num_zeros_dim2))
all_spies = (padded_spy_smiles + spy_proteins)

print("-"*50)
print("padded_smiles_shape:", all_spies.shape)
print("-"*50)
# ###_________________THIS LOGIC WILL HELP IN IDENTIFYING SPIES IN TRAINING SET___________###


print("Number of positive samples in training set: ", len(train_pos_indices))
print("Shape Information of training set: ")
print("Smiles compound: ", train_pos_smiles_compound.shape)
print("Smiles adjacency: ", train_pos_smiles_adjacency.shape)
print("Proteins: ", train_pos_proteins.shape)
print("Targets: ", train_pos_targets.shape)
print("-"*50)

print("Number of spy samples: ", len(spy_indices))
print("Shape Information of spy set: ")
print("Smiles compound: ", spy_smiles_compound.shape)
print("Smiles adjacency: ", spy_smiles_adjacency.shape)
print("Proteins: ", spy_proteins.shape)
print("Targets: ", spy_targets.shape)
print("*"*50)
print("\n")
##-------------------------------TRAIN_AND_SPIES_SPLIT--------------------------------##
#--------------------------------POSITIVE_SAMPLES_SPLIT--------------------------------#

##-------------------------------TRAIN_AND_SPIES_SPLIT--------------------------------##
#--------------------------------POSITIVE_SAMPLES_SPLIT--------------------------------#



#--------------------COMBINE_SPIES_AND_UNLABELED_SAMPLES-----------------#
print("*"*50)

# Extract Unlabeled Indices
unlabeled_indices_e = list(torch.where(targets==0)[0])

unlabeled_and_spies_indices = [int(tensor) for tensor in unlabeled_indices_e] + list(spy_indices)
#print(unlabeled_and_spies_indices[:10])
np.random.seed(iteration+ATTEMPT)
np.random.shuffle(unlabeled_and_spies_indices)
#print(unlabeled_and_spies_indices[:10])
#sys.exit()



print("Number of unlabeled samples: ", len(unlabeled_indices_e))
print("-"*50)
print("Extracting unlabeled data... ")
# unlabeled_smiles_compound_e = tt_smile_compound_data[unlabeled_indices_e]
# unlabeled_smiles_adjacency_e = tt_smile_adjacency_data[unlabeled_indices_e]
# unlabeled_proteins_e = tt_protein_data[unlabeled_indices_e]
# unlabeled_targets_e = targets[unlabeled_indices_e]

print("Combining spy samples with unlabeled data... ")

# unlabeled_smiles_compound = torch.cat((unlabeled_smiles_compound_e, spy_smiles_compound))
# unlabeled_smiles_adjacency = torch.cat((unlabeled_smiles_adjacency_e, spy_smiles_adjacency))
# unlabeled_proteins = torch.cat((unlabeled_proteins_e, spy_proteins))
# unlabeled_targets = torch.cat((unlabeled_targets_e, spy_targets))

print("Shuffling unlabeled data... ")




#-------------------NEW LOGIC TO KEEP TRACK OF UNLABELED AND SPIES------------------------#
#shuffled_indices = torch.randperm(len(unlabeled_smiles_compound))


#random.shuffle(unlabeled_indices_e)

# Convert the shuffled list back to a tensor
#shuffled_indices = torch.tensor(indices_with_labels_list)




#print(shuffled_indices)
#sys.exit()


#this 5 lines are extra
# print("list(unlabeled_indices_e): ", [int(tensor) for tensor in unlabeled_indices_e][:10])
# print("list(spy_samples): ", list(spy_indices)[:10])

# unlabeled_and_spies_indices = [int(tensor) for tensor in unlabeled_indices_e] + list(spy_indices)
# unlabeled_and_spies_indices = [unlabeled_and_spies_indices[index] for index in shuffled_indices]

print("len(unlabeled_and_spies_indices): ", len(unlabeled_and_spies_indices))
#-------------------NEW LOGIC TO KEEP TRACK OF UNLABELED AND SPIES------------------------#




#print(len(shuffled_indices))
# unlabeled_smiles_compound = tt_smile_compound_data[unlabeled_and_spies_indices]
# unlabeled_smiles_adjacency = tt_smile_adjacency_data[unlabeled_and_spies_indices]
# unlabeled_proteins = tt_protein_data[unlabeled_and_spies_indices]
# unlabeled_targets = targets[unlabeled_and_spies_indices]

# mask = unlabeled_targets==1
# unlabeled_targets[mask]=0

# if(sum(unlabeled_targets==1)>0):
#     print("There are still targets of class 1 in unlabeled data")
#     #sys.exit()
    
# else:
#     print("no")
# #sys.exit()


print("-"*50)
#print("Number of unlabeled samples: ", len(unlabeled_smiles_compound))
print("*"*50)
print("\n")
#--------------------COMBINE_SPIES_AND_UNLABELED_SAMPLES-----------------#



train_predictions = {}
train_pos_predictions = {}
spy_predictions = {}
spy_predictions_indices = {}
unlabeled_predictions = {}
loss_values = {}
touched_spy_indices = {}

untouched_unalebed_final_without_spies={}
untouched_unalebed_final_indices_without_spies = {}

scr_values = {"exper_num": [], 
              "num_samples": [],
              "scr_value": []}

protein_dim = 100
atom_dim = 34
hid_dim = 64
n_layers = 3
n_heads = 8
pf_dim = 256
dropout = 0.1

kernel_size = 5

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print("device:", device)



for i in tqdm(range(NUM_REPEATS)):

    torch.cuda.empty_cache()
#print("Current Repeat: ", i+1)



#-----------INITIALIZE MODEL OBJECT-------------#
# Create an instance of the model

    encoder = Encoder(protein_dim, hid_dim, 3, kernel_size, dropout, device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder, decoder, device)

#model.load_state_dict(torch.load('/home/yaganapu/CYP/cyp_update/benchmarks/phase1/transformercpi/Attempt_0/my_model1.pth'))

# CUDA...
    model.to(device)
    model = nn.DataParallel(model)

    # LOSS
    criterion = nn.BCELoss()

    # OPTIMIZER
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    #-----------INITIALIZE MODEL OBJECT-------------#



    # Create an empty list to store unlabeled_predictions
    #unlabeled_predictions[i] = []

    # Create an empty list to store loss values...
    loss_values[i] = []



    #-----------------UNLABELED_TRAIN_AND_TEST_SET_CREATION-------------------#
    print("*"*50)

    # In each repetition we are randomly selecting num_samples of negative or unlabelled samples
    

    # Unlabeled Indices
    #unlabeled_indices = torch.arange(len(unlabeled_smiles_compound))
    
    

    # RANDOMLY SELECT - NUM_SAMPLES unlabeled as training samples.
    np.random.seed(iteration + 1 + i + ATTEMPT)
    train_unlabeled_indices = np.random.choice(unlabeled_and_spies_indices, 
                                               size=NUM_SAMPLES, 
                                               replace=False)
    
    # LIST TO KEEP TRACK OF ORIGINAL SAMPLE NUMBERS...
    #this is an extra line
    # train_unlabeled_and_spies_indices =  [unlabeled_and_spies_indices[index] for index in train_unlabeled_indices]

    
    
    
    #spy_indices = list(torch.where(unlabeled_targets==1)[0])
    
    #print(train_unlabeled_indices)
    #print(spy_indices)
    
    
    
    
    untouched_unlabeled_indices = np.setdiff1d(unlabeled_and_spies_indices,
                                               train_unlabeled_indices)
    #print(untouched_unlabeled_indices)
    #sys.exit()
    
    # LIST TO KEEP TRACK OF ORIGINAL SAMPLE NUMBERS...
    # this is an extra line
    # untouched_unlabeled_and_spies_indices =  [unlabeled_and_spies_indices[index] for index in untouched_unlabeled_indices]
   

    train_unlabeled_smiles_compound = tt_smile_compound_data[train_unlabeled_indices]
    train_unlabeled_smiles_adjacency = tt_smile_adjacency_data[train_unlabeled_indices]
    train_unlabeled_proteins = tt_protein_data[train_unlabeled_indices]
    #train_unlabeled_targets_unchanged = unlabeled_targets[train_unlabeled_indices]
    train_unlabeled_targets = targets[train_unlabeled_indices]
    
    mask = train_unlabeled_targets==1
    train_unlabeled_targets[mask]==0
    
#     #print(train_unlabeled_targets)
#     # Resetting the target values for spies in unlabeled train samples..
#     mask_indices = list(torch.where(train_unlabeled_targets==1))
    
#     #print(mask_indices)
#     train_unlabeled_targets[mask_indices] = 0
#     #print(train_unlabeled_targets)
    
#     untouched_unalebed_final_without_spies[i] = list(torch.where(unlabeled_targets[untouched_unlabeled_indices]!=1)[0])
    
    
    
    
    #this is an extra line
    # untouched_unalebed_final_indices_without_spies[i] = [untouched_unlabeled_and_spies_indices[index] for index in 
    #                                                      untouched_unalebed_final_without_spies[i]]
    #print(untouched_unalebed_final_without_spies[i])
    
    

    # # REMANING WILL BE test samples
    # test_unlabeled_indices = np.setdiff1d(unlabeled_indices, train_unlabeled_indices)
    # test_unlabeled_smiles_compound = unlabeled_smiles_compound[test_unlabeled_indices]
    # test_unlabeled_smiles_adjacency = unlabeled_smiles_adjacency[test_unlabeled_indices]
    # test_unlabeled_proteins = unlabeled_proteins[test_unlabeled_indices]
    # test_unlabeled_targets = unlabeled_targets[test_unlabeled_indices]


    print("Number of unlabeled samples in training set: ", len(train_unlabeled_indices))
    print("Shape Information of training set: ")
    print("Smiles compound: ", train_unlabeled_smiles_compound.shape)
    print("Smiles adjacency: ", train_unlabeled_smiles_adjacency.shape)
    print("Proteins: ", train_unlabeled_proteins.shape)
    print("Targets: ", train_unlabeled_targets.shape)
    print("-"*50)

    # print("Number of unlabeled samples in test set: ", len(test_unlabeled_indices))
    # print("Shape Information of test set: ")
    # print("Smiles compound: ", test_unlabeled_smiles_compound.shape)
    # print("Smiles adjacency: ", test_unlabeled_smiles_adjacency.shape)
    # print("Proteins: ", test_unlabeled_proteins.shape)
    # print("Targets: ", test_unlabeled_targets.shape)
    # print("*"*50)
    # print("\n")
    #-----------------UNLABELED_TRAIN_AND_TEST_SET_CREATION-------------------#




    #---------------COMBINING TRAIN POSITIVE AND UNLABELED-------------------#
    train_smiles_compound = torch.cat((train_pos_smiles_compound, train_unlabeled_smiles_compound))
    train_smiles_adjacency = torch.cat((train_pos_smiles_adjacency, train_unlabeled_smiles_adjacency))
    train_proteins = torch.cat((train_pos_proteins, train_unlabeled_proteins))
    train_targets = torch.cat((train_pos_targets, train_unlabeled_targets))
    #---------------COMBINING TRAIN POSITIVE AND UNLABELED-------------------#

    




    #--------------RANDOMIZE TRAINING INDICES-------------------------------#
    # Randomly permuting the data

    random_indices = [z for z in range(len(train_smiles_compound))]
    np.random.seed(iteration + 1 + i + ATTEMPT)
    np.random.shuffle(random_indices)
    #--------------RANDOMIZE TRAINING INDICES-------------------------------#



    #----------------MODEL TRAINING--------------------#
    model.train()

    # TRAINING...
    for epoch in tqdm(range(NUM_EPOCHS)):

        loss_value = 0
        batch_count = 0

        optimizer.zero_grad()

        # BATCHES...
        b = 0
        accuracy_ = []
        while (b+BATCH_SIZE <=train_smiles_compound.shape[0]):
            batch_count += 1

            #--------------------FORWARD PASS---------------------------------#
            predictions = model(train_smiles_compound[random_indices[b:b+BATCH_SIZE]],train_smiles_adjacency[random_indices[b:b+BATCH_SIZE]],
                                train_proteins[random_indices[b:b+BATCH_SIZE]]).cpu()
            
            #predictions = predictions.cpu()

            # Compute Loss
            loss = criterion(predictions, train_targets[random_indices[b:b+BATCH_SIZE]])

            loss_value += loss.item()

            # Compute Acc...
            accuracy = torch.sum((predictions>0.5).float() ==
                                 train_targets[random_indices[b:b+BATCH_SIZE]])

            accuracy_.append(accuracy.item()/BATCH_SIZE)
            #--------------------FORWARD PASS---------------------------------#



            #--------------------BACKWARD PASS---------------------------------#
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            b += BATCH_SIZE
            #--------------------BACKWARD PASS---------------------------------#



        if (b < train_smiles_compound.shape[0]):
            batch_count += 1

            #--------------------FORWARD PASS---------------------------------#
            predictions = model(train_smiles_compound[random_indices[b:train_smiles_compound.shape[0]]], train_smiles_adjacency[random_indices[b:train_smiles_compound.shape[0]]],
                                train_proteins[random_indices[b:train_smiles_compound.shape[0]]]).cpu()

            #predictions = predictions.cpu()
            # Compute Loss
            loss = criterion(predictions, train_targets[random_indices[b:train_smiles_compound.shape[0]]])
            loss_value += loss.item()
            #loss_values[i].append(loss.item())

            # Compute Acc...
            accuracy = torch.sum((predictions>0.5).float() ==
                                 train_targets[random_indices[b:train_smiles_compound.shape[0]]])

            accuracy_.append(accuracy.item()/BATCH_SIZE)
            #--------------------FORWARD PASS---------------------------------#



            #--------------------BACKWARD PASS---------------------------------#
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            #b += BATCH_SIZE
            #--------------------BACKWARD PASS---------------------------------#
        if((epoch+1) %100 == 0):
            print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item()}, Acc : {np.mean(accuracy_)}',
                  flush=True)
        loss_values[i].append(loss_value/batch_count)    

    #----------------MODEL TRAINING--------------------#



    #-----------------------------------------PLOTS-----------------------------------------#
    #print(len(loss_values[i]))
    plt.plot([index for index in range(len(loss_values[i]))], loss_values[i], marker='x')
    plt.xlabel("epoch numbers")
    plt.ylabel("loss values")
    plt.title("epoch number vs loss values plot")
    plt.savefig(DIR_NAME + "/loss_values_plot_" + str(i+1) + ".png")
    plt.clf()
    #-----------------------------------------PLOTS-----------------------------------------#



    #----------------SAVING THE MODEL--------------------#
    torch.save(model.state_dict(), DIR_NAME + '/my_model_transformercpi' + str(i+1) + '.pth')
    #----------------SAVING THE MODEL--------------------#





    #--------------------------------------PREDICTIONS---------------------------------------#
    torch.cuda.empty_cache()
    


    with torch.no_grad():
        model.eval()


        ###-------------------------TRAIN PREDICTIONS------------------------------------------------###


#         ##-------------------------TRAIN PREDICTIONS-------------------------##
#         print("Obtaining predictions on the complete training dataset...")
#         j = 0
#         train_predictions[i] = []

#         while(j+BATCH_SIZE  <= len(train_smiles_compound)):   
#             train_predictions[i] += list(model(train_smiles_compound[j:j+BATCH_SIZE], train_smiles_adjacency[j:j+BATCH_SIZE], train_proteins[j:j+BATCH_SIZE]).cpu())
#             j += BATCH_SIZE

#         if j < len(train_smiles_compound):
#             train_predictions[i] += list(model(train_smiles_compound[j:len(train_smiles_compound)], train_smiles_adjacency[j:len(train_smiles_compound)], train_proteins[j:len(train_smiles_compound)]).cpu())
#         ##-------------------------TRAIN PREDICTIONS-------------------------##



        ##-------------------------TRAIN PREDICTIONS-------------------------##
        print("Obtaining predictions on the complete training dataset...")
        j = 0
        train_pos_predictions[i] = []

        while(j+BATCH_SIZE  <= len(train_pos_smiles_compound)):   
            train_pos_predictions[i] += list(model(train_pos_smiles_compound[j:j+BATCH_SIZE],train_pos_smiles_adjacency[j:j+BATCH_SIZE], train_pos_proteins[j:j+BATCH_SIZE]).cpu())
            j += BATCH_SIZE

        if j < len(train_pos_smiles_compound):
            train_pos_predictions[i] += list(model(train_pos_smiles_compound[j:len(train_pos_smiles_compound)], train_pos_smiles_adjacency[j:len(train_pos_smiles_compound)],train_pos_proteins[j:len(train_pos_smiles_compound)]).cpu())
        #-------------------------TRAIN PREDICTIONS-------------------------##
        
        # train_predictions = train_predictions.cpu()
        # train_pos_predictions = train_pos_predictions.cpu()






        # Obtaining all training samples predictions..
        # train_predictions[i] = np.array(model(train_smiles_compound,train_smiles_adjacency, train_proteins).cpu())

        # Obtaining training positives predictions...
        # train_pos_predictions[i] = np.array(model(train_pos_smiles_compound, train_pos_smiles_adjacency, train_pos_proteins).cpu())

#         ###-------------------------TRAIN PREDICTIONS------------------------------------------------###

        
    
        
        # Are there any spies that are part of negatives in training set?
        #touched_spy_indices[i] = torch.nonzero(train_unlabeled_targets == 1)
        

#         # If yes, then remove them from computing Spies Capture Rate.
#         if len(touched_spy_indices[i]) > 0:
#             print("Are there any spies that are part of negatives in training set? yes, then remove them from computing Spies Capture Rate")




            # Pad zeros along 2nd dimension (h dimension)
#             padded_smiles_c = F.pad(spy_smiles_compound, (0, 0, 0, zeros_to_pad_h_c))
#             #padded_smiles_a = F.pad(spy_smiles_adjacency, (0, 0, 0, zeros_to_pad_h_a))

#             # Pad zeros along 3rd dimension (w dimension)
#             padded_smiles_c = F.pad(padded_smiles_c, (0, zeros_to_pad_w_c, 0, 0))
            #padded_smiles_a = F.pad(padded_smiles_a, (0, zeros_to_pad_w_a, 0, 0))


            ###____________THIS LOGIC WILL HELP IN IDENTIFYING SPIES THAT ARE IN TRAINING SET___________###

            # zeros_tensor = torch.zeros((touched_spy_indices[:, 0].shape[0], 1, zeros_to_append)).cuda()

#             padded_unlabeled_smiles_c = torch.nn.functional.pad(train_unlabeled_smiles_compound[touched_spy_indices[i][:, 0], :,:], (0, num_zeros_dim3, 0, num_zeros_dim2))
#             #padded_unlabeled_smiles_a = F.pad(train_unlabeled_smiles[touched_spy_indices[:, 0], :, :,:], (0, 0, 0, zeros_to_pad_h_a))

#             #padded_unlabeled_smiles_c = F.pad(padded_unlabeled_smiles_c, (0, zeros_to_pad_w_c, 0, 0))
#             #padded_unlabeled_smiles_a = F.pad(padded_unlabeled_smiles_a, (0, zeros_to_pad_w_c, 0, 0))

#             # result_tensor2 = torch.cat((train_unlabeled_smiles[touched_spy_indices[:, 0], :, :,:], zeros_tensor), dim=2).cuda()
#             spies_in_train = (padded_unlabeled_smiles_c+train_unlabeled_proteins[touched_spy_indices[i][:, 0], :, :])
#             print("Number of unlabeled samples in training set: ", spies_in_train.shape)



#             ###____________THIS LOGIC WILL HELP IN IDENTIFYING SPIES IN TRAINING SET___________###

#             #print("all_spies: ", all_spies.shape, all_spies.reshape([all_spies.shape[0], all_spies.shape[2]]).shape)
#             #print("all_spies2: ", all_spies2.shape, all_spies2.reshape([all_spies2.shape[0], all_spies2.shape[2]]).shape)

#             ###-------------------------LOGIC TO GET THE LIST OF UNTOUCHED SPIES-------------------------###
#             print("Are there any spies in training dataset?")
#             _, idx, counts = torch.cat([all_spies, spies_in_train], dim=0).unique(dim=0, return_inverse=True, return_counts=True)
#             mask = torch.isin(idx, torch.where(counts.gt(1))[0])
#             mask1 = mask[:len(all_spies)]
#             mask2 = mask[len(all_spies):]
#             indices1 = torch.arange(len(mask1))[mask1.cpu()]  # tensor([0, 2])
#             indices2 = torch.arange(len(mask2))[mask2.cpu()]  # tensor([0, 3])
#             touched_spies_indices = [int(index) for index in indices1]
#             untouched_spies_indices = [index for index in range(len(all_spies)) if index not in touched_spies_indices]
            
#             ###-------------------------LOGIC TO GET THE LIST OF UNTOUCHED SPIES-------------------------###

#             print("Obtaining predictions on the spies")
#             #### ADDING IN BATCH LOGIC ####
#             ###--------------------------------------TRAIN PREDICTIONS------------------------------------------###
        
        #untouched_spy_mask = targets[untouched_unlabeled_indices]==1
        #untouched_spies = sum(untouched_spy_mask)
        
        untouched_spy_indices = [index for index in untouched_unlabeled_indices if index <500]
        
        j = 0
        spy_predictions[i] = []
        spy_predictions_indices[i] = untouched_spy_indices
            
        while(j+BATCH_SIZE  <= len(untouched_spy_indices)):   
            spy_predictions[i] += list(model(tt_smile_compound_data[untouched_spy_indices[j:j+BATCH_SIZE]],
                                tt_smile_adjacency_data[untouched_spy_indices[j:j+BATCH_SIZE]],
                                tt_protein_data[untouched_spy_indices[j:j+BATCH_SIZE]]).cpu())
            j += BATCH_SIZE

        if j < len(untouched_spy_indices):
            spy_predictions[i] += list(model(tt_smile_compound_data[untouched_spy_indices[j:len(untouched_spy_indices)]],
                tt_smile_adjacency_data[untouched_spy_indices[j:len(untouched_spy_indices)]],                     tt_protein_data[untouched_spy_indices[j:len(untouched_spy_indices)]]).cpu())

        #print("touched spies")
        #print(spy_predictions_indices[i]) 

            
            ###--------------------------------------TRAIN PREDICTIONS------------------------------------------###
            #### ADDING IN BATCH LOGIC ####
            ###--------------------------------------SPY PREDICTIONS------------------------------------------------###

#         else:
#             ###--------------------------------------SPY PREDICTIONS------------------------------------------------###
#             #spy_predictions[i] = np.array(model(spy_smiles, spy_proteins).cpu())
#             print("Obtaining predictions on the spies")
#             #### ADDING IN BATCH LOGIC ####
#             ###--------------------------------------TRAIN PREDICTIONS------------------------------------------###
#             #spy_indices = (unlabeled_targets==1)
#             j = 0
#             spy_predictions[i] = []
#             spy_predictions_indices[i] = list([k for k in range(len(all_spies))])
#             print("untouched spies")
#             print(spy_predictions_indices[i])
            
# #             while(j+BATCH_SIZE  <= len(untouched_spies_indices)):   
# #                 spy_predictions[i] += list(model(spy_smiles_compound[untouched_spies_indices[j:j+BATCH_SIZE]], spy_smiles_adjacency[untouched_spies_indices[j:j+BATCH_SIZE]], spy_proteins[untouched_spies_indices[j:j+BATCH_SIZE]]).cpu())
# #                 spy_predictions_indices[i] += list(untouched_spies_indices[j:j+BATCH_SIZE])
                
# #                 j += BATCH_SIZE

# #             if j < len(untouched_spies_indices):
# #                 spy_predictions[i] += list(model(spy_smiles_compound[untouched_spies_indices[j:len(untouched_spies_indices)]],spy_smiles_adjacency[untouched_spies_indices[j:len(untouched_spies_indices)]], spy_proteins[untouched_spies_indices[j:len(untouched_spies_indices)]]).cpu())
# #                 spy_predictions_indices[i] += list(untouched_spies_indices[j:len(untouched_spies_indices)])

#             while(j+BATCH_SIZE  <= len(spy_predictions_indices[i])):   
#                 spy_predictions[i] += list(model(spy_smiles_compound[spy_predictions_indices[i][j:j+BATCH_SIZE]], spy_smiles_adjacency[spy_predictions_indices[i][j:j+BATCH_SIZE]], spy_proteins[spy_predictions_indices[i][j:j+BATCH_SIZE]]).cpu())
#                 # spy_predictions_indices[i] += list([index for index, value in enumerate(spy_smiles_compound) if index < j+4])
#                 j += BATCH_SIZE

#             if j < len(spy_predictions_indices[i]):
#                 spy_predictions[i] += list(model(spy_smiles_compound[spy_predictions_indices[i][j:len(spy_predictions_indices[i])]],
#                                                  spy_smiles_adjacency[spy_predictions_indices[i][j:len(spy_predictions_indices[i])]], spy_proteins[spy_predictions_indices[i][j:len(spy_predictions_indices[i])]]).cpu())
                #spy_predictions_indices[i] += list(untouched_spies_indices[j:len(spy_smiles_compound)])
            ###--------------------------------------TRAIN PREDICTIONS------------------------------------------###
            #### ADDING IN BATCH LOGIC ####
            ###--------------------------------------SPY PREDICTIONS------------------------------------------------###

        print("Length of spy_predictions: ", len(spy_predictions[i]))
        #print("Length of touched spies: ", len(touched_spy_indices[i]))
        if len(spy_predictions_indices[i]) != len(spy_predictions[i]):
            print("Error: Lengths of indices and probability_scores lists do not match for spy.")
        
        # Define the file name
        csv_file = DIR_NAME + '/spy_predictions_bin_'  + str(i) + '.csv'

        # Write data to CSV file
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
    
            # Write header
            writer.writerow(['Index', 'Probability Score'])
    
            # Write data rows
            for idx, score in zip(spy_predictions_indices[i], spy_predictions[i]):
                writer.writerow([idx, score])

        print(f"CSV file '{csv_file}' has been created successfully.")

        # Create a histogram using Seaborn
        spy_predictions[i] = [float(val) for val in spy_predictions[i]]
        #print(spy_predictions[i])
        sns.histplot(spy_predictions[i], bins=5, kde=True, color='skyblue')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Histogram of spy predictions')
        plt.savefig(DIR_NAME + '/spy_predictions_histogram_' + 'repeat' + str(i) + '.png')
        plt.clf()


#             ###---------------------LOGIC TO GET THE LIST OF UNTOUCHED SPIES-------------------------###
#             touched_spies = []
#             untouched_spies = []
#             touched_spies_indices = []
#             untouched_spies_indices = []

#             # Loop through each of the spies and check if they are present in training set.
#             for index, spy in enumerate(all_spies):
#                 if sum(spy[0] == spies_in_train[0][0])==len(spy[0]):
#                     print("touched")
#                     touched_spies.append(spy[0])
#                     touched_spies_indices.append(index)

#                 else:
#                     print("untouched")
#                     untouched_spies.append(spy[0])
#                     untouched_spies_indices.append(index)

#             # Convert the list to a torch tensor
#             untouched_spies = torch.stack(untouched_spies)
#             ###---------------------LOGIC TO GET THE LIST OF UNTOUCHED SPIES-------------------------###        




#         ###--------------------------------------SPY PREDICTIONS------------------------------------------------###
#         spy_predictions[i] = np.array(model(spy_smiles_compound[untouched_spies_indices, :, :],spy_smiles_adjacency[untouched_spies_indices, :, :], spy_proteins[untouched_spies_indices, :, :]).cpu())
#         ###--------------------------------------SPY PREDICTIONS------------------------------------------------###




        ###--------------------------------------UNLABELED PREDICTIONS------------------------------------------###
        untouched_test_indices = [index for index in untouched_unlabeled_indices if index >=500]
        j = 0
        unlabeled_predictions[i] = []
        
        while(j+BATCH_SIZE  <= len(untouched_test_indices)):   
            unlabeled_predictions[i] += list(model(tt_smile_compound_data[untouched_test_indices[j:j+BATCH_SIZE]],
                               tt_smile_adjacency_data[untouched_test_indices[j:j+BATCH_SIZE]], 
                               tt_protein_data[untouched_test_indices[j:j+BATCH_SIZE]]).cpu())
            j += BATCH_SIZE

        if j < len(untouched_test_indices):
            unlabeled_predictions[i] += list(model(tt_smile_compound_data[untouched_test_indices[j:]],
                               tt_smile_adjacency_data[untouched_test_indices[j:]],
                               tt_protein_data[untouched_test_indices[j:]]).cpu())

        
        # unlabeled_predictions[i] = temp
        # print(len(untouched_unalebed_final_without_spies[i]))


#         while(j+BATCH_SIZE  <= len(unlabeled_smiles_compound)):   
#             temp += list(model(unlabeled_smiles_compound[j:j+BATCH_SIZE], 
#                                                    unlabeled_smiles_adjacency[j:j+BATCH_SIZE], 
#                                                    unlabeled_proteins[j:j+BATCH_SIZE]).cpu())
#             j += BATCH_SIZE

#         if j < len(unlabeled_smiles_compound):
#             temp += list(model(unlabeled_smiles_compound[j:j+BATCH_SIZE],
#                                                    unlabeled_smiles_adjacency[j:j+BATCH_SIZE], 
#                                                    unlabeled_proteins[j:j+BATCH_SIZE]).cpu())
            
            
        #print(len(temp))
        
        
#         indices = []
#         for k in range(len(temp)):
#             if k not in train_unlabeled_indices and k not in spy_indices:
#                 indices.append(k)
#                 unlabeled_predictions[i].append(temp[k])
        
        
#         print(len(indices))
#         print(len(unlabeled_predictions[i]))
        if len(untouched_test_indices) != len(unlabeled_predictions[i]):
            print("Error: Lengths of indices and probability_scores lists do not match for unlabeled.")
        
        # Define the file name
        # csv_file = DIR_NAME + '/unlabeled_predictions_bin_'  + str(i) + '.csv'
        # #smiles_path = DIR_NAME + '/unlabeled_predictions_bin_smiles_'  + str(i) + '.npy'
        # #proteins_path = DIR_NAME + '/unlabeled_predictions_bin_proteins_'  + str(i) + '.npy'
        # index_path = DIR_NAME + '/unlabeled_predictions_bin_index'  + str(i) + '.npy'
        # scores_path = DIR_NAME + '/unlabeled_predictions_bin_scores_'  + str(i) + '.npy'
        
        # Convert the list of tensors to a list of NumPy arrays
        
        #list_of_smiles = np.array([unlabeled_smiles_compound[f].cpu().detach().numpy() for f in indices])
        #list_of_proteins = np.array([unlabeled_proteins[f].cpu().detach().numpy() for f in indices])
        
        # list_of_index = np.array(indices) #np.array(untouched_unalebed_final_indices_without_spies[i])#np.array(indices)
        # list_of_scores = np.array([tensor.cpu().detach().numpy() for tensor in unlabeled_predictions[i]])
        
        i_df = pd.DataFrame(columns=["original_index", "prediction_score"])
        i_df["original_index"] = untouched_test_indices
        i_df["prediction_score"] = unlabeled_predictions[i]
            
        i_df.to_csv(DIR_NAME +  "/predictions_bin#_" + str(i) + ".csv", index=False)
        
        # Save the list of NumPy arrays
        
        #np.save(smiles_path, list_of_smiles)
        #np.save(proteins_path, list_of_proteins)
        #np.save(index_path, list_of_index)
        #np.save(scores_path, list_of_scores)
        
        

        # Create a DataFrame with the data
        # data = {'Index': untouched_unalebed_final_without_spies[i],'Probability Score': unlabeled_predictions[i],'smile': [smile.cpu().detach().numpy() for smile in unlabeled_smiles_compound[untouched_unalebed_final_without_spies[i]]],'protein': [protein.cpu().detach().numpy() for protein in unlabeled_proteins[untouched_unalebed_final_without_spies[i]]]}
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.expand_frame_repr', False)
        # Assuming df is your DataFrame
        # Convert all columns to numpy arrays
        

        



        #df_ = pd.DataFrame(data)
        # for col in df_.columns:
        #     df_[col] = df_[col].apply(np.array)
            
        # Convert all columns to strings (to avoid ellipsis)
        #df_ = df_.astype(str)
            
        

        # Specify the path to your CSV file
        #csv_file_path = "unlabeled_dataframe_{i}.csv"

        # Write the DataFrame to a CSV file
        #df_.to_csv(csv_file, index=False)

# #         # Write data to CSV file
# #         with open(csv_file, mode='w', newline='') as file:
# #             writer = csv.writer(file)
    
# #             # Write header
# #             writer.writerow(['Index', 'Probability Score', 'smile', 'protein'])
    
# #             # Write data rows
# #             for idx, score, smile, protein in zip(untouched_unalebed_final_without_spies[i]
# #                                                   , unlabeled_predictions[i],
# #                                                   unlabeled_smiles_compound[untouched_unalebed_final_without_spies[i]], 
# #                                                   unlabeled_proteins[untouched_unalebed_final_without_spies[i]]):
# #                 writer.writerow([idx, score, smile.cpu().detach().numpy(), protein.cpu().detach().numpy()])

#         print(f"CSV file '{csv_file}' has been created successfully.")
#         # df = pd.DataFrame(['Index', 'Probability Score', 'smile', 'proteins'])
#         # df["Index"] = untouched_unalebed_final_without_spies[i]
#         # df["Probability Score"] = unlabeled_predictions[i]
#         # df["smile"] = unlabeled_smiles_compound[untouched_unalebed_final_without_spies[i]]
#         # df["proteins"] = unlabeled_proteins[untouched_unalebed_final_without_spies[i]]
#         # # Define the file name
#         # csv_file = DIR_NAME + '/unlabeled_predictions_bin_'  + str(i) + '.csv'
        
        # Create a histogram using Seaborn
        unlabeled_predictions[i] = [float(val) for val in unlabeled_predictions[i]]
        #print(spy_predictions[i])
        sns.histplot(unlabeled_predictions[i], bins=10, kde=True, color='skyblue')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Histogram of unlabeled predictions')
        plt.savefig(DIR_NAME + '/unlabeled_predictions_histogram_' + 'repeat' + str(i) + '.png')
        plt.clf()

# #         # Write data to CSV file
# #         with open(csv_file, mode='w', newline='') as file:
# #             writer = csv.writer(file)
    
# #             # Write header
# #             writer.writerow(['Index', 'Probability Score', 'smile', 'proteins'])
    
# #             # Write data rows
# #             for idx, score in zip(untouched_unalebed_final_without_spies, spy_predictions[i]):
# #                 writer.writerow([idx, score])

#         df.to_csv(csv_file)    
#     print(f"CSV file '{csv_file}' has been created successfully.")    
        ###--------------------------------------UNLABELED PREDICTIONS------------------------------------------###



    #---------------------PRINT STATEMENTS---------------------------------#
    #print("Length of all train predictions: ", len(train_predictions[i]))
    print("Length of all train pos predictions: ", len(train_pos_predictions[i]))
    print("Length of all spy predictions: ", len(spy_predictions[i]))
    #print("Length of all unlabeled predictions: ", len(unlabeled_predictions[i]))
    #---------------------PRINT STATEMENTS---------------------------------#



    #---------------------METRICS-----------------------------#

    print("*"*50)
    print("-"*50)
    print("Training Positives Metrics: ")
    tpr = sum([1 for pred in train_pos_predictions[i] if pred[0] >= 0.5])
    fpr = sum([1 for pred in train_pos_predictions[i] if pred[0]<0.5])
    print("TPR: ", tpr/NUM_SAMPLES)
    print("FPR: ", fpr/NUM_SAMPLES)
    print("-"*50)
    print("\n")

    print("-"*50)
#     print("Spies Capture Rate: ")
#     scr = sum([1 for pred in spy_predictions[i] if pred >= 0.5])


#     scr_values["exper_num"].append(i)
#     scr_values["num_samples"].append(len(spy_predictions[i]))
#     scr_values["scr_value"].append(scr/len(spy_predictions[i]))


#     print("SCR: ", scr/len(spy_predictions[0]))
    print("-"*50)
    print("\n")
    print("*"*50)

#     #---------------------METRICS-----------------------------#

#     plt.plot(list(loss_values.values()))
#     plt.xlabel('Values')
#     plt.ylabel('Frequency')
#     plt.title('Histogram of spy predictions')
#     plt.savefig(DIR_NAME + '/spy_predictions_histogram_' + 'repeat' + str(i) + '.png')
#     plt.clf()
    
    #return scr_values, unlabeled_predictions

#scr_values, unlabeled_predictions = run_experiment()
    
    
    
    
    
    
#     #-----------------------------SAVE UNLABELLED DATA AND PREDICTIONS---------------------#
#     unlabeled_classes = [1 if pred >= 0.5 else 0 for pred in unlabeled_predictions]
#     with open(DIR_NAME + '/unlabeled_classes.pickle', 'wb') as handle:
#         pickle.dump(unlabeled_classes, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     #-----------------------------SAVE UNLABELLED DATA AND PREDICTIONS---------------------#
    
# torch.cuda.empty_cache()  

# scr_df = pd.DataFrame(scr_values)
# scr_df.to_csv(DIR_NAME + '/scr_results.csv', index=Fal
# unlabeled_preds = [0 for _ in range(len(unlabeled_predictions[0]))]
# for key, values in unlabeled_predictions.items():
#     for k, v in enumerate(values):
#         unlabeled_preds[k] += v

# for k, v in enumerate(unlabeled_preds):
#     unlabeled_preds[k] = v/NUM_REPEATS

# unlabeled_classes = []
# for i in range(len(unlabeled_preds)):
#     if unlabeled_preds[i] <= 0.2:
#         unlabeled_classes.append(0)
#     elif unlabeled_preds[i] >= 0.8:
#         unlabeled_classes.append(1)
#     else:
#         unlabeled_classes.append(-1)


# # Create a Seaborn countplot from the list
# sns.countplot(x=unlabeled_classes)
# plt.xlabel('class')
# plt.ylabel('count')
# plt.title('countplot of unlabeled classes')
# plt.savefig(DIR_NAME + '/unlabeled_class_predictions.png')
# plt.clf()

# unlabeled_preds = [float(pred[0]) for pred in unlabeled_preds]
# sns.histplot(unlabeled_preds, bins=5, kde=True, color='skyblue')
# plt.xlabel('Values')
# plt.ylabel('Frequency')
# plt.title('Histogram of unlabeled predictions')
# plt.savefig(DIR_NAME + '/unlabeled_predictions_histogram_' + 'repeat' + str(i) + '.png')
# plt.clf()

# unlabeled_indices = list(torch.where(targets==0)[0])

# #Load the Original data
# original_df = pd.read_csv("/home/yaganapu/CYP/cyp_update/benchmarks/combined_data_no_duplicates.csv")
# original_df["new_class"] = 1
# mask = original_df["class"]==0
# original_df.loc[mask, "new_class"] = unlabeled_classes
# original_df.to_csv(DIR_NAME + 'data_for_phase2.csv')