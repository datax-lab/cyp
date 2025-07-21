import csv
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import torch
from torch import nn

import torch.optim as optim
import torch.nn.functional as F

from cyp_model_cpi_prediction import *

import sys
sys.path.append('/home/yaganapu/CYP/cyp_update/benchmarks/')


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="5"
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')
    
    
    
    
torch.cuda.empty_cache()

# Create an argument parser
parser = argparse.ArgumentParser(description='Run CPI prediction with ATTEMPT parameter.')

# Add the argument for ATTEMPT
parser.add_argument('--attempt', type=int, required=True, help='Iteration attempt number')

# Parse the arguments
args = parser.parse_args()

# Access the ATTEMPT value
ATTEMPT = args.attempt
print(f"Running CPI prediction for iteration: {ATTEMPT}")
BATCH_SIZE = 1

 
# TRAIN_PERCENTAGE = 0.70

print("\n"*2)
print("##########################################################")
# print("PERCENTAGE OF POSITIVE SAMPLES USED FOR TRAINING: ", TRAIN_PERCENTAGE*100)
print("##########################################################")
print("\n"*2)

NUM_SAMPLES = 300
NUM_REPEATS = 20

NUM_EPOCHS = 2


BASE_PATH = "/home/yaganapu/CYP/cyp_update/benchmarks/new_training_setting/generation_pseudo_labels"
COMPOUND_DATA_PATH = "/home/yaganapu/CYP/cyp_update/benchmarks/bacteria_data/cpiprediction/radius2_ngram3/compounds.npy"
ADJACENCY_DATA_PATH = "/home/yaganapu/CYP/cyp_update/benchmarks/bacteria_data/cpiprediction/radius2_ngram3/adjacencies.npy"
PROTEIN_DATA_PATH = "/home/yaganapu/CYP/cyp_update/benchmarks/bacteria_data/cpiprediction/radius2_ngram3/proteins.npy"
TARGET_DATA_PATH = "/home/yaganapu/CYP/cyp_update/benchmarks/bacteria_data/cpiprediction/radius2_ngram3/interactions.npy"

data_dir_input = "/home/yaganapu/CYP/cyp_update/benchmarks/bacteria_data/cpiprediction/radius2_ngram3/"

#--------------------------------Attempt_* FOLDER_CREATION--------------------------------#
DIR_NAME = BASE_PATH + '/phase_1_iterations/cpiprediction/' + 'Attempt_' + str(ATTEMPT) + '/'
if(os.path.exists(DIR_NAME)):
    None
else:
    os.mkdir(DIR_NAME)
#--------------------------------Attempt_* FOLDER_CREATION--------------------------------#


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle = True)]


smile_compound_data = load_tensor(data_dir_input + 'compounds', torch.LongTensor)
smile_adjacency_data = load_tensor(data_dir_input + 'adjacencies', torch.FloatTensor)
protein_data = load_tensor(data_dir_input + 'proteins', torch.LongTensor)
targets = load_tensor(data_dir_input + 'interactions', torch.FloatTensor)

#--------------------------------SMILE_DATA_PREP--------------------------------#
print("*"*50)
print("Shape of Smile Data Before Reshaping: ", smile_adjacency_data[0].shape)


#--------------------------------TARGETS_DATA_PREP--------------------------------#
print("*"*50)

NUM_TRAIN_SAMPLES = 300
NUM_VAL_SAMPLES = 100
NUM_TEST_SAMPLES = 100

# Replace -1s in targets with 0s
#targets[targets==-1]=0
print("Length of targets data: ", len(targets))
print("*"*50)
print("\n")
#--------------------------------TARGETS_DATA_PREP--------------------------------#


targets_flat = torch.cat(targets).cpu()


#--------------------------------POSITIVE_SAMPLES_SPLIT--------------------------------#
##-------------------------------TRAIN_AND_SPIES_SPLIT--------------------------------##
print("*"*50)

# Extract Positive Indices

train_randomized_data_path = f'/home/yaganapu/CYP/cyp_update/benchmarks/phase1_data/iteration_{ATTEMPT}/train_indices.npy'
val_randomized_data_path = f'/home/yaganapu/CYP/cyp_update/benchmarks/phase1_data/iteration_{ATTEMPT}/val_indices.npy'
test_randomized_data_path = f'/home/yaganapu/CYP/cyp_update/benchmarks/phase1_data/iteration_{ATTEMPT}/test_indices.npy'


train_pos_indices = np.load(train_randomized_data_path, allow_pickle = True)
val_pos_indices = np.load(val_randomized_data_path, allow_pickle = True)
#train_pos_indices = np.concatenate((train_pos_indices, val_pos_indices))


print("Number of positive class samples as spies: ", len(train_pos_indices))
print("-"*50)


#positive_indices = torch.where(targets_flat == 1)[0]
#NUM_SAMPLES = int(len(positive_indices)*TRAIN_PERCENTAGE)
#print("Total number of positive class samples: ", len(positive_indices))
print("-"*50)
print("Splitting the positive samples into training and spies sets")
#print("Number of positive class samples in training set: ", NUM_SAMPLES)

# RANDOM SEED TAKES THE VALUE OF CURRENT ATTEMPT
#np.random.seed(ATTEMPT)

# RANDOMLY SELECT - NUM_SAMPLES positives as training samples.
#train_pos_indices = np.random.choice(positive_indices, size=NUM_SAMPLES, replace=False)


train_pos_smiles_compound = [smile_compound_data[i] for i in train_pos_indices]
train_pos_smiles_adjacency = [smile_adjacency_data[i] for i in train_pos_indices]
train_pos_proteins = [protein_data[i] for i in train_pos_indices]
train_pos_targets = [targets[i] for i in train_pos_indices]




# REMANING WILL BE spy samples
#spy_samples = np.setdiff1d(positive_indices, train_pos_indices)

spy_samples = np.load(test_randomized_data_path, allow_pickle = True)

print("Number of positive class samples as spies: ", len(spy_samples))
print("-"*50)


spy_smiles_compound = [smile_compound_data[i] for i in spy_samples]
spy_smiles_adjacency = [smile_adjacency_data[i] for i in spy_samples]
spy_proteins = [protein_data[i] for i in spy_samples]
spy_targets = [targets[i] for i in spy_samples]

#print(spy_smiles_compound)



# # ###_________________THIS LOGIC WILL HELP IN IDENTIFYING SPIES IN TRAINING SET___________###
padded_spy_smiles = [torch.nn.functional.pad(spy_smiles_compound[i], (0, spy_proteins[i].shape[0] - spy_smiles_compound[i].shape[0])) for i in range(len(spy_samples))]

all_spies = [torch.add(padded_spy_smiles[i], spy_proteins[i]) for i in range(len(spy_samples))]

print("-"*50)
print("padded_smiles_shape:", all_spies[0].shape)
print("-"*50)
# ###_________________THIS LOGIC WILL HELP IN IDENTIFYING SPIES IN TRAINING SET___________###


print("Number of positive samples in training set: ", len(train_pos_indices))
print("Shape Information of training set: ")
print("Smiles compound: ", train_pos_smiles_compound[0].shape)
print("Smiles adjacency: ", train_pos_smiles_adjacency[0].shape)
print("Proteins: ", train_pos_proteins[0].shape)
print("Targets: ", train_pos_targets[0].shape)
print("-"*50)

print("Number of spy samples: ", len(spy_samples))
print("Shape Information of spy set: ")
print("Smiles compound: ", spy_smiles_compound[0].shape)
print("Smiles adjacency: ", spy_smiles_adjacency[0].shape)
print("Proteins: ", spy_proteins[0].shape)
print("Targets: ", spy_targets[0].shape)
print("*"*50)
print("\n")

##-------------------------------TRAIN_AND_SPIES_SPLIT--------------------------------##
#--------------------------------POSITIVE_SAMPLES_SPLIT--------------------------------#

##-------------------------------TRAIN_AND_SPIES_SPLIT--------------------------------##
#--------------------------------POSITIVE_SAMPLES_SPLIT--------------------------------#



#--------------------COMBINE_SPIES_AND_UNLABELED_SAMPLES-----------------#
print("*"*50)

unlabeled_indices = torch.where(targets_flat == 0)[0]


print("Number of unlabeled samples: ", len(unlabeled_indices))
print("-"*50)
print("Extracting unlabeled data... ")


unlabeled_smiles_compound_e = [smile_compound_data[i] for i in unlabeled_indices]
unlabeled_smiles_adjacency_e = [smile_adjacency_data[i] for i in unlabeled_indices]
unlabeled_proteins_e = [protein_data[i] for i in unlabeled_indices]
unlabeled_targets_e = [targets[i] for i in unlabeled_indices]


print("Combining spy samples with unlabeled data... ")
unlabeled_smiles_compound = unlabeled_smiles_compound_e + [spy_smiles_compound[i] for i in range(len(spy_samples))]
unlabeled_smiles_adjacency = unlabeled_smiles_adjacency_e + [spy_smiles_adjacency[i] for i in range(len(spy_samples))]
unlabeled_proteins = unlabeled_proteins_e + [spy_proteins[i] for i in range(len(spy_samples))]
unlabeled_targets = unlabeled_targets_e + [spy_targets[i] for i in range(len(spy_samples))]


print("Shuffling unlabeled data... ")
shuffled_indices = torch.randperm(len(unlabeled_smiles_compound))
#print(len(shuffled_indices))
unlabeled_smiles_compound = [unlabeled_smiles_compound[i] for i in shuffled_indices]
unlabeled_smiles_adjacency = [unlabeled_smiles_adjacency[i] for i in shuffled_indices]
unlabeled_proteins = [unlabeled_proteins[i] for i in shuffled_indices]
unlabeled_targets = [unlabeled_targets[i] for i in shuffled_indices]

#print(unlabeled_smiles_compound)

#sys.exit()

print("-"*50)
print("Number of unlabeled samples: ", len(unlabeled_smiles_compound))
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


scr_values = {"exper_num": [], 
              "num_samples": [],
              "scr_value": []}


dim=10
layer_gnn=3
side=5
window=11
layer_cnn=3
layer_output=3
lr=1e-3
lr_decay=0.5
decay_interval=10
weight_decay=1e-4

torch.cuda.empty_cache()
#torch.cuda.reset_peak_memory_stats()

print("device:", device)

for i in tqdm(range(NUM_REPEATS)):
    torch.cuda.empty_cache()
    
    model = CYPModel()

    #model.load_state_dict(torch.load('/home/yaganapu/CYP/cyp_update/benchmarks/phase1/transformercpi/Attempt_0/my_model1.pth'))

    # CUDA...
    model.to(device)
    #model = nn.DataParallel(model)
    
    
    criterion = nn.BCELoss()
    
    # OPTIMIZER
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    #-----------INITIALIZE MODEL OBJECT-------------#



    # Create an empty list to store unlabeled_predictions
    unlabeled_predictions[i] = []

    # Create an empty list to store loss values...
    loss_values[i] = []



    #-----------------UNLABELED_TRAIN_AND_TEST_SET_CREATION-------------------#
    print("*"*50)

    # In each repetition we are randomly selecting num_samples of negative or unlabelled samples
    np.random.seed((ATTEMPT*10) + 1 + i)

    # Unlabeled Indices
    unlabeled_indices = torch.arange(len(unlabeled_smiles_compound))

    # RANDOMLY SELECT - NUM_SAMPLES unlabeled as training samples.
    train_unlabeled_indices = np.random.choice(unlabeled_indices, size=NUM_SAMPLES, replace=False)
    unlabeled_targets_cat = torch.cat(unlabeled_targets)
    spy_indices = list(torch.where(unlabeled_targets_cat==1)[0])
    
    untouched_unlabeled_indices = np.setdiff1d(unlabeled_indices, train_unlabeled_indices)
    untouched_unlabeled_indices_list = untouched_unlabeled_indices.tolist()

    #train_unlabeled_smiles_compound = unlabeled_smiles_compound[train_unlabeled_indices]
    
    #unlabeled_smiles_compound = [unlabeled_smiles_compound[i] for i in shuffled_indices]

    train_unlabeled_smiles_compound = [unlabeled_smiles_compound[i] for i in train_unlabeled_indices]
    train_unlabeled_smiles_adjacency = [unlabeled_smiles_adjacency[i] for i in train_unlabeled_indices]
    train_unlabeled_proteins = [unlabeled_proteins[i] for i in train_unlabeled_indices]
    
    train_unlabeled_targets_unchanged = [unlabeled_targets[i] for i in train_unlabeled_indices]
    train_unlabeled_targets = [unlabeled_targets[i] for i in train_unlabeled_indices]
    print("*"*50)
    concatenated_targets = torch.cat(train_unlabeled_targets)
    #print(concatenated_targets)
    mask_indices = list(torch.where(concatenated_targets==1)[0])
    print("*"*50)
    #print(mask_indices)
    print("*"*50)
    if len(mask_indices) > 0:
        for idx in mask_indices:
            train_unlabeled_targets[idx] = torch.tensor([0.], device='cuda:0')
    #print(train_unlabeled_targets)
    print("*"*50)
    #print(untouched_unlabeled_indices_list)
    #print(unlabeled_targets)
    concatenated_unlabeled_targets = torch.cat(unlabeled_targets)
    untouched_unalebed_final_without_spies[i] = list(torch.where(concatenated_unlabeled_targets[untouched_unlabeled_indices_list]!=1)[0])
    
    

#     # REMANING WILL BE test samples
#     test_unlabeled_indices = np.setdiff1d(unlabeled_indices, train_unlabeled_indices)

#     test_unlabeled_smiles_compound = [unlabeled_smiles_compound[i] for i in test_unlabeled_indices]
#     test_unlabeled_smiles_adjacency = [unlabeled_smiles_adjacency[i] for i in test_unlabeled_indices]
#     test_unlabeled_proteins = [unlabeled_proteins[i] for i in test_unlabeled_indices]
#     test_unlabeled_targets = [unlabeled_targets[i] for i in test_unlabeled_indices]


    print("Number of unlabeled samples in training set: ", len(train_unlabeled_indices))
    print("Shape Information of training set: ")
    print("Smiles compound: ", train_unlabeled_smiles_compound[0].shape)
    print("Smiles adjacency: ", train_unlabeled_smiles_adjacency[0].shape)
    print("Proteins: ", train_unlabeled_proteins[0].shape)
    print("Targets: ", train_unlabeled_targets[0].shape)
    print("-"*50)

    # print("Number of unlabeled samples in test set: ", len(test_unlabeled_indices))
    # print("Shape Information of test set: ")
    # print("Smiles compound: ", test_unlabeled_smiles_compound[0].shape)
    # print("Smiles adjacency: ", test_unlabeled_smiles_adjacency[0].shape)
    # print("Proteins: ", test_unlabeled_proteins[0].shape)
    # print("Targets: ", test_unlabeled_targets[0].shape)
    # print("*"*50)
    # print("\n")
    #-----------------UNLABELED_TRAIN_AND_TEST_SET_CREATION-------------------#




    #---------------COMBINING TRAIN POSITIVE AND UNLABELED-------------------#
    train_smiles_compound = train_pos_smiles_compound + [train_unlabeled_smiles_compound[i] for i in range(len(train_unlabeled_smiles_compound))]
    train_smiles_adjacency = train_pos_smiles_adjacency + [train_unlabeled_smiles_adjacency[i] for i in range(len(train_unlabeled_smiles_adjacency))]
    train_proteins = train_pos_proteins + [train_unlabeled_proteins[i] for i in range(len(train_unlabeled_proteins))]
    train_targets = train_pos_targets + [train_unlabeled_targets[i] for i in range(len(train_unlabeled_targets))]


    # unlabeled_smiles_compound = unlabeled_smiles_compound_e + [spy_smiles_compound[i] for i in range(len(spy_samples))]
    # unlabeled_smiles_adjacency = unlabeled_smiles_adjacency_e + [spy_smiles_adjacency[i] for i in range(len(spy_samples))]
    # unlabeled_proteins = unlabeled_proteins_e + [spy_proteins[i] for i in range(len(spy_samples))]
    # unlabeled_targets = unlabeled_targets_e + [spy_targets[i] for i in range(len(spy_samples))]

    #---------------COMBINING TRAIN POSITIVE AND UNLABELED-------------------#






    #--------------RANDOMIZE TRAINING INDICES-------------------------------#
    # Randomly permuting the data
    #np.random.seed(ATTEMPT + 1 + i)
    random_indices = np.random.permutation([i for i in range(len(train_smiles_compound))])
    #--------------RANDOMIZE TRAINING INDICES-------------------------------#



    #----------------MODEL TRAINING--------------------#
    
    model.train()
    
    for epoch in tqdm(range(NUM_EPOCHS)):

        loss_value = 0
        batch_count = 0

        optimizer.zero_grad()

        # BATCHES...
        b = 0
        accuracy_ = []
        while (b+BATCH_SIZE <= len(train_smiles_compound)):
            batch_count += 1
            

            #--------------------FORWARD PASS---------------------------------#
            predictions = model(train_smiles_compound[random_indices[b:b+BATCH_SIZE][0]],train_smiles_adjacency[random_indices[b:b+BATCH_SIZE][0]],
                                train_proteins[random_indices[b:b+BATCH_SIZE][0]])

            predictions = predictions

            # Compute Loss
            loss = criterion(predictions, train_targets[random_indices[b:b+BATCH_SIZE][0]])

            loss_value += loss.item()

            # Compute Acc...
            accuracy = torch.sum((predictions>0.5).float() ==
                                 train_targets[random_indices[b:b+BATCH_SIZE][0]])

            accuracy_.append(accuracy.item()/BATCH_SIZE)
            #--------------------FORWARD PASS---------------------------------#



            #--------------------BACKWARD PASS---------------------------------#
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            b += BATCH_SIZE
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
    torch.save(model.state_dict(), DIR_NAME + '/my_model_cpi' + str(i+1) + '.pth')
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
#             train_predictions[i] += list(model(train_smiles_compound[j:j+BATCH_SIZE][0], train_smiles_adjacency[j:j+BATCH_SIZE][0], train_proteins[j:j+BATCH_SIZE][0]).cpu())
#             j += BATCH_SIZE

#         # if j < len(train_smiles_compound):
#         #     train_predictions[i] += list(model(train_smiles_compound[j:len(train_smiles_compound)], train_smiles_adjacency[j:len(train_smiles_compound)], train_proteins[j:len(train_smiles_compound)]).cpu())
#         ##-------------------------TRAIN PREDICTIONS-------------------------##



        ##-------------------------TRAIN PREDICTIONS-------------------------##
        print("Obtaining predictions on the complete training dataset...")
        j = 0
        train_pos_predictions[i] = []
        # print(train_pos_smiles_compound[j:j+BATCH_SIZE])
        # print(train_pos_smiles_compound[j:j+BATCH_SIZE][0])

        while(j+BATCH_SIZE  <= len(train_pos_smiles_compound)):   
            
            train_pos_predictions[i] += list(model(train_pos_smiles_compound[j:j+BATCH_SIZE][0],train_pos_smiles_adjacency[j:j+BATCH_SIZE][0], train_pos_proteins[j:j+BATCH_SIZE][0]).cpu())
            j += BATCH_SIZE
            
            

        # if j < len(train_pos_smiles_compound):
        #     train_pos_predictions[i] += list(model(train_pos_smiles_compound[j:len(train_pos_smiles_compound)], train_pos_smiles_adjacency[j:len(train_pos_smiles_compound)],train_pos_proteins[j:len(train_pos_smiles_compound)]).cpu())
        ##-------------------------TRAIN PREDICTIONS-------------------------##

        #train_predictions = train_predictions.cpu()
        #train_pos_predictions = train_pos_predictions.cpu()






    #         # Obtaining all training samples predictions..
    #         train_predictions[i] = np.array(model(train_smiles_compound,train_smiles_adjacency, train_proteins).cpu())

    #         # Obtaining training positives predictions...
    #         train_pos_predictions[i] = np.array(model(train_pos_smiles_compound, train_pos_smiles_adjacency, train_pos_proteins).cpu())

    #         ###-------------------------TRAIN PREDICTIONS------------------------------------------------###


        # Are there any spies that are part of negatives in training set?
        touched_spy_indices[i] = torch.nonzero(torch.tensor(train_unlabeled_targets) == 1.0)










        # If yes, then remove them from computing Spies Capture Rate.
        if len(touched_spy_indices[i]) > 0:
            print("Are there any spies that are part of negatives in training set? yes, then remove them from computing Spies Capture Rate")
            print("*"*50)
            print(touched_spy_indices[i])
            num_items = touched_spy_indices[i].numel()
            # Extract values based on the number of items
            if num_items >= 2:
                touched_spy_indices_list = touched_spy_indices[i].squeeze().tolist()
            else:
                touched_spy_indices_list = [touched_spy_indices[i].squeeze().item()]
            
            print(touched_spy_indices_list)
            print("*"*50)


            # Pad zeros along 2nd dimension (h dimension)
    #             padded_smiles_c = F.pad(spy_smiles_compound, (0, 0, 0, zeros_to_pad_h_c))
    #             #padded_smiles_a = F.pad(spy_smiles_adjacency, (0, 0, 0, zeros_to_pad_h_a))

    #             # Pad zeros along 3rd dimension (w dimension)
    #             padded_smiles_c = F.pad(padded_smiles_c, (0, zeros_to_pad_w_c, 0, 0))
            #padded_smiles_a = F.pad(padded_smiles_a, (0, zeros_to_pad_w_a, 0, 0))


            ###____________THIS LOGIC WILL HELP IN IDENTIFYING SPIES THAT ARE IN TRAINING SET___________###

            # zeros_tensor = torch.zeros((touched_spy_indices[:, 0].shape[0], 1, zeros_to_append)).cuda()

            #padded_unlabeled_smiles_c = torch.nn.functional.pad(train_unlabeled_smiles_compound[touched_spy_indices[i][:, 0], :,:], (0, num_zeros_dim3, 0, num_zeros_dim2))

            padded_unlabeled_smiles_c = [torch.nn.functional.pad(train_unlabeled_smiles_compound[m], (0, train_unlabeled_proteins[m].shape[0] - train_unlabeled_smiles_compound[m].shape[0])) for m in touched_spy_indices_list]

            #padded_unlabeled_smiles_a = F.pad(train_unlabeled_smiles[touched_spy_indices[:, 0], :, :,:], (0, 0, 0, zeros_to_pad_h_a))

            #padded_unlabeled_smiles_c = F.pad(padded_unlabeled_smiles_c, (0, zeros_to_pad_w_c, 0, 0))
            #padded_unlabeled_smiles_a = F.pad(padded_unlabeled_smiles_a, (0, zeros_to_pad_w_c, 0, 0))

            # result_tensor2 = torch.cat((train_unlabeled_smiles[touched_spy_indices[:, 0], :, :,:], zeros_tensor), dim=2).cuda()
            train_unlabeled_proteins_spylist = [train_unlabeled_proteins[n] for n in touched_spy_indices_list]
            spies_in_train = [torch.add(padded_unlabeled_smiles_c[o], train_unlabeled_proteins_spylist[o]) for o in range(len(touched_spy_indices_list))]
            print("Number of unlabeled samples in training set: ", spies_in_train[0].shape)



            ###____________THIS LOGIC WILL HELP IN IDENTIFYING SPIES IN TRAINING SET___________###

            #print("all_spies: ", all_spies.shape, all_spies.reshape([all_spies.shape[0], all_spies.shape[2]]).shape)
            #print("all_spies2: ", all_spies2.shape, all_spies2.reshape([all_spies2.shape[0], all_spies2.shape[2]]).shape)

            ###-------------------------LOGIC TO GET THE LIST OF UNTOUCHED SPIES-------------------------###
            print("Are there any spies in training dataset?")
            touched_spies_indices = []
            untouched_spies_indices = []

            for p, spy in enumerate(all_spies):
                if any(torch.equal(spy, s) for s in spies_in_train):
                    touched_spies_indices.append(p)
                else:
                    untouched_spies_indices.append(p)
            ###-------------------------LOGIC TO GET THE LIST OF UNTOUCHED SPIES-------------------------###

            print("Obtaining predictions on the spies")
            #### ADDING IN BATCH LOGIC ####
            ###--------------------------------------TRAIN PREDICTIONS------------------------------------------###
            j = 0
            spy_predictions[i] = []
            spy_predictions_indices[i] = []
            # print(untouched_spies_indices[j:j+BATCH_SIZE])
            # print(untouched_spies_indices[j:j+BATCH_SIZE][0])
            # print(spy_smiles_compound[untouched_spies_indices[j:j+BATCH_SIZE][0]])
            # break
            

            while(j+BATCH_SIZE  <= len(untouched_spies_indices)):   
                spy_predictions[i] += list(model(spy_smiles_compound[untouched_spies_indices[j:j+BATCH_SIZE][0]], spy_smiles_adjacency[untouched_spies_indices[j:j+BATCH_SIZE][0]], spy_proteins[untouched_spies_indices[j:j+BATCH_SIZE][0]]).cpu())
                spy_predictions_indices[i] += list(untouched_spies_indices[j:j+BATCH_SIZE][0])
                j += BATCH_SIZE

            # if j < len(untouched_spies_indices):
            #     spy_predictions[i] += list(model(spy_smiles_compound[untouched_spies_indices[j:len(untouched_spies_indices)]],spy_smiles_adjacency[untouched_spies_indices[j:len(untouched_spies_indices)]], spy_proteins[untouched_spies_indices[j:len(untouched_spies_indices)]]).cpu())

            ###--------------------------------------TRAIN PREDICTIONS------------------------------------------###
            #### ADDING IN BATCH LOGIC ####
            ###--------------------------------------SPY PREDICTIONS------------------------------------------------###

        else:
            ###--------------------------------------SPY PREDICTIONS------------------------------------------------###
            #spy_predictions[i] = np.array(model(spy_smiles, spy_proteins).cpu())
            print("Obtaining predictions on the spies")
            #### ADDING IN BATCH LOGIC ####
            ###--------------------------------------TRAIN PREDICTIONS------------------------------------------###
            j = 0
            spy_predictions[i] = []
            spy_predictions_indices[i] = list([k for k in range(len(all_spies))])
            print("untouched spies")
            #print(spy_predictions_indices[i])

            while(j+BATCH_SIZE  <= len(spy_predictions_indices[i])): 
                #print(spy_predictions_indices[i][j:j+BATCH_SIZE][0])
                spy_predictions[i] += list(model(spy_smiles_compound[spy_predictions_indices[i][j:j+BATCH_SIZE][0]], spy_smiles_adjacency[spy_predictions_indices[i][j:j+BATCH_SIZE][0]], spy_proteins[spy_predictions_indices[i][j:j+BATCH_SIZE][0]]).cpu())
                j += BATCH_SIZE

            # if j < len(spy_smiles_compound):
            #     spy_predictions[i] += list(model(spy_smiles_compound[j:len(spy_smiles_compound)],spy_smiles_adjacency[j:len(spy_smiles_compound)], spy_proteins[j:len(spy_proteins)]).cpu())
            ###--------------------------------------TRAIN PREDICTIONS------------------------------------------###
            #### ADDING IN BATCH LOGIC ####
            ###--------------------------------------SPY PREDICTIONS------------------------------------------------###

        print("Length of spy_predictions: ", len(spy_predictions[i]))
        print("Length of touched spies: ", len(touched_spy_indices[i]))
        
        if len(spy_predictions_indices[i]) != len(spy_predictions[i]):
            print("Error: Lengths of indices and probability_scores lists do not match.")
        
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
        

        j = 0
        unlabeled_predictions[i] = []
        #print(untouched_unalebed_final_without_spies[i])
        print(len(untouched_unalebed_final_without_spies[i]))

        temp = []
        #print(unlabeled_smiles_compound[j:j+BATCH_SIZE][0])
        while(j+BATCH_SIZE  <= len(unlabeled_smiles_compound)):
            temp += list(model(unlabeled_smiles_compound[j:j+BATCH_SIZE][0], 
                                                   unlabeled_smiles_adjacency[j:j+BATCH_SIZE][0], 
                                                   unlabeled_proteins[j:j+BATCH_SIZE][0]).cpu())
            j += BATCH_SIZE
            
        print(len(temp))
        
        indices = []
        for k in range(len(temp)):
            if k not in train_unlabeled_indices and k not in spy_indices:
                indices.append(k)
                unlabeled_predictions[i].append(temp[k])
        
        print(len(indices))
        print(len(unlabeled_predictions[i]))
        if len(indices) != len(unlabeled_predictions[i]):
            print("Error: Lengths of indices and probability_scores lists do not match for unlabeled.")
            
        # Define the file name
        csv_file = DIR_NAME + '/unlabeled_predictions_bin_'  + str(i) + '.csv'
        smiles_path = DIR_NAME + '/unlabeled_predictions_bin_smiles_'  + str(i) + '.npy'
        proteins_path = DIR_NAME + '/unlabeled_predictions_bin_proteins_'  + str(i) + '.npy'
        index_path = DIR_NAME + '/unlabeled_predictions_bin_index'  + str(i) + '.npy'
        scores_path = DIR_NAME + '/unlabeled_predictions_bin_scores_'  + str(i) + '.npy'    
        
        # Convert the list of tensors to a list of NumPy arrays
        
        #list_of_smiles = np.array([unlabeled_smiles_compound[f].cpu().detach().numpy() for f in indices])
        #list_of_proteins = np.array([unlabeled_proteins[f].cpu().detach().numpy() for f in indices])
        
        list_of_index = np.array(indices)
        list_of_scores = np.array([tensor.cpu().detach().numpy() for tensor in unlabeled_predictions[i]])
        
        #np.save(smiles_path, list_of_smiles)
        #np.save(proteins_path, list_of_proteins)
        np.save(index_path, list_of_index)
        np.save(scores_path, list_of_scores)
        
        data = {'Index': untouched_unalebed_final_without_spies[i],'Probability Score': unlabeled_predictions[i],'smile': [smile.cpu().detach().numpy() for smile in unlabeled_smiles_compound[untouched_unalebed_final_without_spies[i]]],'protein': [protein.cpu().detach().numpy() for protein in unlabeled_proteins[untouched_unalebed_final_without_spies[i]]]}
        
        df_ = pd.DataFrame(data)
        df_ = df_.astype(str)
        df_.to_csv(csv_file, index=False)

#         # Write data to CSV file
#         with open(csv_file, mode='w', newline='') as file:
#             writer = csv.writer(file)
    
#             # Write header
#             writer.writerow(['Index', 'Probability Score', 'smile', 'protein'])
#             smile_thing = [unlabeled_smiles_compound[f].cpu().detach().numpy() for f in indices]
#             protein_thing = [unlabeled_proteins[f].cpu().detach().numpy() for f in indices]
#             print("-"*50)
#             print(len(smile_thing))
#             print(len(protein_thing))
#             print("-"*50)
    
#             # Write data rows
#             for idx, score, smile, protein in zip(indices
#                                                   , unlabeled_predictions[i],
#                                                   smile_thing, 
#                                                   protein_thing):
#                 writer.writerow([idx, score, smile, protein])

        print(f"CSV files '{csv_file}' has been created successfully.")
        # Create a histogram using Seaborn
        unlabeled_predictions[i] = [float(val) for val in unlabeled_predictions[i]]
        #print(spy_predictions[i])
        sns.histplot(unlabeled_predictions[i], bins=10, kde=True, color='skyblue')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Histogram of unlabeled predictions')
        plt.savefig(DIR_NAME + '/unlabeled_predictions_histogram_' + 'repeat' + str(i) + '.png')
        plt.clf()
        

        # if j < len(unlabeled_smiles_compound_e):
        #     unlabeled_predictions[i] += list(model(unlabeled_smiles_compound_e[j:len(unlabeled_smiles_compound_e)],unlabeled_smiles_adjacency_e[j:len(unlabeled_smiles_compound_e)], unlabeled_proteins_e[j:len(unlabeled_smiles_compound_e)]).cpu())
        ###--------------------------------------UNLABELED PREDICTIONS------------------------------------------###



    #---------------------PRINT STATEMENTS---------------------------------#
    #print("Length of all train predictions: ", len(train_predictions[i]))
    print("Length of all train pos predictions: ", len(train_pos_predictions[i]))
    print("Length of all spy predictions: ", len(spy_predictions[i]))
    print("Length of all unlabeled predictions: ", len(unlabeled_predictions[i]))
    #---------------------PRINT STATEMENTS---------------------------------#



    #---------------------METRICS-----------------------------#

    print("*"*50)
    print("-"*50)
    print("Training Positives Metrics: ")
    tpr = sum([1 for pred in train_pos_predictions[i] if pred >= 0.5])
    fpr = sum([1 for pred in train_pos_predictions[i] if pred < 0.5])
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


#     print("SCR: ", scr/len(spy_predictions[i]))
    print("-"*50)
    print("\n")
    print("*"*50)

    #---------------------METRICS-----------------------------#

    # plt.plot(list(loss_values.values()))
    # plt.xlabel('Values')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of spy predictions')
    # plt.savefig(DIR_NAME + '/spy_predictions_histogram_' + 'repeat' + str(i) + '.png')
    # plt.clf()
    
    
torch.cuda.empty_cache()  

# scr_df = pd.DataFrame(scr_values)
# scr_df.to_csv(DIR_NAME + '/scr_results.csv', index=False)

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

# unlabeled_preds = [float(pred) for pred in unlabeled_preds]
# sns.histplot(unlabeled_preds, bins=5, kde=True, color='skyblue')
# plt.xlabel('Values')
# plt.ylabel('Frequency')
# plt.title('Histogram of unlabeled predictions')
# plt.savefig(DIR_NAME + '/unlabeled_predictions_histogram_' + 'repeat' + str(i) + '.png')
# plt.clf()

# unlabeled_indices = torch.where(targets_flat == 0)[0]

# #Load the Original data
# original_df = pd.read_csv("/home/yaganapu/CYP/cyp_update/benchmarks/combined_data_no_duplicates.csv")
# original_df["new_class"] = 1
# mask = original_df["class"]==0
# original_df.loc[mask, "new_class"] = unlabeled_classes
# original_df.to_csv(DIR_NAME + 'data_for_phase2.csv')            
    
    
