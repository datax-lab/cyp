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

from cyp_model_ssnet import *

import sys
sys.path.append('/home/yaganapu/CYP/cyp_update/benchmarks/')


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

    

torch.cuda.empty_cache()

# Defining Constants
ATTEMPT = 4
iteration = 5
BATCH_SIZE = 1

 
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
COMPOUND_DATA_PATH = "/home/yaganapu/CYP/cyp_update/benchmarks/bacteria_data/ssnet/bacteria/input/radius2_ngram3/compounds.npy"
ADJACENCY_DATA_PATH = "/home/yaganapu/CYP/cyp_update/benchmarks/bacteria_data/ssnet/bacteria/input/radius2_ngram3/adjacencies.npy"
PROTEIN_DATA_PATH = "/home/yaganapu/CYP/cyp_update/benchmarks/bacteria_data/ssnet/bacteria/input/radius2_ngram3/proteins.npy"
TARGET_DATA_PATH = "/home/yaganapu/CYP/cyp_update/benchmarks/bacteria_data/ssnet/bacteria/input/radius2_ngram3/interactions.npy"


data_dir_input = "/home/yaganapu/CYP/cyp_update/benchmarks/bacteria_data/ssnet/bacteria/input/radius2_ngram3/"


#--------------------------------Attempt_* FOLDER_CREATION--------------------------------#
DIR_NAME = BASE_PATH + '/phase1_new/iteration_' + str(iteration) +'/ssnet/' + 'Attempt_' + str(ATTEMPT) + '/'

train_randomized_data_path = '/home/yaganapu/CYP/cyp_update/benchmarks/phase1_data/iteration_' + str(iteration)+'/train_indices.npy'
val_randomized_data_path = '/home/yaganapu/CYP/cyp_update/benchmarks/phase1_data/iteration_' + str(iteration)+'/val_indices.npy'
test_randomized_data_path = '/home/yaganapu/CYP/cyp_update/benchmarks/phase1_data/iteration_' + str(iteration)+'/test_indices.npy'

if(os.path.exists(DIR_NAME)):
    None
else:
    os.mkdir(DIR_NAME)
#--------------------------------Attempt_* FOLDER_CREATION--------------------------------#


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle = True)]


smile_compound_data = load_tensor(data_dir_input + 'compounds', torch.LongTensor)
smile_adjacency_data = load_tensor(data_dir_input + 'adjacencies', torch.FloatTensor)

targets = load_tensor(data_dir_input + 'interactions', torch.FloatTensor)


print("*"*50)
# print(smile_compound_data[0].device.type)
# print(smile_adjacency_data[0].device.type)
# print(targets[0].device.type)

print("*"*50)
protein_data_array = np.load(data_dir_input + 'proteins.npy' , allow_pickle = True )


incorrect_indices = [119, 120, 121, 122, 396, 571, 591, 811, 832, 1048, 1069, 1289, 1310, 1529, 1550, 1768, 1789, 2007, 2028, 2247, 2268, 2487, 2508, 2727, 2748, 2966, 2987, 3205, 3226, 3445, 3466, 3686, 3707, 3925, 3946, 4165, 4186, 4405, 4426, 4644, 4665, 4884, 4905, 5123, 5144, 5362, 5383, 5603, 5624, 5843, 5864, 6082, 6103, 6323, 6344, 6561, 6581, 6798, 6819, 7038, 7059, 7277, 7298, 7517, 7538, 7756, 7777, 7997, 8018, 8236, 8257, 8475, 8496, 8714, 8735, 8954, 8975, 9191, 9212, 9431, 9452, 9671, 9692, 9911, 9932, 10150, 10171, 10390, 10411, 10631, 10652, 10870, 10891, 11110, 11131, 11351, 11372, 11590, 11611, 11830, 11851, 12070, 12091, 12309, 12330, 12569, 12788, 12809, 13027, 13048, 13267, 13288, 13507, 13528, 13747, 13768, 13987, 14008, 14225, 14246, 14465, 14486, 14702, 14721, 14935, 14955, 15173, 15194, 15413, 15433, 15652, 15673, 15893, 15914, 16132, 16153, 16373, 16394, 16613, 16634, 16853, 16874, 17092, 17113, 17330, 17351, 17567, 17588, 17806, 17827, 18045, 18066, 18285, 18306, 18525, 18546, 18766, 18787, 19005, 19026, 19245, 19265, 19485, 19506, 19723, 19744, 19960, 19981, 20200, 20221, 20440, 20461, 20679, 20700, 20919, 20940, 21159, 21180, 21399, 21420, 21635, 21655, 21870, 21891, 22110, 22131, 22350, 22371, 22590, 22611, 22830, 22851, 23069, 23090, 23307, 23328, 23545, 23566, 23781, 23802, 24014, 24035, 24255, 24276, 24494, 24515, 24729, 24750, 24970, 24991, 25192, 25211, 25413, 25434, 25653, 25674, 25892, 25913, 26132, 26153, 26371, 26392, 26611, 26632, 26851, 26872, 27090, 27111, 27330, 27351, 27566, 27587, 27804, 27824, 28043, 28064, 28283, 28304, 28523, 28544, 28764, 28785, 29004, 29025, 29244, 29265, 29480, 29501, 29718, 29739, 29958, 29979, 30197, 30218, 30437, 30458, 30676, 30697, 30916, 30937, 31155, 31176, 31396, 31417, 31633, 31653, 31873, 31894, 32113, 32134, 32353, 32374, 32589, 32610, 32829, 32850, 33068, 33089, 33308, 33329, 33548, 33569, 33789, 33810, 34028, 34049, 34268, 34288, 34507, 34528, 34746, 34767, 34986, 35007, 35226, 35247, 35466, 35487, 35705, 35726, 35946, 35967, 36206, 36426, 36447, 36665, 36686, 36906, 36927, 37146, 37165, 37384, 37405, 37624, 37644, 37863, 37884, 38104, 38125, 38344, 38365, 38581, 38602, 38821, 38842, 39061, 39082, 39301, 39321, 39541, 39562, 39778, 39799, 40017, 40037, 40256, 40277, 40494, 40515, 40733, 40754, 40972, 40993, 41213, 41234, 41453, 41474, 41691, 41712, 41928, 41949, 42189, 42409, 42430, 42648, 42669, 42889, 42909, 43129, 43149, 43366, 43387, 43607, 43628, 43847, 44086, 44107, 44327, 44348, 44567, 44588, 44806, 44827, 45044, 45065, 45284, 45305, 45523, 45544, 45764, 45785, 46004, 46025, 46244, 46265, 46483, 46504, 46723, 46744, 46963, 46984, 47203, 47224, 47443, 47464, 47683, 47704, 47922, 47943, 48163, 48184, 48403, 48424, 48640, 48661, 48880, 48901, 49120, 49141, 49360, 49381, 49599, 49620, 49839, 49860, 50080, 50101, 50320, 50341, 50560, 50581, 50799, 50820, 51038, 51059, 51277, 51298, 51517, 51538, 51757, 51778, 51997, 52018, 52235, 52256, 52476, 52497, 52716, 52736, 52956, 52977, 53195, 53216, 53436, 53457, 53676, 53697, 53916, 53937, 54156, 54177, 54396, 54417, 54636, 54657, 54876, 54897, 55114, 55135, 55352, 55373, 55591, 55612, 55831, 55852, 56071, 56092, 56311, 56331, 56550, 56571, 56791, 56812, 57030, 57051, 57266, 57287, 57505, 57526, 57746, 57767, 57986, 58007, 58226, 58247, 58464, 58485, 58704, 58725, 58945, 58965, 59184, 59205, 59424, 59445, 59664, 59685, 59904, 59925, 60144, 60165, 60384, 60404, 60624, 60645, 60863, 60884, 61103, 61124, 61343, 61364, 61582, 61603, 61842, 62061, 62082, 62302, 62323, 62541, 62562, 62781, 62802, 63021, 63042, 63261, 63282, 63500, 63521, 63738, 63759, 63978, 63999, 64216, 64237, 64455, 64476, 64693, 64714, 64933, 64954, 65173, 65194, 65412, 65433, 65652, 65673, 65892, 65913, 66130, 66149, 66365, 66386, 66604, 66625, 66844, 66865, 67085, 67105, 67322, 67343, 67562, 67583, 67803, 67823, 68043, 68064, 68282, 68303, 68523, 68544, 68763, 68784, 69003, 69024, 69243, 69264, 69482, 69503, 69721, 69742, 69962, 69983]

protein_data_array = np.array([arr.astype(np.float32) for arr in protein_data_array])

protein_data = [torch.tensor(arr).cuda() for arr in protein_data_array]


# Remove elements at the specified indices from protein_data

# protein_data = [protein_data[i] for i in range(len(protein_data)) if i not in incorrect_indices]
# # Remove elements at the specified indices from smile_compound_data
# smile_compound_data = [smile_compound_data[i] for i in range(len(smile_compound_data)) if i not in incorrect_indices]

# # Remove elements at the specified indices from smile_adjacency_data
# smile_adjacency_data = [smile_adjacency_data[i] for i in range(len(smile_adjacency_data)) if i not in incorrect_indices]

# # Remove elements at the specified indices from targets
# targets = [targets[i] for i in range(len(targets)) if i not in incorrect_indices]


print("*"*50)
# print(smile_compound_data[0].device.type)
# print(smile_adjacency_data[0].device.type)
# print(targets[0].device.type)
# print(protein_data[0].device.type)

print("*"*50)



#--------------------------------SMILE_DATA_PREP--------------------------------#
print("*"*50)
print("Shape of Smile Data Before Reshaping: ", smile_adjacency_data[0].shape)


#--------------------------------TARGETS_DATA_PREP--------------------------------#
print("*"*50)
NUM_TRAIN_SAMPLES = 400
#NUM_VAL_SAMPLES = 100
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




train_pos_indices_1 = np.load(train_randomized_data_path, allow_pickle = True)
val_pos_indices_1 = np.load(val_randomized_data_path, allow_pickle = True)
train_pos_indices = np.concatenate((train_pos_indices_1, val_pos_indices_1))





# Extract Positive Indices
# positive_indices = torch.where(targets_flat == 1)[0]
# NUM_SAMPLES = int(len(positive_indices)*TRAIN_PERCENTAGE)
# print("Total number of positive class samples: ", len(positive_indices))
print("-"*50)
print("Splitting the positive samples into training and spies sets")
#print("Number of positive class samples in training set: ", NUM_SAMPLES)

# RANDOM SEED TAKES THE VALUE OF CURRENT ATTEMPT


# RANDOMLY SELECT - NUM_SAMPLES positives as training samples.
#train_pos_indices = np.random.choice(positive_indices, size=NUM_SAMPLES, replace=False)


train_pos_indices = [i for i in train_pos_indices if i not in incorrect_indices]

train_pos_smiles_compound = [smile_compound_data[i] for i in train_pos_indices]
train_pos_smiles_adjacency = [smile_adjacency_data[i] for i in train_pos_indices]
train_pos_proteins = [protein_data[i] for i in train_pos_indices]
train_pos_targets = [targets[i] for i in train_pos_indices]

print("Number of positive class samples : ", len(train_pos_smiles_compound))
print("-"*50)

# protein_data = [protein_data[i] for i in range(len(protein_data)) if i not in incorrect_indices]
# smile_compound_data = [smile_compound_data[i] for i in range(len(smile_compound_data)) if i not in incorrect_indices]
# smile_adjacency_data = [smile_adjacency_data[i] for i in range(len(smile_adjacency_data)) if i not in incorrect_indices]
# targets = [targets[i] for i in range(len(targets)) if i not in incorrect_indices]


# REMANING WILL BE spy samples
#spy_samples = np.setdiff1d(positive_indices.cpu(), train_pos_indices)

spy_indices = np.load(val_randomized_data_path, allow_pickle = True)

spy_indices = [i for i in spy_indices if i not in incorrect_indices]
print("Number of positive class samples as spies: ", len(spy_indices))
print("-"*50)


# spy_smiles_compound = [smile_compound_data[i] for i in spy_samples if i not in incorrect_indices]
# spy_smiles_adjacency = [smile_adjacency_data[i] for i in spy_samples if i not in incorrect_indices]
# spy_proteins = [protein_data[i] for i in spy_samples if i not in incorrect_indices]
# spy_targets = [targets[i] for i in spy_samples if i not in incorrect_indices]

#print("Number of positive class samples as spies: ", len(spy_smiles_compound))
print("-"*50)


# spy_smiles_compound_adding_second_dim = [spy_smiles_compound[i].unsqueeze(1) for i in range(len(spy_smiles_compound))]

# print(len(spy_smiles_compound))
# #spy_smiles_compound_adding_second_dim[3].shape


# # # ###_________________THIS LOGIC WILL HELP IN IDENTIFYING SPIES IN TRAINING SET___________###
# padded_spy_smiles = [torch.nn.functional.pad(spy_smiles_compound_adding_second_dim[i], (0,  spy_proteins[i].shape[1] - spy_smiles_compound_adding_second_dim[i].shape[1], 0, spy_proteins[i].shape[0] - spy_smiles_compound_adding_second_dim[i].shape[0])) for i in range(len(spy_smiles_compound))]



# all_spies = [torch.add(padded_spy_smiles[i], spy_proteins[i]) for i in range(len(spy_smiles_compound))]

print("-"*50)
# print("padded_smiles_shape:", all_spies[0].shape)
print("-"*50)
# ###_________________THIS LOGIC WILL HELP IN IDENTIFYING SPIES IN TRAINING SET___________###


print("Number of positive samples in training set: ", len(train_pos_smiles_compound))
print("Shape Information of training set: ")
print("Smiles compound: ", train_pos_smiles_compound[0].shape)
print("Smiles adjacency: ", train_pos_smiles_adjacency[0].shape)
print("Proteins: ", train_pos_proteins[0].shape)
print("Targets: ", train_pos_targets[0].shape)
print("-"*50)

# print("Number of spy samples: ", len(spy_smiles_compound))
# print("Shape Information of spy set: ")
# print("Smiles compound: ", spy_smiles_compound[0].shape)
# print("Smiles adjacency: ", spy_smiles_adjacency[0].shape)
# print("Proteins: ", spy_proteins[0].shape)
# print("Targets: ", spy_targets[0].shape)
# print("*"*50)
# print("\n")


##-------------------------------TRAIN_AND_SPIES_SPLIT--------------------------------##
#--------------------------------POSITIVE_SAMPLES_SPLIT--------------------------------#

##-------------------------------TRAIN_AND_SPIES_SPLIT--------------------------------##
#--------------------------------POSITIVE_SAMPLES_SPLIT--------------------------------#



#--------------------COMBINE_SPIES_AND_UNLABELED_SAMPLES-----------------#
print("*"*50)


unlabeled_indices_e = torch.where(targets_flat == 0)[0]
unlabeled_and_spies_indices = [int(tensor) for tensor in unlabeled_indices_e] + list(spy_indices)
unlabeled_and_spies_indices = [i for i in unlabeled_and_spies_indices if i not in incorrect_indices]
np.random.seed(iteration+ATTEMPT)
np.random.shuffle(unlabeled_and_spies_indices)



print("Number of unlabeled samples: ", len(unlabeled_indices_e))
print("-"*50)
print("Extracting unlabeled data... ")


# unlabeled_smiles_compound_e = [smile_compound_data[i] for i in unlabeled_indices_e if i not in incorrect_indices]
# unlabeled_smiles_adjacency_e = [smile_adjacency_data[i] for i in unlabeled_indices_e if i not in incorrect_indices]
# unlabeled_proteins_e = [protein_data[i] for i in unlabeled_indices_e if i not in incorrect_indices]
# unlabeled_targets_e = [targets[i] for i in unlabeled_indices_e if i not in incorrect_indices]

# print("Number of unlabeled samples: ", len(unlabeled_smiles_compound_e))
# print("-"*50)
# print("Extracting unlabeled data... ")


# print("Combining spy samples with unlabeled data... ")
# unlabeled_smiles_compound = unlabeled_smiles_compound_e + [spy_smiles_compound[i] for i in range(len(spy_smiles_compound))]
# unlabeled_smiles_adjacency = unlabeled_smiles_adjacency_e + [spy_smiles_adjacency[i] for i in range(len(spy_smiles_compound))]
# unlabeled_proteins = unlabeled_proteins_e + [spy_proteins[i] for i in range(len(spy_smiles_compound))]
# unlabeled_targets = unlabeled_targets_e + [spy_targets[i] for i in range(len(spy_smiles_compound))]

# print("Number of unlabeled samples: ", len(unlabeled_smiles_compound))
# print("-"*50)
# print("Extracting unlabeled data... ")


# print("Shuffling unlabeled data... ")
# shuffled_indices = torch.randperm(len(unlabeled_smiles_compound))
# #print(len(shuffled_indices))
# unlabeled_smiles_compound = [unlabeled_smiles_compound[i] for i in shuffled_indices]
# unlabeled_smiles_adjacency = [unlabeled_smiles_adjacency[i] for i in shuffled_indices]
# unlabeled_proteins = [unlabeled_proteins[i] for i in shuffled_indices]
# unlabeled_targets = [unlabeled_targets[i] for i in shuffled_indices]
print("-"*50)
#print("Number of unlabeled samples: ", len(unlabeled_smiles_compound))
print("*"*50)
print("\n")
#--------------------COMBINE_SPIES_AND_UNLABELED_SAMPLES-----------------#


#sys.exit()


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


    model = CYPModel()

    #model.load_state_dict(torch.load('/home/yaganapu/CYP/cyp_update/benchmarks/phase1/transformercpi/Attempt_0/my_model1.pth'))

    # CUDA...
    model.to(device)


    #i=1
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
    

    #unlabeled_indices = torch.arange(len(unlabeled_smiles_compound))
    
    
    

    # RANDOMLY SELECT - NUM_SAMPLES unlabeled as training samples.
    np.random.seed(ATTEMPT + 1 + i + iteration)
    train_unlabeled_indices = np.random.choice(unlabeled_and_spies_indices, size=NUM_SAMPLES, replace=False)
    
    # unlabeled_targets_cat = torch.cat(unlabeled_targets)
    # spy_indices = list(torch.where(unlabeled_targets_cat==1)[0])
    
    untouched_unlabeled_indices = np.setdiff1d(unlabeled_and_spies_indices, train_unlabeled_indices)
    #untouched_unlabeled_indices_list = untouched_unlabeled_indices.tolist()

    #train_unlabeled_smiles_compound = unlabeled_smiles_compound[train_unlabeled_indices]
    
    #unlabeled_smiles_compound = [unlabeled_smiles_compound[i] for i in shuffled_indices]

    train_unlabeled_smiles_compound = [smile_compound_data[i] for i in train_unlabeled_indices]
    train_unlabeled_smiles_adjacency = [smile_adjacency_data[i] for i in train_unlabeled_indices]
    train_unlabeled_proteins = [protein_data[i] for i in train_unlabeled_indices]
    
    
    #train_unlabeled_targets = [unlabeled_targets[i] for i in train_unlabeled_indices]
    #train_unlabeled_targets_unchanged = [unlabeled_targets[i] for i in train_unlabeled_indices]
    train_unlabeled_targets = [targets[i] for i in train_unlabeled_indices]
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
    # concatenated_unlabeled_targets = torch.cat(unlabeled_targets)
    # untouched_unalebed_final_without_spies[i] = list(torch.where(concatenated_unlabeled_targets[untouched_unlabeled_indices_list]!=1)[0])

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
    random_indices = [z for z in range(len(train_smiles_compound))]
    np.random.seed(ATTEMPT + 1 + i + iteration)
    np.random.shuffle(random_indices)
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

            # predictions = predictions.cpu()
            # train_targets = train_targets.cpu()

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
    torch.save(model.state_dict(), DIR_NAME + '/my_model_ssnet' + str(i+1) + '.pth')
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
#         print("Obtaining predictions on the complete training dataset...")
#         j = 0
#         train_pos_predictions[i] = []

#         while(j+BATCH_SIZE  <= len(train_pos_smiles_compound)):   
#             #print(train_pos_smiles_compound[j:j+BATCH_SIZE])
#             train_pos_predictions[i] += list(model(train_pos_smiles_compound[j:j+BATCH_SIZE][0],train_pos_smiles_adjacency[j:j+BATCH_SIZE][0], train_pos_proteins[j:j+BATCH_SIZE][0]).cpu())
#             j += BATCH_SIZE

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
        #touched_spy_indices[i] = torch.nonzero(torch.tensor(train_unlabeled_targets) == 1.0)










#         # If yes, then remove them from computing Spies Capture Rate.
#         if len(touched_spy_indices[i]) > 0:
#             print("Are there any spies that are part of negatives in training set? yes, then remove them from computing Spies Capture Rate")
#             print("*"*50)
#             print(touched_spy_indices[i])
#             num_items = touched_spy_indices[i].numel()
#             # Extract values based on the number of items
#             if num_items >= 2:
#                 touched_spy_indices_list = touched_spy_indices[i].squeeze().tolist()
#             else:
#                 touched_spy_indices_list = [touched_spy_indices[i].squeeze().item()]
            
#             print(touched_spy_indices_list)
#             print("*"*50)


#             # Pad zeros along 2nd dimension (h dimension)
#     #             padded_smiles_c = F.pad(spy_smiles_compound, (0, 0, 0, zeros_to_pad_h_c))
#     #             #padded_smiles_a = F.pad(spy_smiles_adjacency, (0, 0, 0, zeros_to_pad_h_a))

#     #             # Pad zeros along 3rd dimension (w dimension)
#     #             padded_smiles_c = F.pad(padded_smiles_c, (0, zeros_to_pad_w_c, 0, 0))
#             #padded_smiles_a = F.pad(padded_smiles_a, (0, zeros_to_pad_w_a, 0, 0))


#             ###____________THIS LOGIC WILL HELP IN IDENTIFYING SPIES THAT ARE IN TRAINING SET___________###

#             # zeros_tensor = torch.zeros((touched_spy_indices[:, 0].shape[0], 1, zeros_to_append)).cuda()

#             #padded_unlabeled_smiles_c = torch.nn.functional.pad(train_unlabeled_smiles_compound[touched_spy_indices[i][:, 0], :,:], (0, num_zeros_dim3, 0, num_zeros_dim2))
            
#             train_unlabeled_smiles_compound_adding_second_dim = [train_unlabeled_smiles_compound[i].unsqueeze(1) for i in range(len(train_unlabeled_smiles_compound))]

#             padded_unlabeled_smiles_c = [torch.nn.functional.pad(train_unlabeled_smiles_compound_adding_second_dim[i], (0,  train_unlabeled_proteins[i].shape[1] - train_unlabeled_smiles_compound_adding_second_dim[i].shape[1], 0, train_unlabeled_proteins[i].shape[0] - train_unlabeled_smiles_compound_adding_second_dim[i].shape[0])) for i in touched_spy_indices_list]

#             #padded_unlabeled_smiles_a = F.pad(train_unlabeled_smiles[touched_spy_indices[:, 0], :, :,:], (0, 0, 0, zeros_to_pad_h_a))

#             #padded_unlabeled_smiles_c = F.pad(padded_unlabeled_smiles_c, (0, zeros_to_pad_w_c, 0, 0))
#             #padded_unlabeled_smiles_a = F.pad(padded_unlabeled_smiles_a, (0, zeros_to_pad_w_c, 0, 0))

#             # result_tensor2 = torch.cat((train_unlabeled_smiles[touched_spy_indices[:, 0], :, :,:], zeros_tensor), dim=2).cuda()
#             train_unlabeled_proteins_spylist = [train_unlabeled_proteins[n] for n in touched_spy_indices_list]
#             spies_in_train = [torch.add(padded_unlabeled_smiles_c[o], train_unlabeled_proteins_spylist[o]) for o in range(len(touched_spy_indices_list))]
#             print("Number of unlabeled samples in training set: ", spies_in_train[0].shape)



#             ###____________THIS LOGIC WILL HELP IN IDENTIFYING SPIES IN TRAINING SET___________###

#             #print("all_spies: ", all_spies.shape, all_spies.reshape([all_spies.shape[0], all_spies.shape[2]]).shape)
#             #print("all_spies2: ", all_spies2.shape, all_spies2.reshape([all_spies2.shape[0], all_spies2.shape[2]]).shape)

#             ###-------------------------LOGIC TO GET THE LIST OF UNTOUCHED SPIES-------------------------###
#             print("Are there any spies in training dataset?")
#             touched_spies_indices = []
#             untouched_spies_indices = []

#             for p, spy in enumerate(all_spies):
#                 if any(torch.equal(spy, s) for s in spies_in_train):
#                     touched_spies_indices.append(p)
#                 else:
#                     untouched_spies_indices.append(p)
#             ###-------------------------LOGIC TO GET THE LIST OF UNTOUCHED SPIES-------------------------###

#             print("Obtaining predictions on the spies")
#             #### ADDING IN BATCH LOGIC ####
#             ###--------------------------------------TRAIN PREDICTIONS------------------------------------------###
        untouched_spy_indices = [index for index in untouched_unlabeled_indices if index <500]    
        j = 0
        spy_predictions[i] = []
        spy_predictions_indices[i] = untouched_spy_indices

        while(j+BATCH_SIZE  <= len(untouched_spy_indices)):   
            spy_predictions[i] += list(model(smile_compound_data[untouched_spy_indices[j:j+BATCH_SIZE][0]], smile_adjacency_data[untouched_spy_indices[j:j+BATCH_SIZE][0]], protein_data[untouched_spy_indices[j:j+BATCH_SIZE][0]]).cpu())
            #spy_predictions_indices[i] += list(untouched_spies_indices[j:j+BATCH_SIZE][0])
            j += BATCH_SIZE

            # if j < len(untouched_spies_indices):
            #     spy_predictions[i] += list(model(spy_smiles_compound[untouched_spies_indices[j:len(untouched_spies_indices)]],spy_smiles_adjacency[untouched_spies_indices[j:len(untouched_spies_indices)]], spy_proteins[untouched_spies_indices[j:len(untouched_spies_indices)]]).cpu())

            ###--------------------------------------TRAIN PREDICTIONS------------------------------------------###
            #### ADDING IN BATCH LOGIC ####
            ###--------------------------------------SPY PREDICTIONS------------------------------------------------###

#         else:
#             ###--------------------------------------SPY PREDICTIONS------------------------------------------------###
#             #spy_predictions[i] = np.array(model(spy_smiles, spy_proteins).cpu())
#             print("Obtaining predictions on the spies")
#             #### ADDING IN BATCH LOGIC ####
#             ###--------------------------------------TRAIN PREDICTIONS------------------------------------------###
#             j = 0
#             spy_predictions[i] = []
#             spy_predictions_indices[i] = list([k for k in range(len(all_spies))])
#             print("untouched spies")

#             while(j+BATCH_SIZE  <= len(spy_predictions_indices[i])): 
#                 #print(spy_predictions_indices[i][j:j+BATCH_SIZE][0])
#                 spy_predictions[i] += list(model(spy_smiles_compound[spy_predictions_indices[i][j:j+BATCH_SIZE][0]], 
#                                                  spy_smiles_adjacency[spy_predictions_indices[i][j:j+BATCH_SIZE][0]], 
#                                                  spy_proteins[spy_predictions_indices[i][j:j+BATCH_SIZE][0]]).cpu())
#                 j += BATCH_SIZE

#             # if j < len(spy_smiles_compound):
#             #     spy_predictions[i] += list(model(spy_smiles_compound[j:len(spy_smiles_compound)],spy_smiles_adjacency[j:len(spy_smiles_compound)], spy_proteins[j:len(spy_proteins)]).cpu())
#             ###--------------------------------------TRAIN PREDICTIONS------------------------------------------###
#             #### ADDING IN BATCH LOGIC ####
#             ###--------------------------------------SPY PREDICTIONS------------------------------------------------###

        print("Length of spy_predictions: ", len(spy_predictions[i]))
        #print("Length of touched spies: ", len(touched_spy_indices[i]))
        
        
        
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
        
        untouched_test_indices = [index for index in untouched_unlabeled_indices if index >=500]
        j=0
        #temp = []
        unlabeled_predictions[i]=[]
        
        

        while(j+BATCH_SIZE  <= len(untouched_test_indices)):
            unlabeled_predictions[i] += list(model(smile_compound_data[untouched_test_indices[j:j+BATCH_SIZE][0]], 
                                                         smile_adjacency_data[untouched_test_indices[j:j+BATCH_SIZE][0]],
                                                         protein_data[untouched_test_indices[j:j+BATCH_SIZE][0]]).cpu())
            j += BATCH_SIZE
            
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
                                             
        i_df = pd.DataFrame(columns=["original_index", "prediction_score"])
        i_df["original_index"] = untouched_test_indices
        i_df["prediction_score"] = unlabeled_predictions[i]
            
        i_df.to_csv(DIR_NAME +  "/predictions_bin#_" + str(i) + ".csv", index=False)                                     
            
#         # Define the file name
#         csv_file = DIR_NAME + '/unlabeled_predictions_bin_'  + str(i) + '.csv'
#         smiles_path = DIR_NAME + '/unlabeled_predictions_bin_smiles_'  + str(i) + '.npy'
#         proteins_path = DIR_NAME + '/unlabeled_predictions_bin_proteins_'  + str(i) + '.npy'
#         index_path = DIR_NAME + '/unlabeled_predictions_bin_index'  + str(i) + '.npy'
#         scores_path = DIR_NAME + '/unlabeled_predictions_bin_scores_'  + str(i) + '.npy'    
        
#         # Convert the list of tensors to a list of NumPy arrays
        
#         list_of_smiles = np.array([unlabeled_smiles_compound[f].cpu().detach().numpy() for f in indices])
#         list_of_proteins = np.array([unlabeled_proteins[f].cpu().detach().numpy() for f in indices])
        
#         list_of_index = np.array(indices)
#         list_of_scores = np.array([tensor.cpu().detach().numpy() for tensor in unlabeled_predictions[i]])
        
#         np.save(smiles_path, list_of_smiles)
#         np.save(proteins_path, list_of_proteins)
#         np.save(index_path, list_of_index)
#         np.save(scores_path, list_of_scores)    
    

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

        #print(f"numpy files '{csv_file}' has been created successfully.")
        # Create a histogram using Seaborn
        unlabeled_predictions[i] = [float(val) for val in unlabeled_predictions[i]]
        #print(spy_predictions[i])
        sns.histplot(unlabeled_predictions[i], bins=10, kde=True, color='skyblue')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Histogram of unlabeled predictions')
        plt.savefig(DIR_NAME + '/unlabeled_predictions_histogram_' + 'repeat' + str(i) + '.png')
        plt.clf()



    #---------------------PRINT STATEMENTS---------------------------------#
    #print("Length of all train predictions: ", len(train_predictions[i]))
    #print("Length of all train pos predictions: ", len(train_pos_predictions[i]))
    print("Length of all spy predictions: ", len(spy_predictions[i]))
    #print("Length of all unlabeled predictions: ", len(unlabeled_predictions[i]))
    #---------------------PRINT STATEMENTS---------------------------------#



    #---------------------METRICS-----------------------------#

    print("*"*50)
    print("-"*50)
    print("Training Positives Metrics: ")
    # tpr = sum([1 for pred in train_pos_predictions[i] if pred >= 0.5])
    # fpr = sum([1 for pred in train_pos_predictions[i] if pred < 0.5])
    # print("TPR: ", tpr/NUM_SAMPLES)
    # print("FPR: ", fpr/NUM_SAMPLES)
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
# original_df = pd.read_csv("/home/yaganapu/CYP/cyp_update/benchmarks/different_preprocessing/ssnet_data_pdb/combined_data_no_duplicates_69549_for_ssnet.csv")
# original_df["new_class"] = 1
# mask = original_df["class"]==0
# original_df.loc[mask, "new_class"] = unlabeled_classes
# original_df.to_csv(DIR_NAME + 'data_for_phase2.csv')

