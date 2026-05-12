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
from torch.optim.lr_scheduler import StepLR

from sklearn.model_selection import train_test_split

from cyp_model_cpi_prediction import *

import sys
sys.path.append('/home/yaganapu/CYP/cyp_update/benchmarks/phase2')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

#alpha = torch.tensor(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
alpha = torch.tensor(0.2)
print(alpha)
device = torch.device('cuda')

BASE_PATH = "/home/yaganapu/CYP/cyp_update/benchmarks/phase2/"
COMPOUND_DATA_PATH = "/home/yaganapu/CYP/cyp_update/benchmarks/bacteria_data/transformercpi/new_padded_compounds.npy"
ADJACENCY_DATA_PATH = "/home/yaganapu/CYP/cyp_update/benchmarks/bacteria_data/transformercpi/new_padded_adjacencies.npy"
PROTEIN_DATA_PATH = "/home/yaganapu/CYP/cyp_update/benchmarks/bacteria_data/transformercpi/new_padded_proteins.npy"
TARGET_DATA_PATH = "/home/yaganapu/CYP/cyp_update/benchmarks/bacteria_data/transformercpi/interactions.npy"


#--------------------------------Attempt_* FOLDER_CREATION--------------------------------#
DIR_NAME = BASE_PATH + '/transformercpi/' + 'checkingloss_10bins_kp_Attempt_' + str(0) + '/'
if(os.path.exists(DIR_NAME)):
    None
else:
    os.mkdir(DIR_NAME)
#--------------------------------Attempt_* FOLDER_CREATION--------------------------------#

# Loading the Preprocessed Datasets
tt_smile_compound_data = torch.tensor(np.load(COMPOUND_DATA_PATH, allow_pickle = True)).float()
tt_smile_adjacency_data = torch.tensor(np.load(ADJACENCY_DATA_PATH, allow_pickle = True)).float()
tt_protein_data = torch.tensor(np.load(PROTEIN_DATA_PATH, allow_pickle = True)).float()
targets = torch.tensor(np.load(TARGET_DATA_PATH, allow_pickle = True)).float()

#--------------------------------SMILE_DATA_PREP--------------------------------#
print("*"*50)

print("Shape of smile data after aligining with first CNN layer: ", tt_smile_compound_data.shape)

print("Shape of smile data after aligining with first CNN layer: ", tt_smile_adjacency_data.shape)

#--------------------------------SMILE_DATA_PREP--------------------------------#

#--------------------------------PROTEIN_DATA_PREP--------------------------------#
print("*"*50)

print("Shape of protein data after aligining with first CNN layer: ", tt_protein_data.shape)

#--------------------------------PROTEIN_DATA_PREP--------------------------------#

# Complete Results
complete_results = pd.read_csv('/home/yaganapu/CYP/cyp_update/benchmarks/phase2/transformer_data_for_phase2_10bins.csv')
print("Value Counts in Raw Data: \n", complete_results["new_class"].value_counts())
print("Value Counts in Raw Data: \n", complete_results["class"].value_counts())

complete_results = complete_results[complete_results["new_class"].isin([0, 1])].reset_index()

complete_results["labels"] = complete_results["class"]

# If the original label is unknown but the Predicted Class is 1.
mask = (complete_results["class"]==0) & (complete_results["new_class"]==1)
complete_results.loc[mask, "Labels"] = 1

print("Complete Results: \n", complete_results["Labels"].value_counts())


tt_smile_compound_data = tt_smile_compound_data[complete_results["index"]]
tt_smile_adjacency_data = tt_smile_adjacency_data[complete_results["index"]]

tt_protein_data = tt_protein_data[complete_results["index"]]

ori_targets = torch.tensor(complete_results["class"].values).float()

pred_targets = torch.tensor(complete_results["new_class"].values).float()

known_positive_samples = torch.where(ori_targets==1)[0]
pseudo_positive_samples = torch.where((ori_targets == 0) & (pred_targets == 1))[0]
pseudo_negative_samples = torch.where((ori_targets == 0) & (pred_targets == 0))[0]
all_samples = list(known_positive_samples) + list(pseudo_positive_samples) + list(pseudo_negative_samples) 

print("Sample Counts: ", len(known_positive_samples), len(pseudo_positive_samples), len(pseudo_negative_samples))
print("All Samples: ", len(all_samples))

#alpha = torch.tensor(0.4)
#custom_loss
class CustomLoss(nn.Module):
    def __init__(self, class_weights):
        super(CustomLoss, self).__init__()
        self.class_weights = class_weights
        

    def forward(self, input, ori_target, pred_target):
        # input: raw output from your model
        # ori_target: ground truth labels
        #pred_target: phase 1 labels
        
        
        
        bce_loss = - (self.class_weights[0] * pred_target * torch.log(input) +
              self.class_weights[1] * (1 - pred_target) * torch.log(1 - input))
        
        # Calculate mean squared log error loss
        
       
        msle_loss = torch.zeros_like(bce_loss)
        condition = (ori_target == 1) & (pred_target == 1)
        #msle_loss[condition] = (pred_target * torch.log(input))[condition]
        msle_loss[condition] = (torch.log1p(input) - torch.log1p(pred_target))[condition] ** 2
        
        # kl_divergence = 0.5 * (torch.mean(torch.exp(bce_loss) - 1 + torch.square(input)) +
        #                        torch.mean(torch.exp(msle_loss) - 1 + torch.square(input)))
        
        #sqrt_kl_divergence = torch.sqrt(kl_divergence)
        
        #Total_loss = (bce_loss) + (msle_loss)
        Total_loss =  (bce_loss) + (alpha) * (msle_loss)

        
        # Calculate the mean loss over the batch
        custom_loss = torch.mean(Total_loss)

        return custom_loss,torch.mean(msle_loss),torch.mean(bce_loss)

    
phase = 2
test_accuracy_final= []
train_accuracy_final = []

protein_dim = 100
atom_dim = 34
hid_dim = 64
n_layers = 3
n_heads = 8
pf_dim = 256
dropout = 0.1

kernel_size = 5



for m in range(1):
    
    X = all_samples
    y_p = pred_targets[all_samples].cpu()
    y_o = ori_targets[all_samples].cpu()
    
    print("Before Train Test Split: ", len(X), y_p.shape)
    print("Before Train Test Split: ", np.unique(y_p, return_counts=True))
    trainIndices = X.copy()
    np.random.shuffle(trainIndices)
    
    num_epochs = 1000
    
    smileTrainData_compound = tt_smile_compound_data[trainIndices].cuda()
    smileTrainData_adjacency = tt_smile_adjacency_data[trainIndices].cuda()
    
    proteinTrainData = tt_protein_data[trainIndices].cuda()
    
    Train_ori_Targets = ori_targets[trainIndices].cuda()
    
    Train_pred_Targets = pred_targets[trainIndices].cuda()
    
    print("Train Targets Distribution...")
    print("Class 0: ", sum(Train_pred_Targets==0.))
    print("Class 1: ", sum(Train_pred_Targets==1.))
    
    class_weights = torch.tensor([len(Train_pred_Targets)/(2*sum(Train_pred_Targets==0)), len(Train_pred_Targets)/(2*sum(Train_pred_Targets==1))])
    print("Unique: ", np.unique(Train_pred_Targets.cpu(), return_counts=True))
    print("Class Weights: ", class_weights)
    
    # encoder = Encoder(protein_dim, hid_dim, 3, kernel_size, dropout, device)
    # decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device )
    # model = Predictor(encoder, decoder, device)
    model = CYPModel()
    
    model.to(device)
    #model = nn.DataParallel(model)
    
    criterion = CustomLoss(class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.000001,weight_decay = 1e-5)
    #scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    loss_values = []
    msle_values = []
    bce_values = []
    #kl_values = []
    for epoch in tqdm(range(num_epochs)):
        torch.cuda.empty_cache()
        
        # print(epoch)

        BatchSize = 64
        k = 0
        loss_dict1 = []
        msle_dict = []
        bce_dict = []
        #kl_dict = []
        while(k <= len(trainIndices)):
            # print(len(trainIndices))
            # print(k)
            optimizer.zero_grad()

        # Forward pass
            predictions = model(smileTrainData_compound[k:k+BatchSize], smileTrainData_adjacency[k:k+BatchSize], proteinTrainData[k:k+BatchSize])
            targets_pred =  Train_pred_Targets[k:k+BatchSize]
            targets_ori = Train_ori_Targets[k:k+BatchSize]
            

        # Compute the loss
            loss, msle_loss, bce_loss = criterion(predictions, targets_ori, targets_pred )

        # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            #scheduler.step()
            loss_dict1.append(loss.item())
            msle_dict.append(msle_loss.item())
            bce_dict.append(bce_loss.item())
            #kl_dict.append(kl.item())

            k += BatchSize

        loss_o = np.mean(loss_dict1)
        msle_l = np.mean(msle_dict)
        bce_l = np.mean(bce_dict)
        #kl_l = np.mean(kl_dict)

        if(epoch % 1 == 0):

            print(f'Epoch: {epoch}, Loss: {loss_o} ,MSLE: {msle_l} , BCE : {bce_l}', flush = True)
        loss_values.append(loss_o)
        msle_values.append(msle_l)
        bce_values.append(bce_l)
        #kl_values.append(kl_l)

        


    torch.save(model.state_dict(), DIR_NAME + '/my_model_cl' + str(m) + '.pth')
    
    # Plotting the loss graph
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), loss_values, label='Total Loss', color='blue', linestyle='-')
    plt.plot(range(1, num_epochs + 1), msle_values, label='MSLE Loss', color='green', linestyle='--')
    plt.plot(range(1, num_epochs + 1), bce_values, label='BCE Loss', color='red', linestyle=':')
    #plt.plot(range(1, num_epochs + 1), kl_values, label='KL Loss', color='black', linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(DIR_NAME + '/loss_graph.png')
    plt.show()
    
    torch.cuda.empty_cache()


#     with torch.no_grad():
#         model.eval().cuda()
#         train_predictions = model(smileTrainData_compound,smileTrainData_adjacency, proteinTrainData).cpu()
    


#     y_train_pred = [1 if i > 0.5 else 0 for i in train_predictions]
#     y_train_true = [i for i in Train_pred_Targets]
#     print("Train Accuracy: ", sum([1 if i==j else 0 for i, j in zip(y_train_pred, y_train_true)])/len(y_train_true))
#     train_accuracy_final.append(sum([1 if i==j else 0 for i, j in zip(y_train_pred, y_train_true)])/len(y_train_true))

#     train = train_predictions
    
#     train_ = train.numpy()
    
#     trainlist_ = [tensor.tolist() for tensor in trainIndices]
    
#     complete_results.loc[trainlist_, "Phase2_Probabilities"] = train_



#     complete_results.loc[trainlist_, "Phase2_Labels"] = y_train_pred
    


#     complete_results.head()


   


#     complete_results.to_csv(DIR_NAME+"/Phase2_Results_%s.csv"%m, index=False)
# np.save(DIR_NAME+"train_accuracy",np.array(train_accuracy_final))



