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

import math


radius=2
# radius=3

# ngram=2
ngram=3

dim=34
layer_gnn=3
side=5
window=(2*side+1)
layer_cnn=3
layer_output=3
#lr=1e-3
#lr_decay=0.5
#decay_interval=10
#weight_decay=1e-6
#iteration=100

n_fingerprint = 2000
n_words = 7000


class CYPModel(nn.Module):
    def __init__(self):
        super(CYPModel, self).__init__()
        #self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        #self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=23,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(34, 34)
        self.W_attention_prot = nn.Linear(100, 34)
    
        
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2*dim, 1)

    def gnn(self, xs, A, layer):
        #print("xs shape:", xs.shape)
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            #print("hs shape", hs.shape)
            xs = xs + torch.matmul(A, hs)
            #print("xs shape", xs.shape)
            
        #print("xs shape after gnn", xs.shape)    
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.mean(xs, dim=1)

    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""

        #xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs.unsqueeze(1))).squeeze(1)
            
    
        #xs = torch.squeeze(torch.squeeze(xs, 0), 0)
        #print("after cnn layers shape of xs:", xs.shape)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention_prot(xs))
        #print("h shape:", h.shape)
        #print("hs.shape:", hs.shape)
        weights = torch.tanh(torch.matmul(h.unsqueeze(1), hs.permute(0, 2, 1)))
        #print("weights shape:", weights.shape)
        weights = weights.squeeze(1).unsqueeze(-1)
        ys = torch.mul(weights , hs)
        #print("ys shape:", ys.shape)

        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.mean(ys, dim=1)

    def forward(self,smile_compound_data, smile_adjacency_data, protein_data):

        fingerprints, adjacency, words = smile_compound_data, smile_adjacency_data, protein_data
        fingerprint_vectors = fingerprints
        word_vectors = words

        #print("compound_vector_shape:", fingerprint_vectors.shape)
        #print("adjacency_vector_shape:", adjacency.shape)
        #print("protein_vector_shape:", word_vectors.shape)
        """Compound vector with GNN."""
        #fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, layer_gnn)
        #print("after gnn compound_vector_shape:", compound_vector.shape)

        """Protein vector with attention-CNN."""
        #word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(compound_vector,
                                            word_vectors, layer_cnn)
        
        #print("after attention_cnn protein_vector_shape:", protein_vector.shape)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        
        #print("cat_Vector",cat_vector.shape)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector).squeeze(0)
        #print("interaction_Vector",interaction.shape)
        

        return torch.sigmoid(interaction)
