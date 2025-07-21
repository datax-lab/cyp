import random
import pickle
import sys
import timeit

import numpy as np
from sklearn.metrics import roc_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score, precision_score, recall_score

radius=2
# radius=3

# ngram=2
ngram=3
dim=10
layer_gnn=3
side=5
window=11
layer_cnn=3
layer_output=3
lr=1e-3
lr_decay=0.5
decay_interval=10
weight_decay=1e-6
n_fingerprint = 3000

class CYPModel(nn.Module):
    def __init__(self):
        super(CYPModel, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        #proteins = []
        #f = np.load('../../../final_data_dude_rd2_opp.npy').item()
        #self.proteins = torch.from_numpy(np.array(f['proteins']).reshape((-1,)))
        self.w1 = self.branch_conv(5, 2, 28)
        self.w11 = self.branch_conv(5, 28, 64)
        self.w111 = self.branch_conv(5, 64, 128)
        self.w2 = self.branch_conv(10, 2, 28)
        self.w22 = self.branch_conv(10, 28, 64)
        self.w222 = self.branch_conv(10, 64, 128)
        self.w3 = self.branch_conv(15, 2, 28)
        self.w33 = self.branch_conv(15, 28, 64)
        self.w333 = self.branch_conv(15, 64, 128)
        self.w4 = self.branch_conv(20, 2, 28)
        self.w44 = self.branch_conv(20, 28, 64)
        self.w444 = self.branch_conv(20, 64, 128)
        self.embed_word = nn.Linear(512*2, dim)
        #self.bc3 = self.branch_conv(15)
        #print (self.bc1.shape, self.bc2.shape, self.bc3.shape)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        #self.W_attention = nn.Linear(dim, dim)
        self.W_attention = nn.Linear(dim, dim)
        #self.W_attention_prot = nn.Linear(100, 34)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2*dim, 1)

    def branch_conv(self, kernel_size = 5, in_channel = 1, out_channel = 1):
        return nn.Conv1d(
                     in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size,
                     stride=1) # remove cuda when using CPU !!

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""

        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs.unsqueeze(1))).squeeze(1)
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention_prot(xs))
        weights = torch.tanh(F.linear(h, hs))
        # weights = torch.tanh(torch.matmul(h.unsqueeze(1), hs.permute(0, 2, 1)))
        # weights = weights.squeeze(1).unsqueeze(-1)
        ys = torch.matmul(weights,hs)

        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self,smile_compound_data, smile_adjacency_data, protein_data):

        fingerprints, adjacency, words = smile_compound_data, smile_adjacency_data, protein_data
        #fingerprint_vectors = fingerprints
        #word_vectors = words

        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, layer_gnn)
        #print ("compound vector after gnn:", compound_vector.shape)

        """Protein vector with attention-CNN."""
        #word_vectors = self.embed_word(words)
        #print ("protein_vector:", words.shape)
        #words = words.unsqueeze(1)
        words = words.reshape((-1,2,9000))
        #words = torch.tensor(words).to(device)
        
        #print(words.shape)
        #print(words.dtype)
        w1 = self.w1(words)
        #print ("w1_shape:",w1.shape)
        w1 = self.w11(w1)
        #print ("w1_shape:",w1.shape)
        w1 = self.w111(w1)
        #print ("w1_shape:",w1.shape)
        #w1 = self.bc1(w1)
        w_1 = F.avg_pool1d(w1, kernel_size=w1.size()[-1]).view(-1,)
        #print ("w_1_shape:",w_1.shape)
        u = w_1.repeat(w1.shape[-1]).view(-1, w1.shape[-1])
        #print ("u_shape:",u.shape)
        wu_vec = (u-w1)**2
        #print ("wu_vec_shape:",wu_vec.shape)
        wu = F.avg_pool1d(wu_vec, kernel_size=wu_vec.size()[-1]).view(-1,) 
        #print ("wu_shape:",wu.shape)
        w1 = torch.cat((w_1, wu), 0).view(-1,1)
        #print ("w1_shape:",w1.shape)
        
        #stop()

        #words = words.reshape((-1, 1, 18000))
        w2 = self.w2(words)
        #print ("w2_shape:",w2.shape)
        w2 = self.w22(w2)
        #print ("w2_shape:",w2.shape)
        w2 = self.w222(w2)
        #print ("w2_shape:",w2.shape)
        #w1 = self.bc1(w1)
        w_2 = F.avg_pool1d(w2, kernel_size=w2.size()[-1]).view(-1,)
        #print ("w_2_shape:",w_2.shape)
        #print (w1.shape)
        u = w_2.repeat(w2.shape[-1]).view(-1, w2.shape[-1])
        #print ("u_shape:",u.shape)
        wu_vec = (u-w2)**2
        #print ("wu_vec_shape:",wu_vec.shape)
        wu = F.avg_pool1d(wu_vec, kernel_size=wu_vec.size()[-1]).view(-1,)
        #print ("wu_shape:",wu.shape)
        w2 = torch.cat((w_2, wu), 0).view(-1, 1)
        #print ("w2 shape:",w2.shape)
        #print ("w2_shape:",w2.shape)

        #w2 = F.avg_pool1d(w2, kernel_size=w2.size()[-1]).view(-1, 1)

        w3 = self.w3(words)
        #print ("w3_shape:",w3.shape)
        w3 = self.w33(w3)
        #print ("w3_shape:",w3.shape)
        w3 = self.w333(w3)
        #print ("w3_shape:",w3.shape)
        #w1 = self.bc1(w1)
        w_3 = F.avg_pool1d(w3, kernel_size=w3.size()[-1]).view(-1,)
        #print ("w_3_shape:",w_3.shape)
        #print (w1.shape)
        u = w_3.repeat(w3.shape[-1]).view(-1, w3.shape[-1])
        #print ("u_shape:",u.shape)
        wu_vec = (u-w3)**2
        #print ("wu_vec_shape:",wu_vec.shape)
        wu = F.avg_pool1d(wu_vec, kernel_size=wu_vec.size()[-1]).view(-1,)
        #print ("wu_shape:",wu.shape)
        w3 = torch.cat((w_3, wu), 0).view(-1, 1)
        #print ("w3 shape:",w3.shape)

        #w3 = F.avg_pool1d(w3, kernel_size=w3.size()[-1]).view(-1, 1)

        w4 = self.w4(words)
        #print ("w4_shape:",w4.shape)
        w4 = self.w44(w4)
        #print ("w4_shape:",w4.shape)
        w4 = self.w444(w4)
        #print ("w4_shape:",w4.shape)
        #w1 = self.bc1(w1)
        w_4 = F.avg_pool1d(w4, kernel_size=w4.size()[-1]).view(-1,)
        #print ("w_4_shape:",w_4.shape)
        #print (w1.shape)
        u = w_4.repeat(w4.shape[-1]).view(-1, w4.shape[-1])
        #print ("u_shape:",u.shape)
        wu_vec = (u-w4)**2
        #print ("wu_vec_shape:",wu_vec.shape)
        wu = F.avg_pool1d(wu_vec, kernel_size=wu_vec.size()[-1]).view(-1,)
        #print ("wu_shape:",wu.shape)
        w4 = torch.cat((w_4, wu), 0).view(-1, 1)
        #print ("w4 shape:",w4.shape)

        #w4 = F.avg_pool1d(w4, kernel_size=w4.size()[-1]).view(-1, 1)

        #print (w1.shape, w2.shape)
        word_vectors = torch.cat((w1, w2, w3, w4), 0).view(-1,)
        #print ("word_vectors:",word_vectors.shape)
        #words_vectors = self.embed_word(word_vectors)
        protein_vector = self.embed_word(word_vectors)#word_vectors
        protein_vector = protein_vector.view(1, dim)
        #print (protein_vector.shape)
        #protein_vector = self.attention_cnn(compound_vector,
        #                                    word_vectors, layer_cnn)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        #print("cat vector shape:",cat_vector.shape)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector).squeeze(0)
        #print("interaction:",interaction.shape)

        return torch.sigmoid(interaction)

