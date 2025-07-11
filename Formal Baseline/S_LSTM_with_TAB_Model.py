# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 14:14:53 2025

@author: Nia Touko
"""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

numChannels=14
class Net_SLSTM(nn.Module):
    def __init__(self, inputSize=numChannels, hiddenSize=128, numClasses=8,windowSize=400): 
        super().__init__()
        self.hiddenSize = hiddenSize
        self.numClasses = numClasses
        self.slstm1 = snn.SLSTM(inputSize, hiddenSize, spike_grad=surrogate.fast_sigmoid(),learn_threshold=True,reset_mechanism="subtract")
        self.slstm2 = snn.SLSTM(hiddenSize, hiddenSize, spike_grad=surrogate.fast_sigmoid(),learn_threshold=True,reset_mechanism="subtract")
        
        # Instantiate the adapted TAB layers for input and hidden spikes
        self.tab_input = TAB_Layer(num_features=inputSize, time_steps=windowSize)
        self.tab_hidden = TAB_Layer(num_features=hiddenSize, time_steps=windowSize)
        
        #self.bn2 = nn.BatchNorm1d(hiddenSize)
        self.fc = nn.Linear(hiddenSize, numClasses)   #output
    def forward(self, x):  # x: [time, batch, features]
        syn1, mem1 = self.slstm1.init_slstm()
        syn2, mem2 = self.slstm2.init_slstm()
        mem2Rec = []
        spk1Rec=[]
        #spk2Rec=[]
        inputNorm=self.roughTemporalAccumulatedBN(spk1Rec,self.bn1)
        for step in range(x.size(0)):
            spk1, syn1, mem1 = self.slstm1(inputNorm[step], syn1, mem1)
            spk1Rec.append(spk1)
        #TAB 
        spk1NormRec=self.roughTemporalAccumulatedBN(spk1Rec,self.bnh)
        for step in range(x.size(0)):
            spk2, syn2, mem2 = self.slstm2(spk1NormRec[step], syn2, mem2)
            mem2Rec.append(mem2)
        mem2Rec = torch.stack(mem2Rec)
        finalMem = mem2Rec.mean(dim=0)
        out = self.fc(finalMem)
        return out
    def roughTemporalAccumulatedBN(self,spkRec,bn):
        spkRec=torch.stack(spkRec)
        tSteps,bSize,nFeatures=spkRec.shape
        spkFlat=spkRec.view(tSteps*bSize,nFeatures)
        spkNormFlat=bn(spkFlat)
        spkNormRec=spkNormFlat.view(tSteps,bSize,nFeatures)
        return spkNormRec


def _prob_check(p):
    p2 = p**2
    return p2

class TAB_Layer(nn.Module): ## Taken from https://github.com/HaiyanJiang/SNN-TAB/blob/main/models.py and adapted for 1d layers
    def __init__(self, num_features, time_steps=4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(TAB_Layer, self).__init__()
        self.time_steps = time_steps
        self.bn_list = nn.ModuleList([nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats) for i in range(time_steps)])
        self.p = nn.Parameter(torch.ones(time_steps, 1, 1))

    def forward(self, x):
        assert x.dim() == 3 and x.shape[1] == self.time_steps, \
            f"Input shape must be [Batch, Time, Features]. Got {x.shape}"

        p_checked = _prob_check(self.p).to(x.device)
        y_res = []

        for t in range(self.time_steps):
            x_slice = x[:, 0:(t + 1), :].clone() # Shape: [Batch, t+1, Features]
            # Reshape for BatchNorm1d: combine batch and time dimensions
            # [Batch, t+1, Features] -> [Batch * (t+1), Features]
            x_reshaped = x_slice.contiguous().view(-1, x_slice.shape[2])
            y_reshaped = self.bn_list[t](x_reshaped)
            # [Batch * (t+1), Features] -> [Batch, t+1, Features]
            y_slice = y_reshaped.view(x_slice.shape)
            y_res.append(y_slice[:, t, :])
        y_res = torch.stack(y_res, dim=1) # Shape: [Batch, Time, Features]
        y = y_res * p_checked.permute(1, 0, 2)
        
        return y