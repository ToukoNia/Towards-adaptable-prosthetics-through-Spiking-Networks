# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 17:28:00 2025

@author: Nia Touko
"""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import matplotlib.pyplot as plt

numChannels=14
class Net_SLSTM(nn.Module):

    def __init__(self, inputSize=14, hiddenSize=128, numClasses=8):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.numClasses = numClasses
        self.slstm1 = snn.SLSTM(inputSize, hiddenSize, spike_grad=surrogate.fast_sigmoid(),learn_threshold=True,reset_mechanism="subtract")
        self.slstm2 = snn.SLSTM(hiddenSize, hiddenSize, spike_grad=surrogate.fast_sigmoid(),learn_threshold=True,reset_mechanism="subtract")
        self.bn1 = nn.BatchNorm1d(hiddenSize)
        self.bn2=nn.BatchNorm1d(hiddenSize)
        self.fc = nn.Linear(hiddenSize, numClasses)   #output

    def forward(self, x):  # x: [time, batch, features]
        syn1, mem1 = self.slstm1.init_slstm()
        syn2, mem2 = self.slstm2.init_slstm()
        mem2Rec = []

        for step in range(x.size(0)):
            spk1, syn1, mem1 = self.slstm1(x[step], syn1, mem1)
            #spk1 = self.bn1(spk1) 
            spk2, syn2, mem2 = self.slstm2(spk1, syn2, mem2)
            #spk2=self.bn2(spk2)
            #spk2rec.append(spk2)
            mem2Rec.append(mem2)
        #Gonna try swapping mem2 with spk2 and see if I get increased accuracy next
        #spk2rec=torch.stack(spk2rec)
        #finalSpk=spk2rec.mean(dim=0)
        #out=self.fc(finalSpk)
        mem2Rec = torch.stack(mem2Rec)
        finalMem = mem2Rec.mean(dim=0)
        out = self.fc(finalMem)
        return out 


class Net_SLSTM_Conv(nn.Module):
    def __init__(self, inputSize=numChannels, hiddenSize=128, numClasses=8):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.numClasses = numClasses
        
        self.conv1 = nn.Conv1d(in_channels=numChannels, out_channels=32, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=0.9) # A simple spiking neuron layer
        
        self.slstm1 = snn.SLSTM(32, hiddenSize, spike_grad=surrogate.fast_sigmoid(),learn_threshold=True,reset_mechanism="subtract")
        self.slstm2 = snn.SLSTM(hiddenSize, hiddenSize, spike_grad=surrogate.fast_sigmoid(),learn_threshold=True,reset_mechanism="subtract")
        self.bn1 = nn.BatchNorm1d(hiddenSize)
        #self.bn2 = nn.BatchNorm1d(hiddenSize)
        self.fc = nn.Linear(hiddenSize, numClasses)   #output
    def forward(self, x):  # x: [time, batch, features]
        
        self.lif1.reset_mem()
        x=x.permute(0,2,1)
        cur1spk,cur1mem = self.lif1(self.conv1(x))
        cur1=cur1spk.permute(0,2,1)
        syn1, mem1 = self.slstm1.init_slstm()
        syn2, mem2 = self.slstm2.init_slstm()
        mem2Rec = []
        spk1Rec=[]
        #spk2Rec=[]
        for step in range(x.size(0)):
            spk1, syn1, mem1 = self.slstm1(cur1[step], syn1, mem1)
            spk1Rec.append(spk1)
        
        #TAB 
        spk1NormRec=self.roughTemporalAccumulatedBN(spk1Rec,self.bn1)
        
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

class AdaptiveNet_SLSTM(nn.Module): #low preformance (50% acc loss of 1.75) but preforms near identically on unseen session vs seen session (loss difference of 0.01)
    def __init__(self,inputSize=14, hiddenSize=128, numClasses=8):
        super().__init__()
        self.s_lstmNet=Net_SLSTM_Conv(inputSize,hiddenSize,numClasses)
    def adaptiveDeltaModulation(self,x,theta=2.5):
        diff=torch.diff(x,dim=1)
        meanDiff=torch.mean(diff,dim=1,keepdim=True)
        stdDiff=torch.std(diff,dim=1,keepdim=True)
        adaptiveThreshold=meanDiff+theta*stdDiff
        spikeTrain=(torch.abs(diff)>adaptiveThreshold).float()
        padding=torch.zeros(x.shape[0], 1, x.shape[2],device=x.device)
        spikeTrainPadded=torch.cat((padding,spikeTrain),dim=1)
        return spikeTrainPadded
    def forward(self,x):
        spikeData=self.adaptiveDeltaModulation(x)
        output=self.s_lstmNet(spikeData)
        return output
    def plotSpikes(self,x):
        smplData = x[0].unsqueeze(0) # Shape: [1, timesteps, features]
        thetaValues = [0.5, 1.0, 1.5, 2.5]
        fig, axes = plt.subplots(len(thetaValues) + 1, 1, figsize=(12, 9), sharex=True)
        fig.suptitle("Effect of Theta on Spike Generation", fontsize=16)
        axes[0].plot(smplData.squeeze()[:, 0].cpu().numpy(), label="Original Signal (Feature 0)")
        axes[0].set_title("Original Signal")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        for i, theta in enumerate(thetaValues):
            spikeTrain = self.adaptiveDeltaModulation(smplData, theta=theta)
            sparsity = spikeTrain.mean() * 100
            spikeIdx = spikeTrain.squeeze().nonzero().cpu().numpy()
            axes[i+1].scatter(spikeIdx[:, 0], spikeIdx[:, 1], marker='|', s=50)
            axes[i+1].set_title(f"Theta = {theta:.1f}  |  Sparsity = {sparsity:.3f}%")
            axes[i+1].set_ylabel("Feature")
            axes[i+1].grid(True, alpha=0.3)
            axes[i+1].set_ylim(-0.5, smplData.shape[2] - 0.5)
        axes[-1].set_xlabel("Time Step")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
