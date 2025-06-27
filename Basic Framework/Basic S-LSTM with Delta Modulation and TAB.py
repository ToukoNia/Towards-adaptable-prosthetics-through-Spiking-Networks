# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 20:23:22 2025

@author: Nia Touko
"""

import torch
from torch.nn.utils import weight_norm 
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import TensorDataset, DataLoader      #,random_split
from snntorch import spikegen
import scipy
import numpy as np
from tqdm import tqdm
import glob
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Data parameters
numSubjects = 1  
numGestures = 7
numRepetitions = 12
numChannels = 14
windowSize=400

def loadDataset(mat_paths, windowSize=windowSize, strideR=0.5, numClasses=numGestures): #def better ways to implement this for multi files but i was going quickly, also might be worth changing the labels to be like bianry strings or smthing later to reduce overfitting?
    x,y=[],[]
    stride=int(strideR*windowSize)
    for mat_path in tqdm(mat_paths, desc="Files"):
        mat = scipy.io.loadmat(mat_path)
        emg = mat['emg']             
        labels = mat['restimulus']   
        emg = np.delete(emg, [8, 9], axis=1)    #don't contain information
        emg = torch.tensor(emg, dtype=torch.float32)    
        labels = torch.tensor(labels, dtype=torch.long).squeeze()
    
        # Sliding window segmentation for batches (idk if this should be done when not doing tdnn work on transhumeral data)
        for i in range(0, emg.shape[0] - windowSize, stride):
            segment = emg[i:i+windowSize]              # [window_size, channels]
            label_window = labels[i:i+windowSize]
            label = int((torch.mode(label_window)[0])/2)    #floors it after dividing by 2 bc 2 sets of the same gesture created per thingy
            encodedSegment=spikegen.delta(segment,threshold=1e-5)
            x.append(encodedSegment)
            y.append(label)
    x = torch.stack(x)    # [num_samples, window_size, num_channels]
    y = torch.tensor(y)   # [num_samples]
 
    #probably should add a fallout for if std is 0 or add an epsilon but ah well     
    return TensorDataset(x, y)

def roughTemporalAccumulatedBN(spkRec,bn):
    spkRec=torch.stack(spkRec)
    tSteps,bSize,nFeatures=spkRec.shape
    spkFlat=spkRec.view(tSteps*bSize,nFeatures)
    spkNormFlat=bn(spkFlat)
    spkNormRec=spkNormFlat.view(tSteps,bSize,nFeatures)
    return spkNormRec

class Net_SLSTM(nn.Module):
    def __init__(self, inputSize=numChannels, hiddenSize=128, numClasses=7):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.numClasses = numClasses

        self.slstm1 = snn.SLSTM(inputSize, hiddenSize, spike_grad=surrogate.fast_sigmoid(),learn_threshold=True,reset_mechanism="subtract")
        self.slstm2 = snn.SLSTM(hiddenSize, hiddenSize, spike_grad=surrogate.fast_sigmoid(),learn_threshold=True,reset_mechanism="subtract")
        self.bn1 = nn.BatchNorm1d(hiddenSize)
        #self.bn2 = nn.BatchNorm1d(hiddenSize)
        self.fc = nn.Linear(hiddenSize, numClasses)   #output
    def forward(self, x):  # x: [time, batch, features]
        syn1, mem1 = self.slstm1.init_slstm()
        syn2, mem2 = self.slstm2.init_slstm()
        mem2Rec = []
        spk1Rec=[]
        #spk2Rec=[]
        for step in range(x.size(0)):
            spk1, syn1, mem1 = self.slstm1(x[step], syn1, mem1)
            spk1Rec.append(spk1)
        
        #TAB 
        spk1NormRec=roughTemporalAccumulatedBN(spk1Rec,self.bn1)
        
        for step in range(x.size(0)):
            spk2, syn2, mem2 = self.slstm2(spk1NormRec[step], syn2, mem2)
            mem2Rec.append(mem2)
        mem2Rec = torch.stack(mem2Rec)
        finalMem = mem2Rec.mean(dim=0)
        out = self.fc(finalMem)
        
        return out
    
def fileFinder(dataDirectory):
    searchPattern = os.path.join(dataDirectory, '*.mat')
    matFilePaths = glob.glob(searchPattern)  #idk how good glob is but it made this way simpler so... ill revisit it later

    if not matFilePaths:
       raise ValueError("No .mat files found in this directory, please double check and try again :)")
        
    return matFilePaths

num_epochs=5
TESTTHRESHOLD=0.25

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
#model=STCN_Assembled(numChannels,[32, 32, 64, 64],numGestures).to(device)
model=Net_SLSTM().to(device)
 
matFilePaths=fileFinder(r'C:\Users\Nia Touko\Downloads\DB6_s1_a')
testMatFilePaths=fileFinder(r'C:\Users\Nia Touko\Downloads\DB6_s1_b')


validationTestPath=testMatFilePaths.pop(0)

#Isolates training and testing data, and shuffles the training (not the testing to simulate real world )
trainData=loadDataset(matFilePaths)
testData=loadDataset(testMatFilePaths)
validationData=loadDataset([validationTestPath])
'''
data=loadDataset(mat_file_paths)
testSize=int(len(data)*TESTTHRESHOLD)
trainSize=len(data)-testSize
trainData,testData=random_split(data,[trainSize,testSize])
'''
batchSize=128
trainLoader=DataLoader(trainData, batch_size=batchSize, shuffle=True)
testLoader=DataLoader(testData,batch_size=batchSize,shuffle=False)
valLoader=DataLoader(validationData,batch_size=batchSize,shuffle=False)
#Training Settings
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

losses=[]
accuracies=[]
for epoch in range(num_epochs):
    model.train()
    totalLoss = 0
    correctPredictions=0
    loop = tqdm(trainLoader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) #pretty little progress bar
    for x, y in trainLoader:    #loops through trainloader
        #Data is moved to the right place
        x = x.permute(1,0,2)
        x, y = x.to(device), y.to(device)
        #model.reset()   #resets the membrane potential of the LIF neurons (is only needed for the S-TCN architecute)
        output = model(x)
        #calculates the training values
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #updates the readOuts
        totalLoss += loss.item()
        loop.update(1)
        _, predictions = torch.max(output, 1)
        correctPredictions += (predictions == y).sum().item()
        currentAccuracy = (correctPredictions / loop.n)
        loop.set_postfix(loss=loss.item(), acc=f"{currentAccuracy:.2f}%")
    accuracies.append(currentAccuracy)
    losses.append(totalLoss / len(trainLoader))
    print(f"Epoch {epoch+1}: Loss = {losses[-1]:.4f}")
    
    model.eval()  # Set model to evaluation mode
    totalValLoss = 0
    correctPredictions = 0
    totalSamples = 0

    valLoop = tqdm(valLoader, desc=f"Validating Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
    with torch.no_grad():  # Disable gradient calculation for validation
        for x, y in valLoop:
            x, y = x.to(device), y.to(device)
            x = x.permute(1,0,2)
            output = model(x)
            loss = loss_fn(output, y)
            totalValLoss += loss.item()
            
            _, predictions = torch.max(output, 1)
            correctPredictions += (predictions == y).sum().item()
            totalSamples += y.size(0)

    avgValLoss = totalValLoss / len(valLoop)
    valAccuracy = (correctPredictions / totalSamples) * 100
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {losses[-1]:.4f} | Val Loss: {avgValLoss:.4f} | Val Acc: {valAccuracy:.2f}%")
    scheduler.step(avgValLoss)
model.eval()
testLoss=0
correctPredictions=0
testLoop = tqdm(testLoader, desc="Testing", leave=False)
with torch.no_grad():
    for x,y in testLoop:
        x=x.permute(1,0,2)
        x,y=x.to(device),y.to(device)
        #model.reset()
        output=model(x)
        loss = loss_fn(output, y)
        _, predictions = torch.max(output, 1)
        testLoss+=loss.item()
        correctPredictions += (predictions == y).sum().item()
        currentTestAccuracy = (correctPredictions / (testLoop.n+1))
        testLoop.set_postfix(loss=loss.item(), acc=f"{currentTestAccuracy:.2f}%")
        
    print(f"Total loss is {testLoss/len(testLoader):.4f} and your final accuracy is {currentTestAccuracy:.4f}%, compared to your final training loss of {losses[-1]:.4f} and accurcay {currentAccuracy:.4f}% ")
    