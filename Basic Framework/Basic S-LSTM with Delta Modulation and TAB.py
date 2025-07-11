# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 20:23:22 2025

@author: Nia Touko
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader      #,random_split
from snntorch import spikegen
import scipy
import numpy as np
from tqdm import tqdm
import glob
import os
from S_LSTM_with_TAB_Model import Net_SLSTM
# Data parameters
numSubjects = 1  
numGestures = 8
numRepetitions = 12
numChannels = 14
windowSize=400

def loadDataset(mat_paths, windowSize=windowSize, overlapR=0.5, numClasses=numGestures,doEncode=0): #def better ways to implement this for multi files but i was going quickly, also might be worth changing the labels to be like bianry strings or smthing later to reduce overfitting?
    x,y=[],[]
    stride=int((1-overlapR)*windowSize)
    targetLabels = [0, 1, 3, 4, 6, 9, 10, 11]
    labelMap = {label: i for i, label in enumerate(targetLabels)}
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
            label = int((torch.mode(label_window)[0]))
            if label in labelMap: #remaps the labels so that they are within range (is this even neccessary)
                remappedLabel = labelMap[label]
                y.append(remappedLabel) # Append the new, remapped label (e.g., 2 instead of 3)
                if doEncode:
                    encodedSegment=spikegen.delta(segment,threshold=1e-5)
                    x.append(encodedSegment)
                else:
                    x.append(segment)
    
    x = torch.stack(x)    # [num_samples, window_size, num_channels]
    y = torch.tensor(y)   # [num_samples]

    #probably should add a fallout for if std is 0 or add an epsilon but ah well     
    return TensorDataset(x, y)

def fileFinder(dataDirectory):
    searchPattern = os.path.join(dataDirectory, '*.mat')
    matFilePaths = glob.glob(searchPattern)  #idk how good glob is but it made this way simpler so... ill revisit it later

    if not matFilePaths:
       raise ValueError("No .mat files found in this directory, please double check and try again :)")
        
    return matFilePaths

def trainNetwork(model,trainLoader,numEpochs,loss_fn,optimiser,doReset):
    #Training Settings

    losses=[]
    accuracies=[]
    for epoch in range(numEpochs):
        model.train()
        totalLoss = 0
        correctPredictions=0
        loop = tqdm(trainLoader, desc=f"Epoch {epoch+1}/{numEpochs}", leave=False) #pretty little progress bar
        for x, y in trainLoader:    #loops through trainloader
            #Data is moved to the right place
            x = x.permute(1,0,2)
            x, y = x.to(device), y.to(device)
            if doReset:
                model.reset()   #resets the membrane potential of the LIF neurons (is only needed for the S-TCN architecute)
            output = model(x)
            #calculates the training values
            loss = loss_fn(output, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            #updates the readOuts
            totalLoss += loss.item()
            loop.update(1)
            _, predictions = torch.max(output, 1)
            correctPredictions += (predictions == y).sum().item()
            currentAccuracy = (correctPredictions / loop.n)
            loop.set_postfix(loss=loss.item(), acc=f"{currentAccuracy:.2f}%")
        accuracies.append(correctPredictions/len(trainLoader.dataset))
        losses.append(totalLoss / len(trainLoader))
        print(f"Epoch {epoch+1}: Loss = {losses[-1]:.4f}")
    return losses, accuracies

def testNetwork(model, testLoader,loss_fn,optimiser,doReset):
    model.eval()
    testLoss=0
    totalLoss=0
    correctPredictions=0
    with torch.no_grad():
        loop = tqdm(trainLoader, desc="Testing on seen", leave=False) #pretty little progress bar
        for x,y in trainLoader:
            x=x.permute(1,0,2)
            x,y=x.to(device),y.to(device)
            if doReset:
                model.reset()   #resets the membrane potential of the LIF neurons (is only needed for the S-TCN architecute)
            output=model(x)
            loss = loss_fn(output, y)
            _, predictions = torch.max(output, 1)
            totalLoss += loss.item()
            loop.update(1)
            correctPredictions += (predictions == y).sum().item()
            currentAccuracy = (correctPredictions / loop.n)
            loop.set_postfix(loss=loss.item(), acc=f"{currentAccuracy:.2f}%")
        testLoop = tqdm(testLoader, desc="Testing on unseen  ", leave=False)
        correctPredictions=0
        i=0
        for x,y in testLoop:
            x=x.permute(1,0,2)
            x,y=x.to(device),y.to(device)
            if doReset:
                model.reset()   #resets the membrane potential of the LIF neurons (is only needed for the S-TCN architecute)
            output=model(x)
            loss = loss_fn(output, y)
            _, predictions = torch.max(output, 1)
            testLoss+=loss.item()
            i+=1
            correctPredictions += (predictions == y).sum().item()
            currentTestAccuracy = (correctPredictions /i)
            testLoop.set_postfix(loss=loss.item(), acc=f"{currentTestAccuracy:.2f}%")
            
        print(f"Total loss is {testLoss/len(testLoader):.4f} and your final accuracy is {correctPredictions/len(testLoader):.4f}%, compared to your final training loss of {loss.item():.4f} and accurcay {currentAccuracy:.4f}% ")
        
def LOSO(model,loss_fn):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    matFilePaths=[]
    for i in range(1,11):
        matFilePaths+=fileFinder(r'..\Data\DB6_s%s_a' % i)+fileFinder(r'..\Data\DB6_s%s_b' % i)
    for i in range(0,10):
        dataPaths=matFilePaths
        targetDataPath=dataPaths.pop(i)
        trainData=loadDataset(dataPaths)
        testData=loadDataset(targetDataPath)
        trainLoader=DataLoader(trainData, batch_size=batchSize, shuffle=True)
        testLoader=DataLoader(testData,batch_size=batchSize,shuffle=False)
        #Load and run the network
        model=Net_SLSTM().to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses,accuracies=trainNetwork(model,trainLoader,numEpochs,loss_fn,optimiser,0)
        testNetwork(model,testLoader,loss_fn,optimiser,0)
            
    
numEpochs=10
batchSize=128

    

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
#model=STCN_Assembled(numChannels,[32, 32, 64, 64],numGestures).to(device)
model=Net_SLSTM().to(device)

#matFilePaths=fileFinder(r'..\Data\DB6_s1_a')+fileFinder(r'..\Data\DB6_s1_b')+fileFinder(r'..\Data\DB6_s7_a')+fileFinder(r'..\Data\DB6_s7_b')
#testMatFilePaths=fileFinder(r'..\Data\DB6_s2_a')+fileFinder(r'..\Data\DB6_s2_b')[0:2]

matFilePaths=fileFinder(r'C:\Users\Nia Touko\Downloads\DB6_s1_a')
testMatFilePaths=fileFinder(r'C:\Users\Nia Touko\Downloads\DB6_s1_b')
#Isolates training and testing data, and shuffles the training (not the testing to simulate real world )
matFilePaths.append(testMatFilePaths.pop(0))

trainData=loadDataset(matFilePaths,doEncode=1)
testData=loadDataset(testMatFilePaths,doEncode=1)


trainLoader=DataLoader(trainData, batch_size=batchSize, shuffle=True)
testLoader=DataLoader(testData,batch_size=batchSize,shuffle=False)

#Training Settings
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)


losses,accuracies=trainNetwork(model,trainLoader,numEpochs,loss_fn,optimiser,0)
testNetwork(model,testLoader,loss_fn,optimiser,0)
