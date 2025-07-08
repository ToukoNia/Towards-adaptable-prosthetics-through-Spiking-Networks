# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 18:15:53 2025

@author: Nia Touko
"""

#Modified standard loader but for the supercomputer, single subject testing


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader,random_split
from snntorch import spikegen
import scipy
import numpy as np
import glob
import os
from S_LSTM_Model import Net_SLSTM as Net
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Data parameters
numSubjects = 1  
numGestures = 8
numRepetitions = 12
numChannels = 14
windowSize=400
TESTTHRESHOLD=0.3
#inp = int(float(sys.argv[1])-1)
inp=1
def loadDataset(mat_paths, windowSize=windowSize, overlapR=0.5, numClasses=numGestures,doEncode=0): #def better ways to implement this for multi files but i was going quickly, also might be worth changing the labels to be like bianry strings or smthing later to reduce overfitting?
    x,y=[],[]
    stride=int((1-overlapR)*windowSize)
    targetLabels = [0, 1, 3, 4, 6, 9, 10, 11]
    labelMap = {label: i for i, label in enumerate(targetLabels)}
    for mat_path in mat_paths:
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

def trainNetwork(model,trainLoader,testLoader1,testLoader2,numEpochs,loss_fn,optimiser,doReset):
    #Training Settings
    history = {
       'train_loss': [], 'train_acc': [],
       'intra_session_loss': [], 'intra_session_acc': [],
       'inter_session_loss': [], 'inter_session_acc': []
   }

    for epoch in range(numEpochs):
        model.train()
        totalLoss = 0
        correctPredictions=0
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
            _, predictions = torch.max(output, 1)
            correctPredictions += (predictions == y).sum().item()
        currentAccuracy = (correctPredictions / len(trainLoader.dataset))*100
        history['train_loss'].append(totalLoss / len(trainLoader))
        history['train_acc'].append(currentAccuracy)

        
        testAccuracyIntra,testLossIntra=testNetwork(model,testLoader1,loss_fn,0)
        testAccuracyInter,testLossInter =testNetwork(model,testLoader2,loss_fn,0)
        
        history['intra_session_acc'].append(testAccuracyIntra)
        history['intra_session_loss'].append(testLossIntra)
        history['inter_session_acc'].append(testAccuracyInter)
        history['inter_session_loss'].append(testLossInter)
        
    return history

def testNetwork(model, testLoader,loss_fn,doReset):
    model.eval()
    with torch.no_grad():
        correctPredictions=0
        testLoss=0
        for x,y in testLoader:
            x=x.permute(1,0,2)
            x,y=x.to(device),y.to(device)
            if doReset:
                model.reset()   #resets the membrane potential of the LIF neurons (is only needed for the S-TCN architecute)
            output=model(x)
            loss = loss_fn(output, y)
            _, predictions = torch.max(output, 1)
            testLoss+=loss.item()
            correctPredictions += (predictions == y).sum().item()
        testAccuracy=100*correctPredictions/len(testLoader.dataset)
        testLoss=testLoss / len(testLoader)
        return testAccuracy,testLoss
        
def plot_results(history, subject_id):
    """Plots training and validation metrics and saves the figure."""
    df = pd.DataFrame(history)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plotting Loss
    ax1.plot(df['train_loss'], label='Train Loss', color='blue')
    ax1.plot(df['intra_session_loss'], label='Intra-Session Test Loss', linestyle='--', color='green')
    ax1.plot(df['inter_session_loss'], label='Inter-Session Test Loss', linestyle='--', color='red')
    ax1.set_title(f'Subject {subject_id} - Model Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plotting Accuracy
    ax2.plot(df['train_acc'], label='Train Accuracy', color='blue')
    ax2.plot(df['intra_session_acc'], label='Intra-Session Test Accuracy', linestyle='--', color='green')
    ax2.plot(df['inter_session_acc'], label='Inter-Session Test Accuracy', linestyle='--', color='red')
    ax2.set_title(f'Subject {subject_id} - Model Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    # Save the figure
    plt.savefig(f"subject_{subject_id}_training_results.png")
    plt.show()


def SubjectChecker(model,loss_fn,i):
    matFilePaths=[]
    matFilePaths=fileFinder(r'..\Data\DB6_s%s_a' % i)+fileFinder(r'..\Data\DB6_s%s_b' % i)
    dataPaths=matFilePaths[:7]
    targetDataPath=matFilePaths[7:]
    trainData=loadDataset(dataPaths)
    testDataIntra=loadDataset(targetDataPath)
    
    testSize=int(len(trainData)*TESTTHRESHOLD)
    trainSize=len(trainData)-testSize
    trainData,testDataInter=random_split(trainData,[trainSize,testSize])        
    
    trainLoader=DataLoader(trainData, batch_size=batchSize, shuffle=True)
    testLoaderIntra=DataLoader(testDataIntra,batch_size=batchSize,shuffle=False)
    testLoaderInter=DataLoader(testDataInter,batch_size=batchSize,shuffle=False)
    
    #Load and run the network
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    history=trainNetwork(model,trainLoader,testLoaderIntra,testLoaderInter,numEpochs,loss_fn,optimiser,0)
    
    results_df = pd.DataFrame(history)
    results_df.to_csv(f"subject_{i}_training_history.csv", index_label="Epoch")
    print(f"\nResults for subject {i} saved to subject_{i}_training_history.csv")
    
        
            
    
numEpochs=10
batchSize=128

    

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
#model=STCN_Assembled(numChannels,[32, 32, 64, 64],numGestures).to(device)
model=Net().to(device)

lossFn = nn.CrossEntropyLoss()
SubjectChecker(model,lossFn,inp)

