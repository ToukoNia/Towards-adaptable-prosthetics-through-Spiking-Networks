# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 12:53:00 2025

@author: Nia Touko
"""


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader,Subset
from snntorch import spikegen
from scipy import io
import numpy as np
import glob
import os
from SCLSTM_DANN_Model import S_CLSTM_DANN as Net
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
# Data parameters
numSubjects = 1  
numGestures = 8
numRepetitions = 12
numChannels = 14
windowSize=400
TESTTHRESHOLD=0.2
inp = int(float(sys.argv[1]))
def loadDataset(mat_paths, windowSize=windowSize, overlapR=0.5, numClasses=numGestures,doEncode=0,domainLabel=0): #def better ways to implement this for multi files but i was going quickly, also might be worth changing the labels to be like bianry strings or smthing later to reduce overfitting?
    x,y,subject=[],[],[]
    stride=int((1-overlapR)*windowSize)
    targetLabels = [0, 1, 3, 4, 6, 9, 10, 11]
    labelMap = {label: i for i, label in enumerate(targetLabels)}
    for mat_path in mat_paths:  #for the single subject case each file is a new session, so the domain label will be just incremented
        mat = io.loadmat(mat_path)
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
                subject.append(domainLabel)
                if doEncode:
                    encodedSegment = spikegen.delta(segment, threshold=1e-5)
                    x.append(encodedSegment)
                else:
                    x.append(segment)
        domainLabel+=1
    x = torch.stack(x)    # [num_samples, window_size, num_channels]
    y = torch.tensor(y)   # [num_samples]
    subjects=torch.tensor(subject)
    #probably should add a fallout for if std is 0 or add an epsilon but ah well     
    return TensorDataset(x, y, subjects)

def loadAndSplitPerSession(matPaths, testSize=TESTTHRESHOLD, doEncode=0,domainLabel=0):  #Stratified Split within each session to ensure each session is well represented etc

    allTrainX, allTrainY = [], []
    allTestX, allTestY = [], []
    allTrainDomain,allTestDomain=[],[]
    overlapR = 0.5
    stride = int((1 - overlapR) * windowSize)
    targetLabels = [0, 1, 3, 4, 6, 9, 10, 11]
    labelMap = {label: i for i, label in enumerate(targetLabels)}
    for matPath in matPaths:
        mat = io.loadmat(matPath)
        emg = mat['emg']
        labels = mat['restimulus']
        emg = np.delete(emg, [8, 9], axis=1)
        emg = torch.tensor(emg, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long).squeeze()
        sessionX, sessionY = [], []
        for i in range(0, emg.shape[0] - windowSize, stride):
            segment = emg[i:i+windowSize]
            labelWindow = labels[i:i+windowSize]
            label = int(torch.mode(labelWindow)[0])

            if label in labelMap:
                remappedLabel = labelMap[label]
                sessionY.append(remappedLabel)
                if doEncode:
                    encodedSegment = spikegen.delta(segment, threshold=1e-5)
                    sessionX.append(encodedSegment)
                else:
                    sessionX.append(segment)
        if not sessionX:
            continue
        sessionX_tensor = torch.stack(sessionX)
        sessionY_tensor = torch.tensor(sessionY)
        numSamples = sessionX_tensor.shape[0]
        sessionSubjects_tensor = torch.full((numSamples,), domainLabel, dtype=torch.long)
        trainX, testX, trainY, testY, trainDomain, testDomain = train_test_split(
            sessionX_tensor,
            sessionY_tensor,
            sessionSubjects_tensor,
            test_size=testSize,
            stratify=sessionY_tensor,
            random_state=42
        )
        allTrainX.append(trainX)
        allTestX.append(testX)
        allTrainY.append(trainY)
        allTestY.append(testY)
        allTrainDomain.append(trainDomain)
        allTestDomain.append(testDomain)
        domainLabel+=1
    finalTrainX = torch.cat(allTrainX, dim=0)
    finalTestX = torch.cat(allTestX, dim=0)
    finalTrainY = torch.cat(allTrainY, dim=0)
    finalTestY = torch.cat(allTestY, dim=0)
    finalTrainDomain = torch.cat(allTrainDomain, dim=0)
    finalTestDoman = torch.cat(allTestDomain, dim=0)
    trainDataset = TensorDataset(finalTrainX, finalTrainY,finalTrainDomain)
    testDataset = TensorDataset(finalTestX, finalTestY,finalTestDoman)

    return trainDataset, testDataset
    
class DataNormaliser():
    def __init__(self):
        self.mean=0
        self.std=0
    def forwardTrain(self,dataset):
        x,y,d=dataset.tensors
        self.mean=x.mean(dim=0,keepdim=True)
        self.std=x.std(dim=0,keepdim=True)
        xNorm=(x-self.mean)/self.std
        return TensorDataset(xNorm,y,d) #probably should add a fallout for if std is 0 or add an epsilon but ah well     
    def forward(self,dataset):
        x,y,d=dataset.tensors
        xNorm=(x-self.mean)/self.std
        return TensorDataset(xNorm,y,d) 

def fileFinder(dataDirectory):
    searchPattern = os.path.join(dataDirectory, '*.mat')
    matFilePaths = glob.glob(searchPattern)  #idk how good glob is but it made this way simpler so... ill revisit it later

    if not matFilePaths:
       raise ValueError("No .mat files found in this directory '%s', please double check and try again :)"%dataDirectory)
        
    return matFilePaths

def trainNetwork(model,trainLoader,testLoader1,testLoader2,numEpochs,lossGestureFn,lossDomainFn,optimiser):
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
        i=0
        for x, y, subjects in trainLoader:    #loops through trainloader
            #Data is moved to the right place
            x = x.permute(1,0,2)
            x, y, subjects = x.to(device), y.to(device),subjects.to(device)
            p = float(i + epoch * len(trainLoader)) / (numEpochs * len(trainLoader))
            i+=1
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # Inside your training loop
            gestureOutput, domainOutput = model(x, alpha)
            
            lossGesture = lossGestureFn(gestureOutput, y)
            lossDomain = lossDomainFn(domainOutput, subjects)
            
            # Combine the losses
            optimiser.zero_grad()
            loss = lossGesture + lossDomain 
            loss.backward()
            optimiser.step()
            #updates the readOuts
            totalLoss += lossGesture.item()
            _, predictions = torch.max(gestureOutput, 1)
            correctPredictions += (predictions == y).sum().item()
        currentAccuracy = (correctPredictions / len(trainLoader.dataset))*100
        history['train_loss'].append(totalLoss / len(trainLoader))
        history['train_acc'].append(currentAccuracy)

        
        testAccuracyIntra,testLossIntra=testNetwork(model,testLoader1,lossGestureFn)
        testAccuracyInter,testLossInter=testNetwork(model,testLoader2,lossGestureFn)
        
        history['intra_session_acc'].append(testAccuracyIntra)
        history['intra_session_loss'].append(testLossIntra)
        history['inter_session_acc'].append(testAccuracyInter)
        history['inter_session_loss'].append(testLossInter)
        print(f"Epoch Completed with train loss {history['train_loss'][-1]},inner loss {testLossInter} and intra loss {testLossIntra}")
    return history

def testNetwork(model, testLoader,loss_fn):
    model.eval()
    with torch.no_grad():
        correctPredictions=0
        testLoss=0
        for x,y,_ in testLoader:
            x=x.permute(1,0,2)
            x,y=x.to(device),y.to(device)
            gestureOutput,domainOutput=model(x)
            loss = loss_fn(gestureOutput, y)
            _, predictions = torch.max(gestureOutput, 1)
            testLoss+=loss.item()
            correctPredictions += (predictions == y).sum().item()
        testAccuracy=100*correctPredictions/len(testLoader.dataset)
        testLoss=testLoss / len(testLoader)
        return testAccuracy,testLoss
        
def plot_results(history, subject_id):
    #Plots training and validation metrics and saves the figure.
    df = pd.DataFrame(history)
    
    plt.style.use('seaborn-whitegrid')    
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


def SubjectChecker(model,i,encode=0):
    normaliseData=DataNormaliser()
    matFilePaths=fileFinder(r'/home/coa23nt/EMG-SNN/Data/DB6_s%s_a'%i)+fileFinder(r'/home/coa23nt/EMG-SNN/Data/DB6_s%s_b'%i)
    dataPaths=matFilePaths[:7]
    targetDataPath=matFilePaths[7:]
    trainData,testDataInter=loadAndSplitPerSession(dataPaths,doEncode=encode)
    testDataIntra=loadDataset(targetDataPath,doEncode=encode)
    lossGestureFn=nn.CrossEntropyLoss()
    lossDomainFn=nn.CrossEntropyLoss()
    
    if not encode:
        trainData=normaliseData.forwardTrain(trainData)
        testDataIntra=normaliseData.forward(testDataIntra)
        testDataInter=normaliseData.forward(testDataInter)
     
    
    trainLoader=DataLoader(trainData, batch_size=batchSize, shuffle=True)
    testLoaderIntra=DataLoader(testDataIntra,batch_size=batchSize,shuffle=False)
    testLoaderInter=DataLoader(testDataInter,batch_size=batchSize,shuffle=False)
    
    #Load and run the network
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    history=trainNetwork(model,trainLoader,testLoaderIntra,testLoaderInter,numEpochs,lossGestureFn,lossDomainFn,optimiser)
    
    
    results_df = pd.DataFrame(history)
    results_df.to_csv(f"subject_{i}_training_history.csv", index_label="Epoch")
    print(f"\nResults for subject {i} saved to subject_{i}_training_history.csv")
    plot_results(history,i)
    torch.save({'model_state_dict': model.state_dict(),'optimiser_state_dict':optimiser.state_dict()},r"Subject_%s_SLSTM_TAB" % i)    
            
    
numEpochs=4
batchSize=128

    

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model=Net().to(device)

lossFn = nn.CrossEntropyLoss()
SubjectChecker(model,inp,encode=0)