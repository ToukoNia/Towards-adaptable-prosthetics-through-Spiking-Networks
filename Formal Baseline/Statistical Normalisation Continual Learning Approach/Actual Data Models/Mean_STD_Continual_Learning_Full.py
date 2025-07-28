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
from scipy import io
import numpy as np
import glob
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import snntorch as snn
from snntorch import surrogate
import copy
# Data parameters
numSubjects = 1  
numGestures = 8
numRepetitions = 12
numChannels = 14
windowSize=400
TESTTHRESHOLD=0.3
inp = int(float(sys.argv[1]))

class Net_SLSTM_Extractor(nn.Module):
    def __init__(self, inputSize=2, hiddenSize=128, numClasses=8):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.numClasses = numClasses
        self.slstm1 = snn.SLSTM(inputSize, hiddenSize, spike_grad=surrogate.fast_sigmoid(),learn_threshold=True,reset_mechanism="subtract")
        self.slstm2 = snn.SLSTM(hiddenSize, hiddenSize, spike_grad=surrogate.fast_sigmoid(),learn_threshold=True,reset_mechanism="subtract")
        #self.bn1 = nn.BatchNorm1d(hiddenSize)

    def forward(self, x):  # x: [time, batch, features]
        syn1, mem1 = self.slstm1.init_slstm()
        syn2, mem2 = self.slstm2.init_slstm()
        mem2Rec = []
        mem1Rec = []
        for step in range(x.size(0)):
            spk1, syn1, mem1 = self.slstm1(x[step], syn1, mem1)
            #spk1 = self.bn1(spk1)
            spk2, syn2, mem2 = self.slstm2(spk1, syn2, mem2)
            
            mem1Rec.append(mem1)
            mem2Rec.append(mem2)
        mem1Rec = torch.stack(mem1Rec)
        mem2Rec = torch.stack(mem2Rec)
        finalMem = mem2Rec.mean(dim=0)
        
        return finalMem, mem1Rec, mem2Rec
   
class Classifier(nn.Module):
    def __init__(self, hiddenSize=128, numClasses=8):
        super().__init__()
        self.fc = nn.Linear(hiddenSize, numClasses)

    def forward(self, z):
        return self.fc(z)
    
class DERBuffer:
    def __init__(self, device):
        self.x = None
        self.y = None
        self.y_comp = None
        self.mem1 = None
        self.mem2 = None
        self.device = device

    def update(self, x, y, y_comp, mem1, mem2):
        self.x = x.clone().detach().to(self.device)
        self.y = y.clone().detach().to(self.device)
        self.y_comp = y_comp.clone().detach().to(self.device)
        self.mem1 = mem1.clone().detach().to(self.device)
        self.mem2 = mem2.clone().detach().to(self.device)
        
def loadDataset(mat_paths, windowSize=windowSize, overlapR=0.5, numClasses=numGestures,doEncode=0):
    x,y=[],[]
    stride=int((1-overlapR)*windowSize)
    targetLabels = [0, 1, 3, 4, 6, 9, 10, 11]
    labelMap = {label: i for i, label in enumerate(targetLabels)}
    for mat_path in mat_paths:
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
                if doEncode:
                    encodedSegment=spikegen.delta(segment,threshold=1e-5)
                    x.append(encodedSegment)
                else:
                    x.append(segment)
    
    x = torch.stack(x)    # [num_samples, window_size, num_channels]
    y = torch.tensor(y)   # [num_samples]

    
    return TensorDataset(x, y)

def loadAndSplitPerSession(matPaths, testSize=TESTTHRESHOLD, doEncode=0):

    allTrainX, allTrainY = [], []
    allTestX, allTestY = [], []

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
        trainX, testX, trainY, testY = train_test_split(
            sessionX_tensor,
            sessionY_tensor,
            test_size=testSize,
            stratify=sessionY_tensor, 
            random_state=42
        )
        allTrainX.append(trainX)
        allTestX.append(testX)
        allTrainY.append(trainY)
        allTestY.append(testY)
    finalTrainX = torch.cat(allTrainX, dim=0)
    finalTestX = torch.cat(allTestX, dim=0)
    finalTrainY = torch.cat(allTrainY, dim=0)
    finalTestY = torch.cat(allTestY, dim=0)

    trainDataset = TensorDataset(finalTrainX, finalTrainY)
    testDataset = TensorDataset(finalTestX, finalTestY)

    return trainDataset, testDataset

class DataNormaliser():
    def __init__(self):
        self.mean=0
        self.std=0
    def forwardTrain(self,dataset):
        x,y=dataset.tensors
        self.mean=x.mean(dim=0,keepdim=True)
        self.std=x.std(dim=0,keepdim=True)
        xNorm=(x-self.mean)/self.std
        return TensorDataset(xNorm,y) #probably should add a fallout for if std is 0 or add an epsilon but ah well     
    def forward(self,dataset):
        x,y=dataset.tensors
        xNorm=(x-self.mean)/self.std
        return TensorDataset(xNorm,y) 
    
    
def fileFinder(dataDirectory):
    searchPattern = os.path.join(dataDirectory, '*.mat')
    matFilePaths = glob.glob(searchPattern)  #idk how good glob is but it made this way simpler so... ill revisit it later

    if not matFilePaths:
       raise ValueError("No .mat files found in this directory '%s', please double check and try again :)"%dataDirectory)
        
    return matFilePaths

def trainNetwork(encoder,classifier,trainLoader,testLoader1,testLoader2,numEpochs,loss_fn,optimiser,adaptOpt):
    #Training Settings
    history = {
       'train_loss': [], 'train_acc': [],
       'intra_pre_acc':[], 'intra_pre_loss':[],'intra_acc_best': [], 'intra_acc_worst': [], 'intra_acc_mean': [], 'intra_acc_final': [],
       'intra_loss_best': [], 'intra_loss_worst': [], 'intra_loss_mean': [], 'intra_loss_final': [],
       'inter_pre_acc':[], 'inter_pre_loss':[],'inter_acc_best': [], 'inter_acc_worst': [], 'inter_acc_mean': [], 'inter_acc_final': [],
        'inter_loss_best': [], 'inter_loss_worst': [], 'inter_loss_mean': [], 'inter_loss_final': [],
   }
    der_buffer=DERBuffer(device)
    for epoch in range(numEpochs):
        encoder.train()
        classifier.train()
        totalLoss = 0
        correctPredictions=0
        for x, y in trainLoader:    #loops through trainloader
            #Data is moved to the right place
            x = x.permute(1,0,2)
            x,y=x.to(device),y.to(device)
            z, mem1, mem2 = encoder(x)
            output = classifier(z)
            #calculates the training values
            loss = loss_fn(output, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            der_buffer.update(x, y, output, mem1, mem2)
            #updates the readOuts
            totalLoss += loss.item()
            _, predictions = torch.max(output, 1)
            correctPredictions += (predictions == y).sum().item()
        currentAccuracy = (correctPredictions / len(trainLoader.dataset))*100
        history['train_loss'].append(totalLoss / len(trainLoader))
        history['train_acc'].append(currentAccuracy)

        
        intraResults=TTATester(encoder, classifier, der_buffer,testLoader1, adaptOpt)
        InterResults=TTATester(encoder, classifier, der_buffer,testLoader2, adaptOpt)
        
        for key, value in intraResults.items(): history[f'intra_{key}'].append(value)
        for key, value in InterResults.items(): history[f'inter_{key}'].append(value)
        
        print(f"Epoch {epoch+1}/{numEpochs} | Train Acc: {history['train_acc'][-1]:.2f}% | "
             f"Intra-TTA Final Acc: {history['intra_acc_final'][-1]:.2f}% (Best: {history['intra_acc_best'][-1]:.2f}%) | "
             f"Inter-TTA Final Acc: {history['inter_acc_final'][-1]:.2f}% (Best: {history['inter_acc_best'][-1]:.2f}%)")
    return history

def TTATester(encoder, classifier, der_buffer,dataLoader, adaptOpt):  
    encoder_state_before = copy.deepcopy(encoder.state_dict())
    classifier_state_before = copy.deepcopy(classifier.state_dict())
    
    results = TTA(encoder,classifier,der_buffer,dataLoader,adaptOpt)
    
    encoder.load_state_dict(encoder_state_before)
    classifier.load_state_dict(classifier_state_before)
    
    return results

def testNetwork(encoder,classifier, testLoader,loss_fn):
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        correctPredictions=0
        testLoss=0
        for x,y in testLoader:
            x=x.permute(1,0,2)
            x,y=x.to(device),y.to(device)
            z,_,_ = encoder(x)
            output = classifier(z)
            loss = loss_fn(output, y)
            _, predictions = torch.max(output, 1)
            testLoss+=loss.item()
            correctPredictions += (predictions == y).sum().item()
        testAccuracy=100*correctPredictions/len(testLoader.dataset)
        testLoss=testLoss / len(testLoader)
    return testAccuracy,testLoss

def TTA(encoder,classifier,der_buffer,dataLoader,adaptOpt,alpha=1,beta=1.1,nAdaption=5):    #Need to make this not loop through the whole dataset, maybe make it take one example of each gesture type and use that?
    ttaAcc,ttaLoss=[],[]
    lossFn=nn.CrossEntropyLoss()
    preAcc, preLoss = testNetwork(encoder, classifier, dataLoader, lossFn)
    
    encoder.train()
    classifier.train()
    for _ in range(nAdaption):
        for x_batch,_ in dataLoader:
            x_batch = x_batch.permute(1, 0, 2).to(device)
            z, mem1_te, mem2_te=encoder(x_batch)
        
            meanLoss=torch.pow(mem1_te.mean(0)-der_buffer.mem1.mean(0),2).mean()+torch.pow(mem2_te.mean(0)-der_buffer.mem2.mean(0),2).mean()
            std_loss=torch.pow(mem1_te.std(0)-der_buffer.mem1.std(0),2).mean()+torch.pow(mem2_te.std(0)-der_buffer.mem2.std(0),2).mean()
            statLoss = meanLoss + std_loss
            
            zDER, _,_=encoder(der_buffer.x)
            yDER=classifier(zDER)
            derLoss=torch.pow(yDER-der_buffer.yComp,2).mean()+nn.CrossEntropyLoss()(yDER,der_buffer.y)
            loss=alpha*statLoss+beta*derLoss
            adaptOpt.zero_grad()
            loss.backward()
            adaptOpt.step()
        acc, loss = testNetwork(encoder, classifier, dataLoader, lossFn)
        ttaAcc.append(acc)
        ttaLoss.append(loss)
    results = {
        'pre_acc':preAcc, 'pre_loss': preLoss,
        'acc_best': max(ttaAcc), 'loss_best': min(ttaLoss),
        'acc_worst': min(ttaAcc), 'loss_worst': max(ttaLoss),
        'acc_mean': np.mean(ttaAcc), 'loss_mean': np.mean(ttaLoss),
        'acc_final': ttaAcc[-1], 'loss_final': ttaLoss[-1],
    }
    return results
            
    
def plot_results(history, subject_id):
    df = pd.DataFrame(history)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 16), sharex=True)
    fig.suptitle(f'Subject {subject_id} - Detailed Training and TTA Performance', fontsize=18)

    # --- ACCURACY PLOT ---
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.plot(df['train_acc'], label='Train Accuracy', color='black', linewidth=2, linestyle='-')
    # Intra-Session Accuracy (Green/Orange/Red Tones)
    ax1.plot(df['intra_acc_pre'], label='Intra: Pre-TTA', color='#ff6347', linestyle=':') # Tomato
    ax1.plot(df['intra_acc_mean'], label='Intra: TTA Mean', color='#ffa500', linestyle='--') # Orange
    ax1.plot(df['intra_acc_best'], label='Intra: TTA Best', color='#228b22', linewidth=1.5) # ForestGreen
    ax1.fill_between(df.index, df['intra_acc_worst'], df['intra_acc_best'], color='green', alpha=0.1, label='Intra: TTA Range (Worst-Best)')
    # Inter-Session Accuracy (Blue/Purple Tones)
    ax1.plot(df['inter_acc_pre'], label='Inter: Pre-TTA', color='#87ceeb', linestyle=':') # SkyBlue
    ax1.plot(df['inter_acc_mean'], label='Inter: TTA Mean', color='#9370db', linestyle='--') # MediumPurple
    ax1.plot(df['inter_acc_best'], label='Inter: TTA Best', color='#0000cd', linewidth=1.5) # MediumBlue
    ax1.fill_between(df.index, df['inter_acc_worst'], df['inter_acc_best'], color='blue', alpha=0.1, label='Inter: TTA Range (Worst-Best)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend(loc='best', fontsize='small')
    ax1.grid(True)

    # --- LOSS PLOT ---
    ax2.set_title('Model Loss', fontsize=14)
    ax2.plot(df['train_loss'], label='Train Loss', color='black', linewidth=2, linestyle='-')
    # Intra-Session Loss
    ax2.plot(df['intra_loss_pre'], label='Intra: Pre-TTA', color='#ff6347', linestyle=':')
    ax2.plot(df['intra_loss_mean'], label='Intra: TTA Mean', color='#ffa500', linestyle='--')
    ax2.plot(df['intra_loss_best'], label='Intra: TTA Best (Min Loss)', color='#228b22', linewidth=1.5)
    ax2.fill_between(df.index, df['intra_loss_best'], df['intra_loss_worst'], color='green', alpha=0.1, label='Intra: TTA Range (Best-Worst)')
    # Inter-Session Loss
    ax2.plot(df['inter_loss_pre'], label='Inter: Pre-TTA', color='#87ceeb', linestyle=':')
    ax2.plot(df['inter_loss_mean'], label='Inter: TTA Mean', color='#9370db', linestyle='--')
    ax2.plot(df['inter_loss_best'], label='Inter: TTA Best (Min Loss)', color='#0000cd', linewidth=1.5)
    ax2.fill_between(df.index, df['inter_loss_best'], df['inter_loss_worst'], color='blue', alpha=0.1, label='Inter: TTA Range (Best-Worst)')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epochs')
    ax2.legend(loc='best', fontsize='small')
    ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"subject_{subject_id}_TTA_detailed_performance.png")
    plt.close(fig)
    
def createStratifiedSplit(fullDataset, testSize):
    x = fullDataset.tensors[0]
    y = fullDataset.tensors[1]
    labels = fullDataset.tensors[1].numpy()
    indices = list(range(len(fullDataset)))
    trainIndices, testIndices = train_test_split(
        indices,
        test_size=testSize,
        stratify=labels,
        random_state=42  # Ensures the split is the same every time
    )

    trainX = x[trainIndices]
    trainY = y[trainIndices]

    testX = x[testIndices]
    testY = y[testIndices]

    trainSet = TensorDataset(trainX, trainY)
    testSet = TensorDataset(testX, testY)

    return trainSet, testSet

def SubjectChecker(loss_fn,i,encode=0):

    matFilePaths=fileFinder(r'/home/coa23nt/EMG-SNN/Data/DB6_s%s_a'%i)+fileFinder(r'/home/coa23nt/EMG-SNN/Data/DB6_s%s_b'%i)
    dataPaths=matFilePaths[:7]
    targetDataPath=matFilePaths[7:]
    testDataIntra=loadDataset(targetDataPath,doEncode=encode)
    trainData,testDataInter= loadAndSplitPerSession(dataPaths) 
    
    normaliseData=DataNormaliser()

    
    if not encode:
        trainData=normaliseData.forwardTrain(trainData)
        testDataInter=normaliseData.forward(testDataInter)
        testDataIntra=normaliseData.forward(testDataIntra)
     
    
    trainLoader=DataLoader(trainData, batch_size=batchSize, shuffle=True)
    testLoaderIntra=DataLoader(testDataIntra,batch_size=batchSize,shuffle=False)
    testLoaderInter=DataLoader(testDataInter,batch_size=batchSize,shuffle=False)
    
    #Load and run the network
    snn_LSTM=Net_SLSTM_Extractor(inputSize=numChannels, hiddenSize=128)
    readout=Classifier(hiddenSize=128, numClasses=numGestures)
    optimiser = torch.optim.Adam(params=list(snn_LSTM.parameters())+list(readout.parameters()), lr=1e-3)
    adaptOpt=torch.optim.Adam(params=list(snn_LSTM.parameters())+list(readout.parameters()),lr=1e-2)
    history=trainNetwork(snn_LSTM,readout,trainLoader,testLoaderIntra,testLoaderInter,numEpochs,loss_fn,optimiser,adaptOpt)
    
    
    results_df = pd.DataFrame(history)
    results_df.to_csv(f"subject_{i}_training_history.csv", index_label="Epoch")
    print(f"\nResults for subject {i} saved to subject_{i}_training_history.csv")
    plot_results(history,i)
    model_state = {'encoder_state_dict': snn_LSTM.state_dict(), 'classifier_state_dict': readout.state_dict()}
    torch.save(model_state,r"Subject_%s_SLSTM_TAB" % i)    
            
    
numEpochs=15
batchSize=128

    

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

lossFn = nn.CrossEntropyLoss()
SubjectChecker(lossFn,inp,encode=1)

