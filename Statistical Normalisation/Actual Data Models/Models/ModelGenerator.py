# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 21:49:19 2025

@author: touko
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from snntorch import spikegen
from scipy import io
import numpy as np
import glob
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import snntorch as snn
from snntorch import surrogate
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Data parameters
numGestures = 8
numChannels = 14
windowSize = 400
inp = int(float(sys.argv[1]))
batchSize = 128
numEpochs = 50

class Net_SLSTM_Extractor(nn.Module):
    def __init__(self, inputSize=14, hiddenSize=128, numClasses=8):
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

class DataNormalizer():
    def __init__(self):
        self.mean = 0
        self.std = 0
    def forwardTrain(self, dataset):
        x, y = dataset.tensors
        self.mean = x.mean(dim=(0, 1), keepdim=True)
        self.std = x.std(dim=(0, 1), keepdim=True)
        xNorm = (x - self.mean) / self.std
        return TensorDataset(xNorm, y)
    def forward(self, dataset):
        x, y = dataset.tensors
        xNorm = (x - self.mean) / self.std
        return TensorDataset(xNorm, y)


def loadDataset(matPaths, windowSize=windowSize, overlapR=0.5, doEncode=0):
    x, y = [], []
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
        for i in range(0, emg.shape[0] - windowSize, stride):
            segment = emg[i:i + windowSize]
            labelWindow = labels[i:i + windowSize]
            label = int((torch.mode(labelWindow)[0]))
            if label in labelMap:
                remappedLabel = labelMap[label]
                y.append(remappedLabel)
                if doEncode:
                    encodedSegment = spikegen.delta(segment, threshold=1e-5)
                    x.append(encodedSegment)
                else:
                    x.append(segment)
    x = torch.stack(x)
    y = torch.tensor(y)
    return TensorDataset(x, y)


def fileFinder(dataDirectory):
    searchPattern = os.path.join(dataDirectory, '*.mat')
    matFilePaths = glob.glob(searchPattern)
    if not matFilePaths:
        raise ValueError("No .mat files found in this directory '%s'" % dataDirectory)
    return matFilePaths

def testNetwork(encoder, classifier, testLoader, lossFn, device):
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        correctPredictions = 0
        testLoss = 0
        for x, y in testLoader:
            x = x.permute(1, 0, 2).to(device)
            y = y.to(device)
            z,_,_ = encoder(x)
            output = classifier(z)
            loss = lossFn(output, y)
            _, predictions = torch.max(output, 1)
            testLoss += loss.item()
            correctPredictions += (predictions == y).sum().item()
        testAccuracy = 100 * correctPredictions / len(testLoader.dataset)
        testLoss = testLoss / len(testLoader)
    return testAccuracy, testLoss


def plotResults(history, subjectId):
    plt.style.use('seaborn-whitegrid')
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f'Subject {subjectId} - Training and Testing Performance', fontsize=16)

    for i in range(3):
        ax1 = axes[i]
        ax2 = ax1.twinx()
        
        ax1.set_title(f'Test Session {i+1}')
        ax1.plot(df['train_acc'], 'o-', label='Train Accuracy', color='C0', alpha=0.7)
        ax1.plot(df[f'test_acc_{i+1}'], 's-', label=f'Test Accuracy {i+1}', color='C1')
        ax1.set_ylabel('Accuracy (%)', color='C0')
        ax1.tick_params(axis='y', labelcolor='C0')
        ax1.set_ylim(0, 101)

        ax2.plot(df['train_loss'], 'o--', label='Train Loss', color='C2', alpha=0.7)
        ax2.plot(df[f'test_loss_{i+1}'], 's--', label=f'Test Loss {i+1}', color='C3')
        ax2.set_ylabel('Loss', color='C2')
        ax2.tick_params(axis='y', labelcolor='C2')
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='center right')

    axes[2].set_xlabel('Epochs')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f"subject_{subjectId}_training_results.png")
    plt.close(fig)

def trainNetwork(encoder, classifier, trainLoader, testLoaders, numEpochs, lossFn, optimizer, device,scheduler):
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss_1': [], 'test_acc_1': [],
        'test_loss_2': [], 'test_acc_2': [],
        'test_loss_3': [], 'test_acc_3': []
    }
    
    bestAcc = 0.0
    bestEncState = None
    bestClassState = None

    for epoch in range(numEpochs):
        encoder.train()
        classifier.train()
        totalLoss = 0
        correctPredictions = 0
        for x, y in trainLoader:
            x = x.permute(1, 0, 2).to(device)
            y = y.to(device)
            z,_,_ = encoder(x)
            output = classifier(z)
            loss = lossFn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
            _, predictions = torch.max(output, 1)
            correctPredictions += (predictions == y).sum().item()
        
        currentAccuracy = (correctPredictions / len(trainLoader.dataset)) * 100
        history['train_loss'].append(totalLoss / len(trainLoader))
        history['train_acc'].append(currentAccuracy)

        avgTestAcc = 0
        for i, testLoader in enumerate(testLoaders):
            testAcc, testLoss = testNetwork(encoder, classifier, testLoader, lossFn, device)
            history[f'test_acc_{i+1}'].append(testAcc)
            history[f'test_loss_{i+1}'].append(testLoss)
            avgTestAcc += testAcc
        avgTestAcc /= len(testLoaders)
        if avgTestAcc > bestAcc:
            bestAcc = avgTestAcc
            bestEncState = copy.deepcopy(encoder.state_dict())
            bestClassState = copy.deepcopy(classifier.state_dict())
        scheduler.step(avgTestAcc)        
    return history,bestEncState,bestClassState

def SubjectChecker(lossFn, i, encode=0):
    # CHANGED: Logic to split files into 7 for training and 3 for testing
    matFilePaths = fileFinder(f'/home/coa23nt/EMG-SNN/Data/DB6_s{i}_a') + fileFinder(f'/home/coa23nt/EMG-SNN/Data/DB6_s{i}_b')
    matFilePaths.sort()
    dataPaths, targetDataPath = matFilePaths[:7], matFilePaths[7:]
    
    trainData = loadDataset(dataPaths, doEncode=encode)
    testDatasets = [loadDataset([path], doEncode=encode) for path in targetDataPath]
    dataNormalizer = DataNormalizer()
    if not encode:
        trainData = dataNormalizer.forwardTrain(trainData)
        testDatasets = [dataNormalizer.forward(ds) for ds in testDatasets]
    
    trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True)
    testLoaders = [DataLoader(ds, batch_size=batchSize, shuffle=False) for ds in testDatasets]

    # CHANGED: Initialize models from scratch instead of loading them
    encoder = Net_SLSTM_Extractor(inputSize=numChannels).to(device)
    classifier = Classifier(numClasses=numGestures).to(device)
    
    params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.Adam(params=params, lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    history,encoderState,classifierState = trainNetwork(encoder, classifier, trainLoader, testLoaders, numEpochs, lossFn, optimizer, device,scheduler)
    plotResults(history, i)

    model_state = {'encoder_state_dict': encoderState, 'classifier_state_dict': classifierState}
    torch.save(model_state,r"Subject_%s_SLSTM_TAB" % i)   
    print(f"\nSaved encoder and classifier models for subject {i}.")

# Main execution block
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
lossFn = nn.CrossEntropyLoss()
SubjectChecker(lossFn, inp, encode=0)