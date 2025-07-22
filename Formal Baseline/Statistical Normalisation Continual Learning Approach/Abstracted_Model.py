# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 16:30:25 2025

@author: Nia Touko
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader,random_split
from snntorch import spikegen
from scipy import io
import numpy as np
import glob
import os
from SLSTM_GMM_Model import Net_SLSTM_Extractor,Classifier
from SLSTM_GMM_Model import sliced_wasserstein_distance, GMM
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import copy
import torch.nn.functional as F
def consolidated_loss_entropy(encTarg, gmm, classifier, lambdaSwd, batchSize):
    
    output = classifier(encTarg)
    softmax_out = F.softmax(output, dim=1)
    log_softmax_out = F.log_softmax(output, dim=1)
    
    loss_entropy = -torch.sum(softmax_out * log_softmax_out, dim=1).mean()

    pseudoSamples, _ = gmm.sample(batchSize)
    lossSwd = sliced_wasserstein_distance(encTarg, pseudoSamples)
    total = loss_entropy + lambdaSwd * lossSwd
    return total, loss_entropy, lossSwd

def consolidated_loss(encTarg, gmm, classifier, lambdaSwd, batchSize):
    pseudoSamples,pseudoLabels = gmm.sample(batchSize)
    #Classifier Loss    
    classifierOut = classifier(pseudoSamples)
    lossClass = nn.CrossEntropyLoss()(classifierOut, pseudoLabels)

    #Distribution Allignment
    lossSwd = sliced_wasserstein_distance(encTarg, pseudoSamples)
    total = lossClass + lambdaSwd * lossSwd

    return total, lossClass, lossSwd 

def trainNetwork(encoder,classifier,trainLoader,numEpochs,loss_fn,optimiser):
    #Training Settings
    history = {
       'train_loss': [], 'train_acc': [],
       'intra_session_loss': [], 'intra_session_acc': [],
       'inter_session_loss': [], 'inter_session_acc': []
   }

    for epoch in range(numEpochs):
        encoder.train()
        classifier.train()
        totalLoss = 0
        correctPredictions=0
        for x, y in trainLoader:    #loops through trainloader
            #Data is moved to the right place
            x = x.permute(1,0,2)
            x, y = x.to(device), y.to(device)
          
            outFeatures = encoder(x)
            output=classifier(outFeatures)
            
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
        print("Epoch Completed")
    encoder.eval()
    latentVectors, allLabels = [],[]
    with torch.no_grad():
        for x, y in trainLoader:
            x = x.permute(1, 0, 2).to(device)
            latentZ = encoder(x)
            latentVectors.append(latentZ)
            allLabels.append(y)
    latentVectors = torch.cat(latentVectors, dim=0)
    allLabels = torch.cat(allLabels, dim=0).to(device)
    
    gmm = GMM(n_components=8, n_features=128, device=device)
    gmm.fit(latentVectors, allLabels)
    return encoder,classifier,gmm,history

def EMABasedTTACycle(encoder,classifier,gmm,dataLoader,lossFn):
    fastEncoder=encoder
    fastClassifier=classifier
    slowEncoder=copy.deepcopy(fastEncoder)
    slowClassifier=copy.deepcopy(fastClassifier)
    
    fastEncoder.train()
    fastClassifier.train()
    slowEncoder.eval()
    slowClassifier.eval()
    params = list(encoder.parameters()) + list(classifier.parameters())
    optimiser = torch.optim.Adam(params, lr=1e-4) # Smaller LR for fine-tuning
    lambda_swd = 1.0
    emaMomentum=0.999
    
    correctPredictions=0
    samples=0
    testLoss=0
    for i,(x,y) in enumerate(dataLoader):
        
        x, y = x.permute(1, 0, 2).to(device), y.to(device)
        optimiser.zero_grad()
        latentFeaturesFast = fastEncoder(x)
        adaptionLoss,_,_ = consolidated_loss_entropy(
            latentFeaturesFast, gmm, fastClassifier, lambda_swd, x.size(1)
        )
        adaptionLoss.backward()
        optimiser.step()
        # The formula is: slow_weight = momentum * slow_weight + (1 - momentum) * fast_weight
        with torch.no_grad():
            for slow_param, fast_param in zip(slowEncoder.parameters(), fastEncoder.parameters()):
                slow_param.data.mul_(emaMomentum).add_(fast_param.data, alpha=1 - emaMomentum)
            for slow_param, fast_param in zip(slowClassifier.parameters(), fastClassifier.parameters()):
                slow_param.data.mul_(emaMomentum).add_(fast_param.data, alpha=1 - emaMomentum)
        with torch.no_grad():
            latentFeatures = slowEncoder(x)
            output = slowClassifier(latentFeatures)

            _, predictions = torch.max(output, 1)
            correctPredictions += (predictions == y).sum().item()
            samples+=x.size(1)
        sAdaptionLoss,_,_ = consolidated_loss_entropy(
            latentFeatures, gmm, slowClassifier, lambda_swd, x.size(1)
        )
        testLoss+=lossFn(output,y)
        if (i + 1) % 20 == 0:
            current_acc = 100 * correctPredictions / samples
            print(f"Batch {i+1}/{len(dataLoader)}, "
                  f"Fast Adaptation Loss: {adaptionLoss.item():.4f}, "
                  f"Slow Adaptation Loss: {sAdaptionLoss.item():.4f}, "
                  f"Actual Loss: {testLoss.item()/i:.4f}, "
                  f"Current Target Accuracy: {current_acc:.2f}%")
            
    testAccuracy=100*correctPredictions/len(dataLoader.dataset)
    return testAccuracy


def generateSyntheticDataset(numSamples=10000, numClasses=8, numChannels=14,
                             windowSize=400, overlapR=0.5, doEncode=False,
                             meanShift=0.0, stdMultiplier=1.0):
 
    x, y = [], []
    stride = int((1 - overlapR) * windowSize)
    all_class_data = []
    chunkSize = numSamples // numClasses
    
    for i in range(numClasses):
        class_specific_mean = (torch.rand(numChannels) - 0.5) * (i + 1) * 0.5
        class_data = (torch.randn(chunkSize, numChannels) * stdMultiplier) + class_specific_mean
        class_data = class_data + meanShift
        class_labels = torch.full((chunkSize,), fill_value=i, dtype=torch.long)
        
        all_class_data.append((class_data, class_labels))

    random.shuffle(all_class_data)
    
    rawEmgData = torch.cat([d[0] for d in all_class_data], dim=0)
    rawLabels = torch.cat([d[1] for d in all_class_data], dim=0)
    
    for i in range(0, rawEmgData.shape[0] - windowSize, stride):
        segment = rawEmgData[i:i + windowSize]
        labelWindow = rawLabels[i:i + windowSize]

        if len(labelWindow) > 0:
            label = int(torch.mode(labelWindow).values)
            y.append(label)

            if doEncode:
                encodedSegment = spikegen.delta(segment, threshold=1e-5, off_spike=True)
                x.append(encodedSegment)
            else:
                x.append(segment)

    xTensor = torch.stack(x)
    yTensor = torch.tensor(y, dtype=torch.long)

    return TensorDataset(xTensor, yTensor)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
numClasses = 8
featureDim = 14  
hiddenDim = 128
windowSize = 400
numEpochs = 5
batchSize = 128

print("\nGenerating synthetic source and target datasets")
sourceDataset = generateSyntheticDataset(
    numSamples=10000000, numClasses=numClasses, numChannels=featureDim,
    windowSize=windowSize, meanShift=0.0, stdMultiplier=1.0
)
sourceLoader = DataLoader(sourceDataset, batch_size=batchSize, shuffle=True)

targetDataset = generateSyntheticDataset(numSamples=4280000, numClasses=numClasses, numChannels=featureDim,windowSize=windowSize, meanShift=0.5, stdMultiplier=1.5)
targetLoader = DataLoader(targetDataset, batch_size=batchSize, shuffle=False)
print(f"Source samples: {len(sourceDataset)}, Target samples: {len(targetDataset)}")

encoder = Net_SLSTM_Extractor(inputSize=featureDim, hiddenSize=hiddenDim).to(device)
classifier = Classifier(hiddenSize=hiddenDim, numClasses=numClasses).to(device)

print("\n--- Phase 1: Training on Source Data ---")
lossFn = nn.CrossEntropyLoss()
params = list(encoder.parameters()) + list(classifier.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)

encoder,classifier,gmm,history = trainNetwork(encoder,classifier,sourceLoader,numEpochs,lossFn,optimizer)

print(f"Source training complete. Final training accuracy: {history['train_acc'][-1]:.2f}%")

print("\n--- Phase 2: Test-Time Adaptation on Target Data ---")
finalAcc = EMABasedTTACycle(
    encoder, classifier, gmm, targetLoader, lossFn
)
print(f"\nAdaptation complete. Final accuracy on target data: {finalAcc:.2f}%")
