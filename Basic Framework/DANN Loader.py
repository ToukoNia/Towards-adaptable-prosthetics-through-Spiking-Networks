# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 22:51:23 2025

@author: Nia Touko
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader      #,random_split
import scipy
import numpy as np
from tqdm import tqdm
import glob
import os
from SCLSTM_DANN_Model import S_CLSTM_DANN
import re 
import pandas as pd
# Data parameters
numSubjects = 1  
numGestures = 8
numRepetitions = 12
numChannels = 14
windowSize=400

def loadDataset(mat_paths, windowSize=windowSize, overlapR=0.5, numClasses=numGestures,subjectMap={1:0}): #def better ways to implement this for multi files but i was going quickly, also might be worth changing the labels to be like bianry strings or smthing later to reduce overfitting?
    x,y,subject=[],[],[]
    stride=int((1-overlapR)*windowSize)
    targetLabels = [0, 1, 3, 4, 6, 9, 10, 11]
    labelMap = {label: i for i, label in enumerate(targetLabels)}
    for mat_path in tqdm(mat_paths, desc="Files"):
        basename = os.path.basename(mat_path)
        subjectID = re.search(r'(?<=S)\d+(?=_)', basename)
        if not subjectID:
            continue
        subjectID = int(subjectID.group())
        if subjectID not in subjectMap:
            continue
        domainLabel = subjectMap[subjectID]
        
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
                subject.append(domainLabel)
                x.append(segment)
            
    x = torch.stack(x)    # [num_samples, window_size, num_channels]
    y = torch.tensor(y)   # [num_samples]
    subjects=torch.tensor(subject)
    #probably should add a fallout for if std is 0 or add an epsilon but ah well     
    return TensorDataset(x, y, subjects)

def fileFinder(dataDirectory):
    searchPattern = os.path.join(dataDirectory, '*.mat')
    matFilePaths = glob.glob(searchPattern)  #idk how good glob is but it made this way simpler so... ill revisit it later

    if not matFilePaths:
       raise ValueError("No .mat files found in this directory, please double check and try again :)")
        
    return matFilePaths

alpha=1
def trainNetwork(model,trainLoader,numEpochs,optimizer):
    #Training Settings

    losses=[]
    accuracies=[]
    for epoch in range(numEpochs):
        model.train()
        totalLoss = 0
        correctPredictions=0
        i=0
        loop = tqdm(trainLoader, desc=f"Epoch {epoch+1}/{numEpochs}", leave=False) #pretty little progress bar
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
            optimizer.zero_grad()
            loss = lossGesture + lossDomain 
            loss.backward()
            optimizer.step()
            #updates the readOuts
            totalLoss += lossGesture.item()
            loop.update(1)
            _, predictions = torch.max(gestureOutput, 1)
            correctPredictions += (predictions == y).sum().item()
            currentAccuracy = (correctPredictions / loop.n)
            loop.set_postfix(loss=lossGesture.item(), acc=f"{currentAccuracy:.2f}%")
        accuracies.append(currentAccuracy)
        losses.append(totalLoss / len(trainLoader))
        print(f"Epoch {epoch+1}: Loss = {losses[-1]:.4f}")
    return losses, accuracies

def testNetwork(model, testLoader,optimizer):
    model.eval()
    testLoss=0
    correctPredictions=0
    with torch.no_grad():
        testLoop = tqdm(testLoader, desc="Testing on unseen  ", leave=False)
        i=0
        for x,y,_ in testLoop:
            x=x.permute(1,0,2)
            x,y=x.to(device),y.to(device)
            output,_=model(x,alpha=0)
            loss = lossGestureFn(output, y)
            _, predictions = torch.max(output, 1)
            testLoss+=loss.item()
            i+=1
            correctPredictions += (predictions == y).sum().item()
            currentTestAccuracy = (correctPredictions /i)
            testLoop.set_postfix(loss=loss.item(), acc=f"{currentTestAccuracy:.2f}%")
        return testLoss, correctPredictions
       

numEpochs=8
batchSize=128

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
#model=STCN_Assembled(numChannels,[32, 32, 64, 64],numGestures).to(device)


def LOSO(): #Leave one subject out: iterates through all the subjects, leaving one out to test how it adapts to unseen users (and reduces erronous data from individual people)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    matFilePaths=[]
    SUBJECT_IDs=[1,2,3,4,5,6,7,8,9,10]
    results = {}
    for i in range(1,11):
        matFilePaths+=fileFinder(r'..\Data\DB6_s%s_a' % i)+fileFinder(r'..\Data\DB6_s%s_b' % i)
    for i in range(0,1):
        #sets up the data
        trainSubjectIDs=SUBJECT_IDs
        testSubject=trainSubjectIDs.pop(i)
        subject_map = {original_id: i for i, original_id in enumerate(trainSubjectIDs)}
        dataPaths=matFilePaths
        targetDataPath=dataPaths[i:i+10]
        dataPaths=matFilePaths[:i]+matFilePaths[i+10:]
        testData=loadDataset(targetDataPath,subjectMap={testSubject: 0})
        trainData=loadDataset(dataPaths,subjectMap=subject_map)
        
        trainLoader=DataLoader(trainData, batch_size=batchSize, shuffle=True)
        testLoader=DataLoader(testData,batch_size=batchSize,shuffle=False)
        #Load and run the network
        model=S_CLSTM_DANN(numSubjects=len(trainSubjectIDs)).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses,accuracies=trainNetwork(model,trainLoader,numEpochs,optimiser)
        testLoss,testAccuracy=testNetwork(model,testLoader,optimiser)

        #outputs
        fTrainLoss = losses[-1] if losses else float('nan')
        fTrainAcc = accuracies[-1] if accuracies else float('nan')
        print(f"Final Training Accuracy: {fTrainAcc:.4f}, Loss: {fTrainLoss:.4f}")

        # 5. Test the Network
        print(f"Test Accuracy: {testAccuracy:.4f}, Loss: {testLoss:.4f}")

        # 6. Save the results for this fold into the dictionary
        results[f"Subject {testSubject}"] = {
            "training_accuracy": fTrainAcc,
            "training_loss": fTrainLoss,
            "testing_accuracy": testAccuracy,
            "testing_loss": testLoss,
        }
        torch.save(model,r'Subject%sLeftOut' % i)
    print("\nCross-validation finished. Saving final results...")
    
    # Save as a CSV file 
    resultsDF = pd.DataFrame.from_dict(results, orient='index')
    resultsDF.index.name = 'Fold'
    # Calculate and add average results (to easily see how valid a model is)
    mean_results = resultsDF.mean()
    resultsDF.loc['Average'] = mean_results
    resultsDF.to_csv('LOSO_DANN_Results.csv')
    
    print("Final results saved to LOSO_DANN_Results.csv")
    return results
             

lossGestureFn = nn.CrossEntropyLoss()
lossDomainFn = nn.CrossEntropyLoss()
    
LOSO()