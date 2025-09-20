# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 17:59:41 2025

@author: Nia Touko
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy import io
import numpy as np
import glob
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import CCA # Import CCA
import snntorch as snn
from snntorch import surrogate

numSubjects = 1  
numGestures = 8
numRepetitions = 12
numChannels = 14
windowSize=400
TESTTHRESHOLD=0.3
inp = int(float(sys.argv[1]))

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
    
class DERBuffer:
    def __init__(self, device):
        self.x = None
        self.y = None
        self.yComp = None
        self.mem1 = None
        self.mem2 = None
        self.device = device

    def update(self, x, y, y_comp, mem1, mem2):
        self.x = x.clone().detach().to(self.device)
        self.y = y.clone().detach().to(self.device)
        self.yComp = y_comp.clone().detach().to(self.device)
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
                x.append(segment)
    
    x = torch.stack(x)    # [num_samples, window_size, num_channels]
    y = torch.tensor(y)   # [num_samples]

    
    return TensorDataset(x, y)



def extractOneOfEachGesture(dataset):    #should allow me to get the test dataset and DER Buffer
    uniqueLabels=[0,1,2,3,4,5,6,7]   
    X=list()
    Y=list()    
    lastY=-1
    for x,y in dataset:
        if lastY==y:
            X.append(x)
            Y.append(y)
        else:
            lastY=-1
        if y in uniqueLabels:
            X.append(x)
            Y.append(y)
            lastY=y
            uniqueLabels.remove(y)
    if uniqueLabels:
        print("Error, couldn't find", uniqueLabels)
    X = torch.stack(X)
    Y = torch.stack(Y)
    return TensorDataset(X,Y)

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
    
def extractGesturePerSession(listDatasets):
    samplesList = []
    for sessionData in listDatasets:
        gestureExtraction = extractOneOfEachGesture(sessionData)
        if gestureExtraction:
            samplesList.append(gestureExtraction)
    
    if not samplesList:
        return None

    X = torch.cat([ds.tensors[0] for ds in samplesList])
    Y = torch.cat([ds.tensors[1] for ds in samplesList])
    return TensorDataset(X, Y)

def loadModels(subjectID):
    snn_LSTM = Net_SLSTM_Extractor(inputSize=14, hiddenSize=128, numClasses=8).to(device)
    readout = Classifier(hiddenSize=128, numClasses=8).to(device)
    PATH = f"../Model-Pre-Gen-TTA-Test/Subject_{subjectID}_SLSTM_TAB"
    checkpoint = torch.load(PATH, map_location=device)
    snn_LSTM.load_state_dict(checkpoint['encoder_state_dict'])
    readout.load_state_dict(checkpoint['classifier_state_dict'])
    snn_LSTM.eval()
    readout.eval()
    return snn_LSTM, readout
    
def fileFinder(dataDirectory):
    searchPattern = os.path.join(dataDirectory, '*.mat')
    matFilePaths = glob.glob(searchPattern)  

    if not matFilePaths:
       raise ValueError("No .mat files found in this directory '%s', please double check and try again :)"%dataDirectory)
        
    return matFilePaths

def testNetwork(encoder, classifier, testLoader, loss_fn):
    encoder.eval()
    classifier.eval()
    correctPredictions = 0
    with torch.no_grad():
        for x, y in testLoader:
            x, y = x.permute(1, 0, 2).to(device), y.to(device)
            z, _, _ = encoder(x)
            output = classifier(z)
            _, predictions = torch.max(output, 1)
            correctPredictions += (predictions == y).sum().item()
    return 100 * correctPredictions / len(testLoader.dataset)

def testNetworkCCA(encoder, classifier, testLoader, loss_fn, cca_transformer):
    encoder.eval()
    classifier.eval()
    correctPredictions = 0
    hidden_size = classifier.fc.in_features
    with torch.no_grad():
        for x, y in testLoader:
            x, y = x.permute(1, 0, 2).to(device), y.to(device)
            z, _, _ = encoder(x)
            
            # Apply CCA transformation in the feature space
            z_np = z.detach().cpu().numpy()
            z_transformed_np = cca_transformer.transform(z_np)
            
            # Pad with zeros if CCA reduces dimensionality
            n_components = z_transformed_np.shape[1]
            if n_components < hidden_size:
                padding = np.zeros((z_transformed_np.shape[0], hidden_size - n_components))
                z_transformed_np = np.concatenate([z_transformed_np, padding], axis=1)
            
            z_transformed = torch.from_numpy(z_transformed_np.astype(np.float32)).to(device)
            output = classifier(z_transformed)
            _, predictions = torch.max(output, 1)
            correctPredictions += (predictions == y).sum().item()
    return 100 * correctPredictions / len(testLoader.dataset)

def SubjectChecker(loss_fn, i):
    matFilePaths = fileFinder(f'/home/coa23nt/EMG-SNN/Data/DB6_s{i}_a') + fileFinder(f'/home/coa23nt/EMG-SNN/Data/DB6_s{i}_b')
    dataPaths, targetDataPath = matFilePaths[:7], matFilePaths[7:]
    
    encoder, classifier = loadModels(i)
    
    trainListDataset = [loadDataset([path]) for path in dataPaths]
    testListDataset = [loadDataset([path]) for path in targetDataPath]
    
    fullTrainData = loadDataset(dataPaths)
    normaliseData = DataNormaliser()
    _ = normaliseData.forwardTrain(fullTrainData)
    trainListDataset = [normaliseData.forward(ds) for ds in trainListDataset]
    testListDataset = [normaliseData.forward(ds) for ds in testListDataset]
    
    # Get a sample of FEATURE VECTORS from the original training distribution (source)
    DERSamples = extractOneOfEachGesture(torch.utils.data.ConcatDataset(trainListDataset))
    x_der, _ = DERSamples.tensors
    with torch.no_grad():
        z_source_np, _, _ = encoder(x_der.permute(1, 0, 2).to(device))
    z_source_np = z_source_np.cpu().numpy()

    results = []
    for session_idx, testSessionData in enumerate(testListDataset):
        TTASamples = extractOneOfEachGesture(testSessionData)
        if not TTASamples:
            print(f"Skipping session {session_idx + 1}, not enough gesture samples.")
            continue
            
        x_tta, _ = TTASamples.tensors
        with torch.no_grad():
            z_target_sample_np, _, _ = encoder(x_tta.permute(1, 0, 2).to(device))
        z_target_sample_np = z_target_sample_np.cpu().numpy()
        
        # Fit CCA on the FEATURE vectors
        n_features = z_target_sample_np.shape[1]
        n_samples = z_target_sample_np.shape[0]
        n_components = min(n_samples, n_features)
        
        cca = CCA(n_components=n_components)
        cca.fit(z_target_sample_np, z_source_np[:n_samples]) # Match sample sizes
        
        # Evaluate performance
        originalLoader = DataLoader(testSessionData, batch_size=batchSize, shuffle=False)
        pre_acc = testNetwork(encoder, classifier, originalLoader, loss_fn)
        post_acc = testNetworkCCA(encoder, classifier, originalLoader, loss_fn, cca)
        results.append({'Session': session_idx + 1, 'Pre_CCA_Accuracy': pre_acc, 'Post_CCA_Accuracy': post_acc})

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"subject_{i}_CCA_feature_results.csv", index=False)
    
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df['Session'], results_df['Pre_CCA_Accuracy'], marker='o', linestyle='--', label='Before CCA Adaptation')
    ax.plot(results_df['Session'], results_df['Post_CCA_Accuracy'], marker='s', linestyle='-', label='After CCA Adaptation')
    ax.set_title(f'Subject {i} - Inter-Session Adaptation using CCA on Feature Space')
    ax.set_xlabel('Test Session Index')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(results_df['Session'])
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"subject_{i}_CCA_feature_results.png")


batchSize = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lossFn = nn.CrossEntropyLoss()
SubjectChecker(lossFn, inp)