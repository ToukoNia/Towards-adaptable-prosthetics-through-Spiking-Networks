# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 21:49:19 2025

@author: touko
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
from torch.cuda.amp import autocast, GradScaler
# Data parameters
numSubjects = 1  
numGestures = 8
numRepetitions = 12
numChannels = 14
windowSize=400
TESTTHRESHOLD=0.3
inp = int(float(sys.argv[1]))
#inp=1

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
                if doEncode:
                    encodedSegment=spikegen.delta(segment,threshold=1e-5)
                    x.append(encodedSegment)
                else:
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
    
    
def fileFinder(dataDirectory):
    searchPattern = os.path.join(dataDirectory, '*.mat')
    matFilePaths = glob.glob(searchPattern)  

    if not matFilePaths:
       raise ValueError("No .mat files found in this directory '%s', please double check and try again :)"%dataDirectory)
        
    return matFilePaths


def trainNetwork(encoder,classifier,trainLoader,testLoader,numEpochs,loss_fn,optimiser,adaptOpt,derLoader,ttaLoader):
    #Training Settings
    history = {
       'train_loss': [], 'train_acc': [],
       'intra_pre_acc':[], 'intra_pre_loss':[],'intra_acc_best': [], 'intra_acc_worst': [], 'intra_acc_mean': [], 'intra_acc_final': [],
       'intra_loss_best': [], 'intra_loss_worst': [], 'intra_loss_mean': [], 'intra_loss_final': []
   }
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
            #updates the readOuts
            totalLoss += loss.item()
            _, predictions = torch.max(output, 1)
            correctPredictions += (predictions == y).sum().item()
        currentAccuracy = (correctPredictions / len(trainLoader.dataset))*100
        history['train_loss'].append(totalLoss / len(trainLoader))
        history['train_acc'].append(currentAccuracy)

        
      
        
        
        
        print(f"Epoch {epoch+1}/{numEpochs} | Train Acc: {history['train_acc'][-1]:.2f}% | "
             f"Intra-TTA Final Acc: {history['intra_acc_final'][-1]:.2f}% (Best: {history['intra_acc_best'][-1]:.2f}%) | "
             f"Intra Pre Acc: {history['intra_pre_acc'][-1]:.2f}%")
    return history

def TTATester(encoder, classifier, der_buffer,dataLoader, ttaLoader,adaptOpt):  
    encoder_state_before = copy.deepcopy(encoder.state_dict())
    classifier_state_before = copy.deepcopy(classifier.state_dict())
    
    results = TTA(encoder,classifier,der_buffer,dataLoader, ttaLoader,adaptOpt)
    
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


def rbf_kernel(X, Y, kernel_mul=2.0, kernel_num=5):
    dist_sq = torch.cdist(X, Y, p=2).pow(2)
    # Calculate bandwidth for the kernel using median heuristic

    with torch.no_grad():
        total = torch.cat([X, Y], dim=0)
        total_dist_sq = torch.cdist(total, total, p=2).pow(2)
        bandwidth = total_dist_sq[total_dist_sq > 0].median()
    
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    # RBF kernel formula
    kernel_val = [torch.exp(-dist_sq / (bw)) for bw in bandwidth_list]
    
    return sum(kernel_val)

def MMDLoss(source, target, kernel_mul=2.0, kernel_num=5):
    xx = rbf_kernel(source, source, kernel_mul, kernel_num)
    yy = rbf_kernel(target, target, kernel_mul, kernel_num)
    xy = rbf_kernel(source, target, kernel_mul, kernel_num)

    # The MMD loss is the mean of each component
    loss = torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy)
    return loss

def TTA(encoder,classifier,der_buffer,dataLoader,ttaLoader, adaptOpt,alpha=1,beta=1,nAdaption=150):    #Need to make this not loop through the whole dataset, maybe make it take one example of each gesture type and use that?
    scaler = GradScaler()
    beta=1
    alpha=1
    ttaAcc,ttaLoss,statLosses,derLosses=[],[],[],[]
    lossFn=nn.CrossEntropyLoss()
    preAcc, preLoss = testNetwork(encoder, classifier, dataLoader, lossFn)
    ttaAcc.append(preAcc)
    ttaLoss.append(preLoss)
    statLosses.append(0)
    derLosses.append(0)
    encoder.train()
    classifier.train()
    
    for _ in range(nAdaption):
        x_batch,_=next(iter(ttaLoader))
        x_batch = x_batch.permute(1, 0, 2).to(device)
        with autocast():
          z, mem1_te, mem2_te=encoder(x_batch)
          #meanLoss=torch.pow(mem1_te.mean(0).mean(0)-der_buffer.mem1.mean(0).mean(0),2).mean()+torch.pow(mem2_te.mean(0).mean(0)-der_buffer.mem2.mean(0).mean(0),2).mean()
          #stdLoss=torch.pow(mem1_te.std(0).mean(0)-der_buffer.mem1.std(0).mean(0),2).mean()+torch.pow(mem2_te.std(0).mean(0)-der_buffer.mem2.std(0).mean(0),2).mean()
          statLoss = MMDLoss(mem1_te.mean(0),der_buffer.mem1.mean(0)) + MMDLoss(mem2_te.mean(0),der_buffer.mem2.mean(0))
          #statLoss=meanLoss
          
          zDER, _,_=encoder(der_buffer.x)
          yDER=classifier(zDER)
        derLoss=torch.pow(yDER-der_buffer.yComp,2).mean()+nn.CrossEntropyLoss()(yDER,der_buffer.y)


        loss=alpha*statLoss+beta*derLoss
        adaptOpt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(adaptOpt)
        scaler.update()
        acc, loss = testNetwork(encoder, classifier, dataLoader, lossFn)
        ttaAcc.append(acc)
        ttaLoss.append(loss)
        statLosses.append(statLoss.clone().detach())
        derLosses.append(derLoss.clone().detach())
    results = {
         'loss':ttaLoss,
         'intra_session_acc': ttaAcc,
         'statistical_loss': statLosses,
         'der_loss': derLosses
     }
     
    return results 

        
   
def plot_results(history, subject_id):
    #Plots training and validation metrics and saves the figure.
    df = pd.DataFrame(history)
    
    plt.style.use('seaborn-whitegrid')    
    fig, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4, figsize=(15, 6))
    
    # Plotting Loss
    ax1.plot(df['loss'], label='Intra-Session Test Loss', color='black')
    ax1.set_title(f'Subject {subject_id} - Model Loss')
    ax1.set_xlabel('Adaption Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plotting Accuracy
    ax2.plot(df['intra_session_acc'], label='Intra-Session Test Accuracy', color='black')
    ax2.set_title(f'Subject {subject_id} - Model Accuracy')
    ax2.set_xlabel('Adaption Steps')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    #Plotting Statistical Loss
    ax3.plot(df['statistical_loss'], label='TTA Loss Statistic', color='black')
    ax3.set_title(f'Subject {subject_id} - Statistic Loss')
    ax3.set_xlabel('Adaption Steps')
    ax3.set_ylabel('Loss Value')
    ax3.legend()
    
    #Plotting DER Loss
    ax4.plot(df['der_loss'], label='DER Loss', color='black')
    ax4.set_title(f'Subject {subject_id} - DER Loss')
    ax4.set_xlabel('Adaption Steps')
    ax4.set_ylabel('Loss Value')
    ax4.legend()
    
    plt.tight_layout()
    # Save the figure.
    plt.savefig(f"subject_{subject_id}_training_results.png")

    
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

def extractGesturePerSession(listDatasets):
    samplesList=list()
    for sessionData in listDatasets:
        gestureExtraction = extractOneOfEachGesture(sessionData)
        samplesList.append(gestureExtraction)
        
    X = torch.cat([ds.tensors[0] for ds in samplesList])
    Y = torch.cat([ds.tensors[1] for ds in samplesList])
    sampleDataset = TensorDataset(X, Y)
    return sampleDataset
    
def loadModels(subjectID):
    snn_LSTM = Net_SLSTM_Extractor(inputSize=14, hiddenSize=128, numClasses=8).to(device)
    readout = Classifier(hiddenSize=128, numClasses=8).to(device)
    PATH = f"Subject_{subjectID}_SLSTM_TAB"

    # 2. Load the entire dictionary object
    checkpoint = torch.load(PATH)
    
    # 3. Load the state dictionaries into your model instances
    snn_LSTM.load_state_dict(checkpoint['encoder_state_dict'])
    readout.load_state_dict(checkpoint['classifier_state_dict'])
    
    # 4. Set the models to evaluation mode (very important for inference)
    snn_LSTM.eval()
    readout.eval()
    return snn_LSTM,readout
def SubjectChecker(loss_fn,i,encode=0):

    matFilePaths=fileFinder(r'/home/coa23nt/EMG-SNN/Data/DB6_s%s_a'%i)+fileFinder(r'/home/coa23nt/EMG-SNN/Data/DB6_s%s_b'%i)
    dataPaths=matFilePaths[:7]
    targetDataPath=matFilePaths[7:]
    testDataIntra=loadDataset(targetDataPath,doEncode=encode)
    trainData=loadDataset(dataPaths,doEncode=encode)
    normaliseData=DataNormaliser()
    
    encoder,classifier=loadModels(i)
    
    testListDataset = [loadDataset([path], doEncode=encode) for path in targetDataPath]
    trainListDataset = [loadDataset([path], doEncode=encode) for path in dataPaths]
    if not encode:  
        trainData=normaliseData.forwardTrain(trainData)
        #testDataInter=normaliseData.forward(testDataInter)
        testDataIntra=normaliseData.forward(testDataIntra)
        testListDataset = [normaliseData.forward(ds) for ds in testListDataset]
        trainListDataset=[normaliseData.forward(ds) for ds in trainListDataset]
    
    DERSamples=extractGesturePerSession(trainListDataset)
    TTASamples=extractGesturePerSession(testListDataset)
    
    TTALoader=DataLoader(TTASamples,batch_size=len(TTASamples),shuffle=True)
    DERLoader=DataLoader(DERSamples,batch_size=len(DERSamples),shuffle=True)
    
    derBuffer=DERBuffer(device)
    x_der, y_der = next(iter(DERLoader))
    derBuffer.x = x_der.permute(1, 0, 2).to(device)
    derBuffer.y = y_der.to(device)
    
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        z_der, mem1_der, mem2_der = encoder(derBuffer.x)
        output_der = classifier(z_der)
    
    derBuffer.yComp = output_der.clone().detach()
    derBuffer.mem1 = mem1_der.clone().detach()
    derBuffer.mem2 = mem2_der.clone().detach()
    
    
    #trainData,testDataInter= loadAndSplitPerSession(dataPaths) 
    testLoaderIntra=DataLoader(testDataIntra,batch_size=batchSize,shuffle=False)
    #testLoaderInter=DataLoader(testDataInter,batch_size=batchSize,shuffle=False)
    print(f"Created TTASamples dataset with {len(TTASamples)} windows.")
    print(f"Created combined testDataIntra with {len(testDataIntra)} windows.")
    
    
    adaptOpt=torch.optim.Adam(params=list(encoder.parameters())+list(classifier.parameters()),lr=1e-3)
  
    history=TTATester(encoder, classifier, derBuffer, testLoaderIntra, TTALoader, adaptOpt)
    results_df = pd.DataFrame(history)
    results_df.to_csv(f"subject_{i}_training_history.csv", index_label="Adaption Step")
    print(f"\nResults for subject {i} saved to subject_{i}_training_history.csv")
    plot_results(history,i)
   
            
    
batchSize=128

    

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

lossFn = nn.CrossEntropyLoss()
SubjectChecker(lossFn,inp,encode=0)
