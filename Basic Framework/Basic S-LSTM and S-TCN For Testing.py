import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader      #,random_split
import scipy
import numpy as np
from tqdm import tqdm
import glob
import os
from S_LSTM_Model import S_CLSTM

# Data parameters
numSubjects = 1  
numGestures = 8
numRepetitions = 12
numChannels = 14
windowSize=400

def loadDataset(mat_paths, windowSize=windowSize, stride=200, numClasses=numGestures): #def better ways to implement this for multi files but i was going quickly, also might be worth changing the labels to be like bianry strings or smthing later to reduce overfitting?
    x,y=[],[]
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
            labelWindow = labels[i:i+windowSize]
            label= int((torch.mode(labelWindow)[0]))   
            if label in labelMap:
                remappedLabel = labelMap[label]
                x.append(segment)
                y.append(remappedLabel) # Append the new, remapped label (e.g., 2 instead of 3)
                

    x = torch.stack(x)    # [num_samples, window_size, num_channels]
    y = torch.tensor(y)   # [num_samples]
 
    #probably should add a fallout for if std is 0 or add an epsilon but ah well     
    #mean = x.mean(dim=(0, 1), keepdim=True)
    #std = x.std(dim=(0, 1), keepdim=True)
    #x=(x-mean)/(std)
    
    return TensorDataset(x, y)


def normaliseData(dataset):
    x,y=dataset.tensors
    mean=x.mean(dim=0,keepdim=True)
    std=x.std(dim=0,keepdim=True)
    xNorm=(x-mean)/std
    return TensorDataset(xNorm,y)

    
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
        accuracies.append(currentAccuracy)
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
        

numEpochs=5

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
#model=STCN_Assembled(numChannels,[32, 32, 64, 64],numGestures).to(device)
model=S_CLSTM().to(device)


matFilePaths=fileFinder(r'C:\Users\Nia Touko\Downloads\DB6_s1_a')
testMatFilePaths=fileFinder(r'C:\Users\Nia Touko\Downloads\DB6_s1_b')

#matFilePaths=fileFinder(r'..\Data\DB6_s1_a')+fileFinder(r'..\Data\DB6_s1_b')+fileFinder(r'..\Data\DB6_s7_a')+fileFinder(r'..\Data\DB6_s7_b')
#testMatFilePaths=fileFinder(r'..\Data\DB6_s2_a')+fileFinder(r'..\Data\DB6_s2_b')[0:2]

matFilePaths.append(testMatFilePaths.pop(0))

#Isolates training and testing data, and shuffles the training (not the testing to simulate real world )

trainData=loadDataset(matFilePaths)
trainData=normaliseData(trainData)
testData=loadDataset(testMatFilePaths)
'''
data=loadDataset(mat_file_paths)
testSize=int(len(data)*TESTTHRESHOLD)
trainSize=len(data)-testSize
trainData,testData=random_split(data,[trainSize,testSize])
'''
batchSize=128
trainLoader=DataLoader(trainData, batch_size=batchSize, shuffle=True)
testLoader=DataLoader(testData,batch_size=batchSize,shuffle=False)

loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

losses,accuracies=trainNetwork(model,trainLoader,numEpochs,loss_fn,optimiser,0)
testNetwork(model,testLoader,loss_fn,optimiser,0)
