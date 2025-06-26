import torch
from torch.nn.utils import weight_norm 
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import TensorDataset, DataLoader,random_split
import scipy
import numpy as np
from tqdm import tqdm
import glob
import os
# Data parameters
numSubjects = 1  
numGestures = 7
numRepetitions = 12
numChannels = 14
windowSize=100

def loadDataset(mat_paths, windowSize=windowSize, stride=50, numClasses=numGestures): #def better ways to implement this for multi files but i was going quickly, also might be worth changing the labels to be like bianry strings or smthing later to reduce overfitting?
    x,y=[],[]
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
            label = int((torch.mode(label_window)[0])/2)    #floors it after dividing by 2 bc 2 sets of the same gesture created per thingy
            x.append(segment)
            y.append(label)
    x = torch.stack(x)    # [num_samples, window_size, num_channels]
    y = torch.tensor(y)   # [num_samples]
 
    #probably should add a fallout for if std is 0 or add an epsilon but ah well     
    mean = x.mean(dim=(0, 1), keepdim=True)
    std = x.std(dim=(0, 1), keepdim=True)
    x=(x-mean)/(std)
    
    return TensorDataset(x, y)

class Net_SLSTM(nn.Module):
    def __init__(self, inputSize=numChannels, hiddenSize=128, numClasses=7):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.numClasses = numClasses

        self.slstm1 = snn.SLSTM(inputSize, hiddenSize, spike_grad=surrogate.fast_sigmoid(),learn_threshold=True,reset_mechanism="subtract")
        self.slstm2 = snn.SLSTM(hiddenSize, hiddenSize, spike_grad=surrogate.fast_sigmoid(),learn_threshold=True,reset_mechanism="subtract")
        self.bn1 = nn.BatchNorm1d(hiddenSize)
        self.fc = nn.Linear(hiddenSize, numClasses)   #output
    def forward(self, x):  # x: [time, batch, features]
        syn1, mem1 = self.slstm1.init_slstm()
        syn2, mem2 = self.slstm2.init_slstm()
        mem2Rec = []
        spk1Rec=[]
        for step in range(x.size(0)):
            spk1, syn1, mem1 = self.slstm1(x[step], syn1, mem1)
            #spk1 = self.bn1(spk1) 
            spk1Rec.append(spk1)
        spk1Rec=torch.stack(spk1Rec)
        tSteps,bSize,nFeatures=spk1Rec.shape
        spk1Flat=spk1Rec.view(tSteps*bSize,nFeatures)
        spk1NormFlat=self.bn1(spk1Flat)
        spk1NormRec=spk1NormFlat.view(tSteps,bSize,nFeatures)
        for step in range(x.size(0)):
            spk2, syn2, mem2 = self.slstm2(spk1NormRec[step], syn2, mem2)
            mem2Rec.append(mem2)
            
        #Gonna try swapping mem2 with spk2 and see if I get increased accuracy next
        #spk2rec=torch.stack(spk2rec)
        #finalSpk=spk2rec.mean(dim=0)
        #out=self.fc(finalSpk)
        mem2Rec = torch.stack(mem2Rec)
        finalMem = mem2Rec.mean(dim=0)
        out = self.fc(finalMem)
        return out
    
class STCN_Extractor_Building_Block(nn.Module):
    def __init__(self,nInputs,nOutputs,kernelSize,stride,dilation):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(nInputs, nOutputs, kernelSize,stride=stride, dilation=dilation,padding='same'))
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(nOutputs, nOutputs, kernelSize,stride=stride, padding='same', dilation=dilation))
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        self.downsample = nn.Conv1d(nInputs, nOutputs, 1) if nInputs != nOutputs else None
        self.lif_res = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x, mem1, mem2, memRes):
        spk1, mem1 = self.lif1(self.conv1(x), mem1)
        spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
       
        # Residues
        res = x if self.downsample is None else self.downsample(x)
        outRes, memRes = self.lif_res(spk2 + res, memRes)
        return outRes, mem1, mem2, memRes

class STCN_Assembled(nn.Module): #channels are called stages to reduce confusion with electrode channels, and because I was listening to a song about putting on a preformance whilst coding
    def __init__(self,nInputs,nStages,nGestures,kernelSize=2):
        super().__init__()
        layers=[]
        for i in range(len(nStages)):
            dilations=2**i
            inStage=nInputs if i==0 else nStages[i-1]
            outStage=nStages[i]
            layers.append(STCN_Extractor_Building_Block(inStage,outStage,kernelSize,stride=1,dilation=dilations))
        self.net = nn.ModuleList(layers)
        self.fc=nn.Linear(nStages[-1],nGestures)
    def forward(self, x):
        memFwd = [None] * len(self.net) * 3  # 3 LIF neurons per block
        out = []
        
        # Temporal loop
        for t in range(x.size(0)):
            xt = x[t]   
            mem_idx = 0
            for i, layer in enumerate(self.net):
                if (i==0):
                    xt=xt.unsqueeze(2)
                xt, mem1out, mem2out, memResOut = layer(xt, memFwd[mem_idx], memFwd[mem_idx+1], memFwd[mem_idx+2])
                memFwd[mem_idx], memFwd[mem_idx+1], memFwd[mem_idx+2] = mem1out, mem2out, memResOut
                mem_idx += 3
            out.append(xt)
        out=torch.stack(out, dim=1)
        rateCode= torch.sum(out, dim=1)
        '''
        batch_size = rateCode.size(0)
        rateCode_flat = rateCode.view(batch_size, -1)
        '''
        out=self.fc(rateCode)
        return out
    def reset(self):
       for layer in self.modules():
           if hasattr(layer, 'reset_mem'):
               layer.reset_mem()
    
def fileFinder(dataDirectory):
    searchPattern = os.path.join(dataDirectory, '*.mat')
    matFilePaths = glob.glob(searchPattern)  #idk how good glob is but it made this way simpler so... ill revisit it later

    if not matFilePaths:
       raise ValueError("No .mat files found in this directory, please double check and try again :)")
        
    return matFilePaths

num_epochs=5
TESTTHRESHOLD=0.25

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
#model=STCN_Assembled(numChannels,[32, 32, 64, 64],numGestures).to(device)
model=Net_SLSTM().to(device)
 
matFilePaths=fileFinder(r'C:\Users\Nia Touko\Downloads\DB6_s1_a')
testMatFilePaths=fileFinder(r'C:\Users\Nia Touko\Downloads\DB6_s1_b')


matFilePaths.append(testMatFilePaths.pop(0))

#Isolates training and testing data, and shuffles the training (not the testing to simulate real world )
trainData=loadDataset(matFilePaths)
testData=loadDataset(testMatFilePaths)
'''
data=loadDataset(mat_file_paths)
testSize=int(len(data)*TESTTHRESHOLD)
trainSize=len(data)-testSize
trainData,testData=random_split(data,[trainSize,testSize])
'''
trainLoader=DataLoader(trainData, batch_size=windowSize, shuffle=True)
testLoader=DataLoader(testData,batch_size=windowSize,shuffle=False)

#Training Settings
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses=[]
accuracies=[]
for epoch in range(num_epochs):
    model.train()
    totalLoss = 0
    correctPredictions=0
    loop = tqdm(trainLoader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) #pretty little progress bar
    for x, y in trainLoader:    #loops through trainloader
        #Data is moved to the right place
        x = x.permute(1,0,2)
        x, y = x.to(device), y.to(device)
        #model.reset()   #resets the membrane potential of the LIF neurons (is only needed for the S-TCN architecute)
        output = model(x)
        #calculates the training values
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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

model.eval()
testLoss=0
correctPredictions=0
testLoop = tqdm(testLoader, desc="Testing", leave=False)
with torch.no_grad():
    for x,y in testLoop:
        x=x.permute(1,0,2)
        x,y=x.to(device),y.to(device)
        #model.reset()
        output=model(x)
        loss = loss_fn(output, y)
        _, predictions = torch.max(output, 1)
        testLoss+=loss.item()
        correctPredictions += (predictions == y).sum().item()
        currentTestAccuracy = (correctPredictions / (testLoop.n+1))
        testLoop.set_postfix(loss=loss.item(), acc=f"{currentTestAccuracy:.2f}%")
        
    print(f"Total loss is {testLoss/len(testLoader):.4f} and your final accuracy is {currentTestAccuracy:.4f}%, compared to your final training loss of {losses[-1]:.4f} and accurcay {currentAccuracy:.4f}% ")
    