import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader
import scipy
import numpy as np
from tqdm import tqdm

# Data parameters
numSubjects = 1  
numGestures = 7
numRepetitions = 12
numChannels = 14
numSteps = 2000 # Dataset is sampled at 2000Hz so 1s (might downsample this later)
batchSize=100
def loadDataset(mat_path, batchSize=batchSize, stride=50, numClasses=numGestures): #need to make this collate all the different files but this will do for now
    mat = scipy.io.loadmat(mat_path)
    
    emg = mat['emg']             
    labels = mat['restimulus']   
    emg = np.delete(emg, [8, 9], axis=1)    #don't contain information
    emg = torch.tensor(emg, dtype=torch.float32)    
    labels = torch.tensor(labels, dtype=torch.long).squeeze()
    
    # Normalize EMG
    emg = (emg - emg.mean(dim=0)) / emg.std(dim=0)

    # Sliding window segmentation for batches (idk if this should be done when not doing tdnn work on transhumeral data)
    x=[]; y=[];
    for i in range(0, emg.shape[0] - batchSize, stride):
        segment = emg[i:i+batchSize]              # [window_size, channels]
        label_window = labels[i:i+batchSize]
        label = int((torch.mode(label_window)[0])/2)    #floors it after dividing by 2 bc 2 sets of the same gesture created per thingy
        x.append(segment)
        y.append(label)
    
    x = torch.stack(x)    # [num_samples, window_size, num_channels]
    y = torch.tensor(y)   # [num_samples]
    return TensorDataset(x, y)

x=loadDataset(r'C:\Users\Nia Touko\Downloads\DB6_s1_a\S1_D3_T2'  )



# Hopefully a working basic LSTM model based on the documentation for snn.slstm (Can't test till I make the data loader that I am procrastinating)
class Net_LSTM(nn.Module):
    def __init__(self, inputSize=numChannels, hiddenSize=128, numClasses=7):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.numClasses = numClasses

        self.slstm1 = snn.SLSTM(inputSize, hiddenSize, spike_grad=surrogate.fast_sigmoid(),learn_threshold=True,reset_mechanism="subtract")
        self.slstm2 = snn.SLSTM(hiddenSize, hiddenSize, spike_grad=surrogate.fast_sigmoid(),learn_threshold=True,reset_mechanism="subtract")
        self.bn1 = snn.BatchNormTT1d(hiddenSize,time_steps=batchSize)
        self.fc = nn.Linear(hiddenSize, numClasses)   #output
    def forward(self, x):  # x: [time, batch, features]
       
        batch_size = x.size(1)
        device = x.device
        syn1 = torch.zeros(batch_size, self.hiddenSize, device=device)  #this instead of mem_reset() because I was running into issues with wrong number of samples in certain batch sizes
        mem1 = torch.zeros(batch_size, self.hiddenSize, device=device)
        syn2 = torch.zeros(batch_size, self.hiddenSize, device=device)
        mem2 = torch.zeros(batch_size, self.hiddenSize, device=device)
        
        mem2_rec = []

        for step in range(x.size(0)):
            spk1, syn1, mem1 = self.slstm1(x[step], syn1, mem1)
            #spk1 = self.bn1(spk1) # bn1 works correctly on the [batch, hidden] tensor
            
            spk2, syn2, mem2 = self.slstm2(spk1, syn2, mem2)
            
            mem2_rec.append(mem2)

        mem2_rec = torch.stack(mem2_rec)
        final_mem = mem2_rec.mean(dim=0)
    
        out = self.fc(final_mem)
        return out


    
num_epochs=10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model=Net_LSTM().to(device)
#model=Net_TCN(numChannels,numGestures).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses=[]
trainLoader=DataLoader(x, batch_size=batchSize, shuffle=True)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correctPredictions=0
    loop = tqdm(trainLoader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
    for x, y in trainLoader:
        x = x.permute(1, 0, 2)
        #print(x)
        x, y = x.to(device), y.to(device)
        output = model(x)

        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.update(1)
        _, predictions = torch.max(output, 1)
        correctPredictions += (predictions == y).sum().item()
        currentAccuracy = (correctPredictions / loop.n)
        loop.set_postfix(loss=loss.item(), acc=f"{currentAccuracy:.2f}%")
        
    losses.append(total_loss / len(trainLoader))
    print(f"Epoch {epoch+1}: Loss = {total_loss / len(trainLoader):.4f}")