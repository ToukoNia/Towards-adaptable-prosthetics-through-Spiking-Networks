import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
import os
from scipy import io
import snntorch as snn
from snntorch import surrogate
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Data_Manager(nn.Module):
    
    def __init__(self, N_data=10, N_features=2, Time=5,device=device):
        super().__init__()

        self.device = device
        self.X_tr=torch.rand([Time,N_data,N_features]).to(self.device)
        self.Y_tr=torch.zeros([N_data,2]).to(self.device)
        self.Y_tr[0:int(N_data/2),0]=1
        self.Y_tr[int(N_data/2):,1]=1
        self.X_te=self.X_tr*(2*torch.rand([1,N_data,N_features]).to(self.device)-0.5)+torch.rand([1,N_data,N_features]).to(self.device)
        self.Y_te=self.Y_tr

    def Batch(self,batch_size):

        rand_ind=torch.randint(0,self.X_tr.size()[1],[batch_size])

        x=self.X_tr[:,rand_ind,:]
        y=self.Y_tr[rand_ind,:]

        return x, y

    def Test(self):

        return self.X_te, self.Y_te
Data=Data_Manager()
numChannels=14
class Net_SLSTM_Extractor(nn.Module):
    def __init__(self, inputSize=2, hiddenSize=128, numClasses=8):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.numClasses = numClasses
        self.slstm1 = snn.SLSTM(inputSize, hiddenSize, spike_grad=surrogate.fast_sigmoid(),learn_threshold=True,reset_mechanism="subtract")
        self.slstm2 = snn.SLSTM(hiddenSize, hiddenSize, spike_grad=surrogate.fast_sigmoid(),learn_threshold=True,reset_mechanism="subtract")
        #self.bn1 = nn.BatchNorm1d(hiddenSize)
        self.inputBn=nn.BatchNorm1d(inputSize)

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
    

snn_LSTM=Net_SLSTM_Extractor(inputSize=2,hiddenSize=50,numClasses=2).to(device)
readout=Classifier(hiddenSize=50,numClasses=2).to(device)

import copy

N_batch=10000
opt=optim.Adam(params=list(snn_LSTM.parameters())+list(readout.parameters()),lr=1e-3)
opt_1=optim.Adam(params=list(snn_LSTM.parameters())+list(readout.parameters()),lr=1e-2)

Loss=torch.nn.CrossEntropyLoss()

N_eval=10
N_adapt=100

Stat_tr=torch.zeros([N_batch,2])
Stat_te=torch.zeros([int(N_batch/N_eval),3,N_adapt])


ind_help=0
for n in range(N_batch):

    x_batch,y_batch=Data.Batch(10)

    z, mem1, mem2=snn_LSTM(x_batch)
    y=readout(z)

    loss=Loss(y,y_batch)

    _,pred=torch.max(y,1)
    _,y_B=torch.max(y_batch,1)
    acc=(pred==y_B).sum().item()/y_batch.size(0)
    
    loss.backward()
    opt.step()

    opt.zero_grad()
    opt_1.zero_grad()
    
    Stat_tr[n,0] = acc
    Stat_tr[n,1] = loss.detach().cpu()


    if n%N_eval==0 and n>0:
        
        ## FOR ADAPTATION, STORE MODEL'S PARAMETERS HERE
        #Parameter_before=deepcopy(list(...))
        
        snn_lstm_state_before = copy.deepcopy(snn_LSTM.state_dict())
        readout_state_before = copy.deepcopy(readout.state_dict())
        
        xDERBatch,yDERBatch,DERy=x_batch,y_batch,y.detach()
        ## ADAPT
        x_batch,y_batch=Data.Test()
        mem1,mem2=mem1.detach(),mem2.detach()
        alpha,beta=1,1.1
        for l in range(N_adapt):
            z, mem1_te, mem2_te=snn_LSTM(x_batch)
            y=readout(z)
            loss_te=Loss(y,y_batch)
            #loss=torch.pow(mem1_te.mean(0)-mem1.mean(0),2).mean()+torch.pow(mem2_te.mean(0)-mem2.mean(0),2).mean()
            #loss=torch.pow(mem1_te-mem1,2).mean()+torch.pow(mem2_te-mem2,2).mean()
            meanLoss=torch.pow(mem1_te.mean(0)-mem1.mean(0),2).mean()+torch.pow(mem2_te.mean(0)-mem2.mean(0),2).mean()
            std_loss=torch.pow(mem1_te.std(0)-mem1.std(0),2).mean()+torch.pow(mem2_te.std(0)-mem2.std(0),2).mean()
            statLoss = meanLoss + std_loss
            
            zDER, _,_=snn_LSTM(xDERBatch)
            yDER=readout(zDER)
            derLoss=torch.pow(yDER-DERy,2).mean()+nn.CrossEntropyLoss()(yDER,yDERBatch)
            loss=alpha*statLoss+beta*derLoss
            loss.backward()
            opt_1.step()
            
            opt_1.zero_grad()
            opt.zero_grad()
            
            _,pred=torch.max(y,1)
            _,y_B=torch.max(y_batch,1)
            acc_te=(pred==y_B).sum().item()/y_batch.size(0)

            Stat_te[ind_help,0,l]=acc_te
            Stat_te[ind_help,1,l]=loss_te.detach()
            Stat_te[ind_help,2,l]=loss.detach()


        snn_LSTM.load_state_dict(snn_lstm_state_before)
        readout.load_state_dict(readout_state_before)

        print('Training :', n, Stat_tr[n-N_eval:n,:].mean(0))
        print('Testing :', n, Stat_te[ind_help,0,:], Stat_te[ind_help,1,:], Stat_te[ind_help,2,:])
        
        
        ind_help+=1
   