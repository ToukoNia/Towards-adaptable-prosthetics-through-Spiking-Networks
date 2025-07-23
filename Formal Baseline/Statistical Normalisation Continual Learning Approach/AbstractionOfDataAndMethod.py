# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 19:41:31 2025

@author: Nia Touko
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
import os
from scipy import io
import snntorch as snn
from snntorch import surrogate
from SLSTM_GMM_Model import Net_SLSTM_Extractor,Classifier
from SLSTM_GMM_Model import sliced_wasserstein_distance, GMM
import copy
import torch.nn.functional as F
class Data_Manager(nn.Module):
    
    def __init__(self, N_data=10, N_features=2, Time=5):
        super().__init__()

        self.X_tr=torch.rand([Time,N_data,N_features])
        self.Y_tr=torch.zeros([N_data,2])

        self.Y_tr[0:int(N_data/2),0]=1
        self.Y_tr[int(N_data/2):,1]=1

        self.X_te=self.X_tr*(2*torch.rand([1,N_data,N_features])-0.5)+torch.rand([1,N_data,N_features])
        self.Y_te=self.Y_tr

    def Batch(self,batch_size):

        rand_ind=torch.randint(0,self.X_tr.size()[1],[batch_size])

        x=self.X_tr[:,rand_ind,:]
        y=self.Y_tr[rand_ind,:]

        return x, y

    def Test(self):

        return self.X_te, self.Y_te
'''
class Data_Manager(nn.Module):
    def __init__(self, N_data=200, N_features=2, Time=5):
        super().__init__()
        
        # --- Create two distinct blobs of data ---
        # Class 0 centered around a positive mean
        mean_0 = torch.tensor([2.0] * N_features)
        cov_0 = torch.eye(N_features)
        class_0_data = torch.distributions.MultivariateNormal(mean_0, cov_0).sample((int(N_data/2), Time))
        
        # Class 1 centered around a negative mean
        mean_1 = torch.tensor([-2.0] * N_features)
        cov_1 = torch.eye(N_features)
        class_1_data = torch.distributions.MultivariateNormal(mean_1, cov_1).sample((int(N_data/2), Time))
        
        # Combine and permute to [Time, N_data, N_features]
        self.X_tr = torch.cat([class_0_data, class_1_data], dim=0).permute(1, 0, 2)

        # Create corresponding labels
        self.Y_tr = torch.zeros([N_data, 2])
        self.Y_tr[0:int(N_data/2), 0] = 1
        self.Y_tr[int(N_data/2):, 1] = 1

        # Create a "shifted" test set by adding some noise and a bias
        self.X_te = self.X_tr * (2 * torch.rand([1, N_data, N_features], device=self.X_tr.device) - 2) + (2 * torch.rand([1, N_data, N_features], device=self.X_tr.device) - 1)
        self.Y_te = self.Y_tr

    def Batch(self,batch_size):
        rand_ind=torch.randint(0,self.X_tr.size(1),[batch_size])
        x=self.X_tr[:,rand_ind,:]
        y=self.Y_tr[rand_ind,:]
        return x, y

    def Test(self):
        return self.X_te, self.Y_te
'''    
def swd(source_samples, target_samples, n_projections=50):
    """
    Calculates a differentiable Sliced-Wasserstein distance.
    Args:
        source_samples: Tensor of shape (n_samples, n_features) from the source
        target_samples: Tensor of shape (m_samples, n_features) from the target
        n_projections: The number of random projections to use
    """
    n_features = source_samples.size(1)
    
    # Generate random projections
    projections = torch.randn(n_features, n_projections, device=source_samples.device)
    projections = projections / torch.norm(projections, dim=0, keepdim=True) # Normalize

    # Project the samples
    source_proj = torch.matmul(source_samples, projections)
    target_proj = torch.matmul(target_samples, projections)

    # Sort the projected samples along the sample dimension
    source_proj_sorted, _ = torch.sort(source_proj, dim=0)
    target_proj_sorted, _ = torch.sort(target_proj, dim=0)

    # Calculate the L1 distance between the sorted projections
    # This is equivalent to the 1-Wasserstein Distance on the projected samples
    distance = torch.abs(source_proj_sorted - target_proj_sorted).mean()
    
    return distance
'''
def consolidated_loss(encTarg, gmm, classifier, lambdaSwd, batchSize):
    
    output = classifier(encTarg)
    softmax_out = F.softmax(output, dim=1)
    log_softmax_out = F.log_softmax(output, dim=1)
    
    loss_entropy = -torch.sum(softmax_out * log_softmax_out, dim=1).mean()

    pseudoSamples, _ = gmm.sample(batchSize)
    lossSwd = swd(encTarg, pseudoSamples)
    total = loss_entropy + lambdaSwd * lossSwd
    return total, loss_entropy, lossSwd
'''
def consolidated_loss(encTarg, gmm, classifier, lambdaSwd, batchSize):
    pseudoSamples,pseudoLabels = gmm.sample(batchSize)
    #Classifier Loss    
    classifierOut = classifier(pseudoSamples)
    lossClass = nn.CrossEntropyLoss()(classifierOut, pseudoLabels)
    lossClass=0
    #Distribution Allignment
    lossSwd = swd(encTarg, pseudoSamples)
    total = lossClass + lambdaSwd * lossSwd
    return total, lossClass, lossSwd

device = 'cuda' if torch.cuda.is_available() else 'cpu' 
Data=Data_Manager()
snn_LSTM=Net_SLSTM_Extractor(inputSize=2,hiddenSize=50,numClasses=2).to(device)
readout=Classifier(hiddenSize=50,numClasses=2).to(device)
N_batch=1000
opt=optim.Adam(params=list(snn_LSTM.parameters())+list(readout.parameters()),lr=1e-3)
Loss=torch.nn.CrossEntropyLoss()

N_eval=10
Stat_tr=torch.zeros([N_batch,2])
Stat_te=torch.zeros([int(N_batch/N_eval),2])
Stat_ad=torch.zeros([int(N_batch/N_eval),2])
feature_accumulator = []
label_accumulator = []

ind_help=0
for n in range(N_batch):
    snn_LSTM.train()
    readout.train()
    x_batch,y_batch=Data.Batch(10)
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    z=snn_LSTM(x_batch)
    y=readout(z)
    _,y_B=torch.max(y_batch,1)
    loss=Loss(y,y_B)

    _,pred=torch.max(y,1)
    
    acc=(pred==y_B).sum().item()/y_batch.size(0)
    
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(list(snn_LSTM.parameters()) + list(readout.parameters()), max_norm=1.0)
    opt.step()
    
    opt.zero_grad()
    
    Stat_tr[n,0]=acc*100
    Stat_tr[n,1]=loss.detach()
    feature_accumulator.append(z.detach()) 
    label_accumulator.append(y_B)
    
    if n%N_eval==0 and n>0:
        all_features = torch.cat(feature_accumulator)
        all_labels = torch.cat(label_accumulator)
        Parameter_before=copy.deepcopy(list(snn_LSTM.parameters()))
        classifier_params_before=copy.deepcopy(list(readout.parameters()))
        optimiser=torch.optim.Adam(params=list(snn_LSTM.parameters())+list(readout.parameters()),lr=1e-3)
        
        # Fit the GMM on all the accumulated data
        gmm = GMM(n_components=2, n_features=50, device=device)
        gmm.fit(all_features, all_labels)
        feature_accumulator.clear()
        label_accumulator.clear()
        snn_LSTM.eval()
        readout.eval()
        with torch.no_grad():

           ## ADAPT
           x_batch,y_batch=Data.Test()
           x_batch, y_batch = x_batch.to(device), y_batch.to(device)
           z=snn_LSTM(x_batch)
           y=readout(z)
           
           
           _,pred=torch.max(y,1)
           _,y_B=torch.max(y_batch,1)
           loss_te=Loss(y,y_B)
           acc_te=(pred==y_B).sum().item()/y_batch.size(0)*100

           Stat_te[ind_help,0]=acc_te
           Stat_te[ind_help,1]=loss_te
                      
        
        ## FOR ADAPTATION, STORE MODEL'S PARAMETERS HERE
        #Parameter_before=deepcopy(list(...))
       
        lambda_swd = 1
        snn_LSTM.train()
        readout.train()
        testLoss=0
        correctPredictions=0
        samples=0
        
        x_batch,y_batch=Data.Test()
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        z=snn_LSTM(x_batch)

        optimiser.zero_grad()
        latentFeatures = snn_LSTM(x_batch)
        loss, loss_clf, loss_swd = consolidated_loss(
            latentFeatures, gmm, readout, lambda_swd, y_batch.size(0)
        )
        loss.backward()
        optimiser.step()
        snn_LSTM.eval()
        readout.eval()
        with torch.no_grad():
            
            ## ADAPT
            x_batch,y_batch=Data.Test()
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            z=snn_LSTM(x_batch)
            y=readout(z)
            _, y_B = torch.max(y_batch, 1)
            loss_ad=Loss(y,y_B)
            
            _,pred=torch.max(y,1)
            acc_ad=(pred==y_B).sum().item()/y_batch.size(0)*100

            Stat_ad[ind_help,0]=acc_ad
            Stat_ad[ind_help,1]=loss_ad
            print(loss_swd)
            loss, loss_clf, loss_swd = consolidated_loss(
                latentFeatures, gmm, readout, lambda_swd, y_batch.size(0)
            )
            print(loss, loss_swd)
            print(n, Stat_tr[n-N_eval:n,:].mean(0), Stat_te[ind_help,:],Stat_ad[ind_help,:])
            
            ind_help+=1

            ## FOR ADAPTATION, RESET MODEL'S PARAMETERS TO THE ORIGINAL VALUE

        with torch.no_grad():
            Parameter_before = copy.deepcopy(snn_LSTM.state_dict())
            classifier_params_before = copy.deepcopy(readout.state_dict())
            
            # To Restore:
            snn_LSTM.load_state_dict(Parameter_before)
            readout.load_state_dict(classifier_params_before)
            
            #make sure parameters are being copied correctly. Addjust GMM fitting to be over the last x samples