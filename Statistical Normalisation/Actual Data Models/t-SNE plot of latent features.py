# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 13:48:39 2025

@author: Nia Touko
"""
from sklearn.manifold import TSNE
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader,random_split
import numpy as np
import matplotlib.pyplot as plt
import snntorch as snn
from snntorch import surrogate
import os
import glob
from tqdm import tqdm
import scipy
from scipy import io
import umap
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
   

    
def plotTNSEUMAP(encoder, trainData, device, subjectID, sample_size=4000):
    encoder.eval()  
    dataLoader = DataLoader(trainData, batch_size=128, shuffle=False)
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(dataLoader, desc=f"Generating Latents for Subject {subjectID}"):
            x = x.permute(1, 0, 2).to(device)
            z, _, _ = encoder(x)
            all_latents.append(z.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_latents = np.concatenate(all_latents, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    num_points = all_latents.shape[0]
    if num_points > sample_size:
        print(f"Dataset has {num_points} points. Taking a random sample of {sample_size}.")
        random_indices = np.random.permutation(num_points)
        sample_indices = random_indices[:sample_size]
        
        latents_sample = all_latents[sample_indices]
        labels_sample = all_labels[sample_indices]
    else:
        print("Dataset is smaller than sample size, using all points.")
        latents_sample = all_latents
        labels_sample = all_labels

    '''
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=300)
    results = tsne.fit_transform(latents_sample)
    '''
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    results = reducer.fit_transform(latents_sample)
    
    print("Creating plot...")
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("hsv", len(np.unique(all_labels))) # Use all labels for consistent color mapping
    sns.scatterplot(
        x=results[:, 0],
        y=results[:, 1],
        hue=labels_sample,
        palette=palette,
        hue_order=range(len(np.unique(all_labels))), 
        legend="full",
        alpha=0.8
    )
    plt.title(f'UMAP Visualization of Latent Vectors for Subject {subjectID} (Sample of {len(labels_sample)})')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Gesture')
    
    # Save the figure
    plot_filename = f"subject_{subjectID}_umap_plot.png"
    plt.savefig(plot_filename)
    print(f"t-SNE plot saved as {plot_filename}")
    plt.show()

def loadDataset(mat_paths, windowSize=400, overlapR=0.5, numClasses=8,doEncode=0):
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


def fileFinder(dataDirectory):
    searchPattern = os.path.join(dataDirectory, '*.mat')
    matFilePaths = glob.glob(searchPattern)  #idk how good glob is but it made this way simpler so... ill revisit it later

    if not matFilePaths:
       raise ValueError("No .mat files found in this directory, please double check and try again :)")
        
    return matFilePaths

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
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
matFilePaths=[]
for subjectID in range(1,11):
    normaliseData=DataNormaliser()
    print("Processing subject",subjectID)
    matFilePaths=fileFinder(r'..\..\Data\DB6_s%s_a' % subjectID)+fileFinder(r'..\..\Data\DB6_s%s_b' % subjectID)
    dataPaths=matFilePaths[:7]
    trainData=loadDataset(dataPaths)
    trainData=normaliseData.forwardTrain(trainData)
    trainLoader=DataLoader(trainData, batch_size=128, shuffle=True)
     
    snn_LSTM = Net_SLSTM_Extractor(inputSize=14, hiddenSize=128, numClasses=8).to(device)
    PATH = f"Models/Subject_{subjectID}_SLSTM_TAB"
    
    # 2. Load the entire dictionary object
    checkpoint = torch.load(PATH)
    
    # 3. Load the state dictionaries into your model instances
    snn_LSTM.load_state_dict(checkpoint['encoder_state_dict'],strict=False)
    plotTNSEUMAP(snn_LSTM, trainData, device, subjectID)
    