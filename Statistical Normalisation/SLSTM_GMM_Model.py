# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 16:43:51 2025

@author: Nia Touko
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

numChannels=14
class Net_SLSTM_Extractor(nn.Module):
    def __init__(self, inputSize=14, hiddenSize=128, numClasses=8):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.numClasses = numClasses
        self.slstm1 = snn.SLSTM(inputSize, hiddenSize, spike_grad=surrogate.fast_sigmoid(),learn_threshold=True,reset_mechanism="subtract")
        self.slstm2 = snn.SLSTM(hiddenSize, hiddenSize, spike_grad=surrogate.fast_sigmoid(),learn_threshold=True,reset_mechanism="subtract")
        self.bn1 = nn.BatchNorm1d(hiddenSize)
        self.inputBn=nn.BatchNorm1d(inputSize)

    def forward(self, x):  # x: [time, batch, features]
        syn1, mem1 = self.slstm1.init_slstm()
        syn2, mem2 = self.slstm2.init_slstm()
        mem2Rec = []
 
        for step in range(x.size(0)):
            spk1, syn1, mem1 = self.slstm1(x[step], syn1, mem1)
            spk1 = self.bn1(spk1) 
            spk2, syn2, mem2 = self.slstm2(spk1, syn2, mem2)
            mem2Rec.append(mem2)
        mem2Rec = torch.stack(mem2Rec)
        finalMem = mem2Rec.mean(dim=0)
        return finalMem 
    
class Classifier(nn.Module):
    def __init__(self, hiddenSize=128, numClasses=8):
        super().__init__()
        self.fc = nn.Linear(hiddenSize, numClasses)

    def forward(self, z):
        return self.fc(z)

from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily
import ot 

# GMM Utility Class
class GMM:
    def __init__(self, n_components, n_features, device):
        self.n_components = n_components
        self.n_features = n_features
        self.device = device
        self.mus = None
        self.covs = None
        self.alphas = None
        self.gmm_dist = None

    def fit(self, latent_vectors, labels):
        if labels.ndim > 1:
            _, labels = torch.max(labels, 1)
        unique_labels = torch.unique(labels)
        if self.n_components!= len(unique_labels):
            raise ValueError("Number of components must match number of labels. Please check this number :)")

        mus, covs, alphas =[],[],[]
        
        for j in sorted(unique_labels.cpu().numpy()):
            class_vectors = latent_vectors[labels == j]
            num_samples = len(class_vectors)

            # Estimate mixture weight, mean, and covariance [1]
            alpha = len(class_vectors) / len(latent_vectors)
            mu = torch.mean(class_vectors, dim=0)
            if num_samples < 2:
                    # If not enough samples to compute covariance, use an identity matrix
                    cov = torch.eye(self.n_features, device=self.device)
            else:
                # Otherwise, compute covariance normally
                cov = torch.cov(class_vectors.T) + torch.eye(self.n_features, device=self.device) * 1e-6
            alphas.append(torch.tensor(alpha, device=self.device))
            mus.append(mu)
            covs.append(cov)

        self.alphas = torch.stack(alphas)
        self.mus = torch.stack(mus)
        self.covs = torch.stack(covs)

        mix = Categorical(self.alphas)
        comp = MultivariateNormal(self.mus, self.covs)
        self.gmm_dist = MixtureSameFamily(mix, comp)

    def sample(self, num_samples):

        if self.gmm_dist is None:
            raise RuntimeError("GMM must be fitted before sampling.")
        component_indices = self.gmm_dist.mixture_distribution.sample((num_samples,))
        all_means = self.gmm_dist.component_distribution.loc
        all_covs = self.gmm_dist.component_distribution.covariance_matrix
        selected_means = all_means[component_indices]
        selected_covs = all_covs[component_indices]
        temp_dist = torch.distributions.MultivariateNormal(
        loc=selected_means, 
        covariance_matrix=selected_covs
        )
        samples = temp_dist.sample()
        
        return samples, component_indices

def sliced_wasserstein_distance(source_samples, target_samples, n_projections=500):
    source_np = source_samples.detach().cpu().numpy()
    target_np = target_samples.detach().cpu().numpy()
    
    swd = ot.sliced_wasserstein_distance(
        source_np,
        target_np,
        n_projections=n_projections
    )
    
    return torch.tensor(swd, device=source_samples.device, dtype=torch.float32)
