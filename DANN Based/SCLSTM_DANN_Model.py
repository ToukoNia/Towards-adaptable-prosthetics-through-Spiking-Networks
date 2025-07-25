# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 22:51:23 2025

@author: Nia Touko
"""


import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        # Reverse the gradient and scale it by alpha
        return grads.neg() * ctx.alpha, None

class S_CLSTM_DANN(nn.Module):
    def __init__(self, input_features=14, hiddenSize=128, numClasses=8, numSubjects=10):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.lif1 = snn.Leaky(beta=0.9, learn_threshold=True)
        self.slstm1 = snn.SLSTM(input_size=64, hidden_size=hiddenSize, spike_grad=surrogate.fast_sigmoid(), learn_threshold=True)
        self.slstm2 = snn.SLSTM(input_size=hiddenSize, hidden_size=hiddenSize, spike_grad=surrogate.fast_sigmoid(), learn_threshold=True)
        self.fc_gesture = nn.Linear(hiddenSize, numClasses)

        self.fc_domain_1 = nn.Linear(hiddenSize, 64)
        self.lif_domain = snn.Leaky(beta=0.9, learn_threshold=True)
        self.fc_domain_2 = nn.Linear(64, numSubjects)

    def forward(self, x, alpha=1.0): # x shape: [time, batch, features]
        syn1, mem1 = self.slstm1.init_slstm()
        syn2, mem2 = self.slstm2.init_slstm()
        self.lif1.reset_mem()
        self.lif_domain.reset_mem()
        mem2_rec = []
        x_conv = x.permute(1, 2, 0)
        conv_out = self.conv1(x_conv)
        bn_out = self.bn1(conv_out)
        spk_conv, _ = self.lif1(bn_out)
        slstm_input = spk_conv.permute(2, 0, 1)
        for step in range(slstm_input.size(0)):
            spk1, syn1, mem1 = self.slstm1(slstm_input[step], syn1, mem1)
            spk2, syn2, mem2 = self.slstm2(spk1, syn2, mem2)
            mem2_rec.append(mem2)
        mem2_rec = torch.stack(mem2_rec)
        features = torch.mean(mem2_rec, dim=0)
        
        gesture_output = self.fc_gesture(features)
        
        # 2. Domain prediction with Gradient Reversal
        reversed_features = GradientReversal.apply(features, alpha)
        domain_hidden, _ = self.lif_domain(self.fc_domain_1(reversed_features))
        domain_output = self.fc_domain_2(domain_hidden)
        
        return gesture_output, domain_output