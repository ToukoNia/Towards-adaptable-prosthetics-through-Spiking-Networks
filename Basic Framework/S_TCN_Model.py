# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 14:21:36 2025

@author: Nia Touko
"""
import torch
from torch.nn.utils.parametrizations import weight_norm 
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


   
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
    def __init__(self,nInputs,nStages,nGestures,kernelSize=1.5):
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
        batch_size = rateCode.size(0)
        rateCode_flat = rateCode.view(batch_size, -1)
        out=self.fc(rateCode_flat)
        return out
    def reset(self):
       for layer in self.modules():
           if hasattr(layer, 'reset_mem'):
               layer.reset_mem()