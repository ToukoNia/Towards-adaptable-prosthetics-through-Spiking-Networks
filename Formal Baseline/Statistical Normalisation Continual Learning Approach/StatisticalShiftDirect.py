
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
class SourceStatisticsTracker(nn.Module):
    def __init__(self, feature_dim, momentum=0.01, device='cpu'):
        super().__init__()
        self.momentum = momentum
        self.feature_dim = feature_dim
        self.device = device
        
        self.register_buffer('running_mean', torch.zeros(feature_dim, device=device))
        self.register_buffer('running_cov', torch.eye(feature_dim, device=device))
        self.register_buffer('is_initialized', torch.tensor(False, device=device))

    def update(self, features):
        if not self.is_initialized:
            self.running_mean.data = torch.mean(features, dim=0)
            self.running_cov.data = torch.cov(features.T)
            self.is_initialized.data = torch.tensor(True)
        else:
            current_mean = torch.mean(features, dim=0)
            current_cov = torch.cov(features.T)
            
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * current_mean
            self.running_cov.data = (1 - self.momentum) * self.running_cov.data + self.momentum * current_cov
    
    def get_stats(self):
        return self.running_mean, self.running_cov

def consolidated_loss(target_features, source_stats_tracker, classifier, lambda_align):
    source_mean, source_cov = source_stats_tracker.get_stats()
    
    target_mean = torch.mean(target_features, dim=0)
    target_cov = torch.cov(target_features.T) + torch.eye(target_features.size(1), device=target_features.device) * 1e-6

    loss_align = F.mse_loss(target_mean, source_mean) + F.mse_loss(target_cov, source_cov)
    total_loss = lambda_align * loss_align
    
    return total_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_FEATURES = 2
HIDDEN_SIZE = 50
NUM_CLASSES = 2
BATCH_SIZE = 20
N_ITERATIONS = 1000
N_EVAL = 10
LAMBDA_ALIGN = 1 

Data = Data_Manager(N_features=INPUT_FEATURES)
snn_LSTM = Net_SLSTM_Extractor(inputSize=INPUT_FEATURES, hiddenSize=HIDDEN_SIZE).to(device)
readout = Classifier(hiddenSize=HIDDEN_SIZE, numClasses=NUM_CLASSES).to(device)
source_stats_tracker = SourceStatisticsTracker(feature_dim=HIDDEN_SIZE, device=device)

opt = optim.Adam(params=list(snn_LSTM.parameters()) + list(readout.parameters()), lr=1e-3)
Loss = nn.CrossEntropyLoss()

N_eval=10
Stat_tr=torch.zeros([N_ITERATIONS,2])
Stat_te=torch.zeros([int(N_ITERATIONS/N_eval),2])
Stat_ad=torch.zeros([int(N_ITERATIONS/N_eval),2])
Stat_tr_av=torch.zeros([int(N_ITERATIONS/N_eval),2])

print(f"--- Starting Training and Adaptation on {device} ---")
ind_help = 0
for n in range(N_ITERATIONS):
    snn_LSTM.train()
    readout.train()
    
    x_batch, y_batch = Data.Batch(BATCH_SIZE)
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    
    z = snn_LSTM(x_batch)
    y = readout(z)
    
    _, y_B = torch.max(y_batch, 1)
    loss = Loss(y, y_B)

    _, pred = torch.max(y, 1)
    acc = (pred == y_B).sum().item() / y_batch.size(0)
    
    loss.backward()
    opt.step()
    opt.zero_grad()
    
    source_stats_tracker.update(z.detach())
    
    Stat_tr[n, 0] = acc * 100
    Stat_tr[n, 1] = loss.item()

    if n > 0 and n % N_EVAL == 0:
        snn_LSTM.eval()
        readout.eval()
        
        with torch.no_grad():
            x_test, y_test = Data.Test()
            x_test, y_test = x_test.to(device), y_test.to(device)
            
            z_te = snn_LSTM(x_test)
            y_pred_te = readout(z_te)
            
            _, y_test_labels = torch.max(y_test, 1)
            loss_te = Loss(y_pred_te, y_test_labels)
            
            _, pred_te = torch.max(y_pred_te, 1)
            acc_te = (pred_te == y_test_labels).sum().item() / y_test.size(0) * 100

            Stat_te[ind_help, 0] = acc_te
            Stat_te[ind_help, 1] = loss_te.item()
        snn_lstm_state_before = copy.deepcopy(snn_LSTM.state_dict())
        readout_state_before = copy.deepcopy(readout.state_dict())

        adapt_optimizer = optim.Adam(params=list(snn_LSTM.parameters()) + list(readout.parameters()), lr=1e-4)
        
        snn_LSTM.train() 
        readout.train()

        x_adapt, y_adapt = Data.Test()
        x_adapt, y_adapt = x_adapt.to(device), y_adapt.to(device)

        adapt_optimizer.zero_grad()
        
        target_features = snn_LSTM(x_adapt)
        
        adapt_loss = consolidated_loss(target_features, source_stats_tracker, readout, LAMBDA_ALIGN)
        
        adapt_loss.backward()
        adapt_optimizer.step()

        snn_LSTM.eval()
        readout.eval()
        with torch.no_grad():
            z_ad = snn_LSTM(x_test)
            y_pred_ad = readout(z_ad)
            
            loss_ad = Loss(y_pred_ad, y_test_labels)
            
            _, pred_ad = torch.max(y_pred_ad, 1)
            acc_ad = (pred_ad == y_test_labels).sum().item() / y_test.size(0) * 100

            Stat_ad[ind_help, 0] = acc_ad
            Stat_ad[ind_help, 1] = loss_ad.item()

        snn_LSTM.load_state_dict(snn_lstm_state_before)
        readout.load_state_dict(readout_state_before)

        Stat_tr_av[ind_help, :] = Stat_tr[n - N_EVAL:n, :].mean(0)
        print(f"Iter: {n} | Avg Train Acc: {Stat_tr_av[ind_help, 0]:.2f}% | "
              f"Test Acc (Before): {Stat_te[ind_help, 0]:.2f}% | "
              f"Test Acc (After): {Stat_ad[ind_help, 0]:.2f}%")
        
        ind_help += 1

num_evals = ind_help
eval_batches = np.arange(N_EVAL, N_ITERATIONS + 1, N_EVAL)[:num_evals]

Stat_tr_av_rec = Stat_tr_av[:num_evals].cpu()
Stat_te_recorded = Stat_te[:num_evals].cpu()
Stat_ad_recorded = Stat_ad[:num_evals].cpu()

fig, ax = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
fig.suptitle("S-LSTM Performance with Statistical Alignment TTA", fontsize=16)

ax[0].set_title("Model Accuracy vs. Batch Number")
ax[0].plot(Stat_tr_av_rec[:, 0], label="Training Accuracy", alpha=0.7)
ax[0].plot(eval_batches, Stat_te_recorded[:, 0], label="Test Accuracy (Before TTA)", marker='o', linestyle='--',color='green')
ax[0].plot(eval_batches, Stat_ad_recorded[:, 0], label="Test Accuracy (After TTA)", marker='x', linestyle=':',color='red')
ax[0].set_xlabel("Batch Number")
ax[0].set_ylabel("Accuracy (%)")
ax[0].legend()
ax[0].grid(True)

ax[1].set_title("Model Loss vs. Batch Number")
ax[1].plot(eval_batches, Stat_tr_av_rec[:, 1], label="Avg. Training Loss", color='orange', marker='.')
ax[1].plot(eval_batches, Stat_te_recorded[:, 1], label="Test Loss (Before TTA)", color='green', marker='o', linestyle='--')
ax[1].plot(eval_batches, Stat_ad_recorded[:, 1], label="Test Loss (After TTA)", color='red', marker='x', linestyle=':')
ax[1].set_xlabel("Batch Number")
ax[1].set_ylabel("Loss")
ax[1].legend()
ax[1].grid(True)

ax[2].set_title("Loss Improvement from TTA")
loss_difference = Stat_te_recorded[:, 1] - Stat_ad_recorded[:, 1]
ax[2].plot(eval_batches, loss_difference, label="Loss Reduction (Before - After)", color='purple', marker='d')
ax[2].axhline(0, color='grey', linestyle='--', linewidth=1)
ax[2].set_xlabel("Batch Number")
ax[2].set_ylabel("Loss Reduction")
ax[2].legend()
ax[2].grid(True)

plt.tight_layout()
plt.show()