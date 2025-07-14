# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 19:15:54 2025

@author: Nia Touko
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_results(history,model):
    #Plots training and validation metrics and saves the figure.
    df = pd.DataFrame(history)
    
    plt.style.use('seaborn-v0_8-whitegrid')    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plotting Loss
    ax1.plot(df['train_loss'], label='Train Loss', color='blue')
    ax1.plot(df['intra_session_loss'], label='Intra-Session Test Loss', linestyle='--', color='green')
    ax1.plot(df['inter_session_loss'], label='Inter-Session Test Loss', linestyle='--', color='red')
    ax1.set_title(f'Mean of sessions- {model} Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plotting Accuracy
    ax2.plot(df['train_acc'], label='Train Accuracy', color='blue')
    ax2.plot(df['intra_session_acc'], label='Intra-Session Test Accuracy', linestyle='--', color='green')
    ax2.plot(df['inter_session_acc'], label='Inter-Session Test Accuracy', linestyle='--', color='red')
    ax2.set_title(f'Mean of sessions- {model} Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    # Save the figure
    plt.savefig(f"Overall_training_results_{model}.png")
    plt.show()
    
def fileFinder(dataDirectory):
    searchPattern = os.path.join(dataDirectory, '*.csv')
    matFilePaths = glob.glob(searchPattern)  #idk how good glob is but it made this way simpler so... ill revisit it later

    if not matFilePaths:
       raise ValueError("No .csv files found in this directory, please double check and try again :)")
        
    return matFilePaths
#For loop to iterate all the files in a folder (and load all the files in the folder). Maybe make you able to leave subjects out
#Store in dataframe, then I'll need to average each unit and then graph
# Note: might be worth adding either a seperate plot or a ghost trail for a without outliers (where if a subject avg is x outside the statistical mean its ignored)
dirs = [f for f in os.listdir() if os.path.isdir(f)]
print(dirs)
for direct in dirs:
    directory = direct+r'\CSV'
    print(directory)
    paths=fileFinder(directory)
    history = []
    for path in paths:
        df = pd.read_csv(path)
        history.append(df)
    
    dfCat=pd.concat(history)
    idx=dfCat.groupby(dfCat.Epoch)
    df=idx.mean()
    print(df) 
    plot_results(df,direct)
      