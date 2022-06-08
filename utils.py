# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:53:20 2022

@author: Bohan
"""
import os
from os import listdir
from os.path import isfile, join
from scipy.io import wavfile
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split

from models import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from time import strftime, time
import sys

from sklearn.model_selection import KFold



###########################Functions for data preparation###########################################
# dictionary for labels
label_map = {"F": 0, "W": 1, "L": 2, "E": 3, "A": 4, "T": 5, "N": 6}

def FileNameLabel(files):
    labels = []
    for file in files:
        letter = file[5]
        if letter not in label_map.keys():
            raise ValueError('The label of {} is not found'.format(file))
        
        labels.append(label_map[letter])
    return labels

def zero_padding(dataset, max_length):
    pad_dataset = np.zeros((len(dataset), max_length))
    for i, signal in enumerate(dataset):

        if signal.shape[0] > max_length:
            signal = signal[:max_length]
        else:
            signal = np.pad(signal, (0, max_length-signal.shape[0]), 'constant', constant_values=0)
        pad_dataset[i] = signal

    return pad_dataset    

def augmentation(X_train, y_train, n_aug=10):
    aug_X_train = []
    aug_y_train = []
    sr = 16000

    print('Start augmentation')
    for i, signal in enumerate(X_train):
        for j in range(n_aug):
            aug_signal = librosa.effects.time_stretch(signal, rate=np.random.uniform(0.3, 2.0, 1)[0])
            aug_signal = librosa.effects.pitch_shift(aug_signal, sr, n_steps=np.random.uniform(-2, 2, 1)[0])

            aug_X_train.append(aug_signal)
            aug_y_train.append(y_train[i])        
    print('End augmentation')

    return aug_X_train, aug_y_train

###########################Function to extract features###########################################
def extract_feature(signals, sr=16000):
    output = []

    for signal in signals:
        stft = np.abs(librosa.stft(signal))
        # features
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
        mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(signal, sr=sr).T,axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=10000).T,axis=0)

        # merge features
        features = np.hstack((chroma, mfcc, mel, contrast, tonnetz))

        # merge features of signals
        output.append(features)
    
    return np.vstack(output)

###########################Functions to train and test models###########################################
def train(train_loader, net, optimizer, criterion, epoch, device):
    net.train()

    train_loss, correct, total = 0, 0, 0
    start_time = time()
    for idx, data_dict in enumerate(train_loader):
        inputs, label = data_dict[0].to(device, dtype=torch.float), data_dict[1].to(device)
        
        optimizer.zero_grad()
        pred = net(inputs)
        loss = criterion(pred, label)
        assert not torch.isnan(loss), 'NaN loss'
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        _, predicted  = torch.max(pred.data, 1)
        total += label.size(0)
        correct += predicted.eq(label).cpu().sum()

        if idx % 10 == 0:
            diff_time = time() - start_time
            acc = float(correct) / total
            m2 = ('Time: {:.04f}, Epoch: {}, Epoch iters: {} / {}\t'
                  'Loss: {:.04f}, Acc: {:.06f}')
            print(m2.format(diff_time, epoch, idx, len(train_loader),
                            float(train_loss), acc))
    
    train_loss = train_loss / total
    train_acc = correct / total
    train_time = time() - start_time

    state = {
          'epoch': epoch,
          'train_time': train_time,
          'train_loss': train_loss,
          'train_acc': train_acc,
        }
    
    return net, state

def test(test_loader, net, criterion, epoch, device): 
    net.eval()
    test_loss, correct, total = 0, 0, 0

    for (idx, data) in enumerate(test_loader):
        sys.stdout.write('\r [%d/%d]' % (idx + 1, len(test_loader)))
        sys.stdout.flush()

        inputs = data[0].to(device, dtype=torch.float)
        label = data[1].to(device)

        with torch.no_grad():
            pred = net(inputs)
            loss = criterion(pred, label)

        _, predicted = pred.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
        test_loss += loss.item()

    
    test_loss = test_loss / total
    test_acc = correct / total
        
    state = {
          'epoch': epoch,
          'test_loss': test_loss,
          'test_acc': test_acc,
        }
    
    return test_acc, state          

def cross_validation(trainset, lr_list=[1e-1, 1e-2, 1e-3], wd_list=[1e-3, 5e-3, 1e-4], gamma_list=[0.9, 0.85, 0.95]):
    kf = KFold(n_splits=5, shuffle=True)
    batch_size = 64
    # number of epochs
    nb_epochs = 50

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        print("No GPU :(")
        device = 'cpu'

    best_mean_acc, best_acc = 0, 0
    result = {}
    
    for lr in lr_list:
        for wd in wd_list:
            for gamma in gamma_list:
                print('\nCross validation for lr:{} weight decay:{} gamma:{}'.format(lr, wd, gamma))
                cross_validation_result = []
                for fold,(train_idx,test_idx) in enumerate(kf.split(trainset)):
                    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
                    test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

                    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_subsampler)
                    testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=test_subsampler)

                    # loss function
                    loss_fn = nn.CrossEntropyLoss().to(device)

                    # model
                    net = SpeechCNN().to(device)

                    # optimizer
                    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

                    # scheduler
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

                    for epoch in range(1, nb_epochs + 1):
                        net, trainstate = train(trainloader, net, optimizer, loss_fn, epoch, device)

                        # # testing mode to evaluate accuracy. 
                        acc, teststate = test(testloader, net, loss_fn, epoch, device=device)

                        if acc > best_acc:
                            best_acc = acc
                            best_epoch = epoch

                        msg = 'Epoch:{}.\tAcc: {:.03f}.\t Best_Acc:{:.03f} (epoch: {}).'
                        print(msg.format(epoch,  acc, best_acc, best_epoch))

                        scheduler.step()
                    cross_validation_result.append(best_acc)
                mean_acc = np.array(cross_validation_result).mean()

                if mean_acc > best_mean_acc:
                    best_mean_acc = mean_acc
                    result['lr'] = lr
                    result['wd'] = wd
                    result['gamma'] = gamma
                    result['score'] = mean_acc
    print('Cross validation ends!')
    print('The best hyperparameters:')
    print(result)

    return result
 
###########################Functions to visualize###########################################
def visualize(train_stats, test_stats, model_name='CNN1d'):
    current_path = os.getcwd()
    
    train_acc = [train_stats[i]['train_acc'] for i in range(len(train_stats))]
    test_acc = [test_stats[i]['test_acc'] for i in range(len(test_stats))]
    train_loss = [train_stats[i]['train_loss'] for i in range(len(train_stats))]
    test_loss = [test_stats[i]['test_loss'] for i in range(len(test_stats))]  
    best_test_acc = max(test_acc)  
    epochs = np.arange(1, len(train_stats)+1)
    
    print('The best test accuracy is {:.2f}'.format(best_test_acc))
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, color='blue', linestyle='dashed', label='Train Loss')
    plt.plot(epochs, test_loss, color='green', linestyle='dashed', label='Test Loss')
    plt.legend(fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Loss', fontsize=20)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, color='blue', linestyle='solid', label='Train Accuracy')
    plt.plot(epochs, test_acc, color='green', linestyle='solid', label='Test Accuracy')
    plt.legend(fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.title('Accuracy', fontsize=20)
    plt.savefig(join(current_path, 'loss_accuracy.jpg'))

       
        
        
        
    