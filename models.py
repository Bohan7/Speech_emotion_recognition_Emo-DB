# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:53:20 2022

@author: Bohan
"""
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, inplane, plane, kernel_size, dropout_prob=0):
        super(Block, self).__init__()
        
        self.conv = nn.Conv1d(inplane, plane, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(plane)
        self.active = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
    
    def forward(self, x):
        out = self.active(self.bn(self.conv(x)))
        out = self.dropout(out)
        return out
        
        
        
class SpeechCNN(nn.Module):
    def __init__(self, inplane=1, kernel_size=5, planes=[256, 128, 64], input_dim=193, num_class=7, dropout_prob=0):
        super(SpeechCNN, self).__init__()
        
        self.block1 = Block(inplane, planes[0], kernel_size, dropout_prob=dropout_prob)
        self.block2 = Block(planes[0], planes[1], kernel_size, dropout_prob=dropout_prob)
        self.block3 = Block(planes[1], planes[2], kernel_size, dropout_prob=dropout_prob)
        
        self.linear = nn.Linear(planes[2]*input_dim, num_class)
    
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        
        
        
        
    