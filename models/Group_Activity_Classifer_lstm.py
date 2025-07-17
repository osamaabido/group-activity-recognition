import os 
import sys
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
import  torchvision.models as models
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

class Group_Activity_Classifer_lstm(nn.Module):
    def __init__(self , person_feature_extraction , num_classes):
        super(Group_Activity_Classifer_lstm, self).__init__()
        self.resent50 = person_feature_extraction.resnet50
        self.lstm = person_feature_extraction.lstm
        
        for module in [self.resent50, self.lstm]:
            for param in module.parameters():
                param.requires_grad = False
        
        self.pool = nn.AdaptiveMaxPool2d((1, 2048))  
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes), 
        )
        
        def forward(self , x):
          batch_size , bounding_boxes ,seq_len , channels , height , width = x.shape
          x = x.view(batch_size * bounding_boxes*seq_len, channels, height, width)
          x1 = self.resent50(x)
          x1 = x1.view(batch_size *bounding_boxes , seq_len, -1)
          x2 , (h , c) = self.lstm(x1)
          x = torch.cat([x1, x2], dim=2) 
          x = x.contiguous()            
          x = x[:, -1, :]                
          x = x.view(batch_size, bounding_boxes, -1) 
          x = self.pool(x)  
          x = x.squeeze(dim=1) 

          x = self.fc(x) 
          return x