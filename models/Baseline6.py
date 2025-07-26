import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader


class Person_Activity_Classifer(nn.Module):
    def  __init__(self , nums_classes):
      super(Person_Activity_Classifer, self).__init__()
      self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
      self.resnet50.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features=self.resnet50.fc.in_features, out_features=nums_classes)
      )
    def forward(self, x):
      return self.resnet50(x)
    
class Group_Activity_Classifer_lstm(nn.Module):
  def __init__(self ,person_feature_extraction, hidden_size, num_classes):
    super(Group_Activity_Classifer_lstm, self).__init__()
    
    self.feature_extraction = nn.Sequential(*list(person_feature_extraction.resnet50.children())[:-1])

    for param in self.feature_extraction.parameters():
        param.requires_grad = False
    
    self.pool = nn.AdaptiveMaxPool2d((1, 2048))  
    
    self.lstm = nn.LSTM(
        input_size=2048,
        hidden_size=hidden_size,
        batch_first=True
    ) 

    self.fc = nn.Sequential(
        nn.Linear(hidden_size, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes),
    )
    
  def forward(self, x):
    batch, bounding_box, seq_len, c, h, w = x.shape
    x = x.view(batch * bounding_box * seq_len, c, h, w)
    x= self.feature_extraction(x)
    x = x.view(batch*seq_len, bounding_box, -1) 
    x = self.pool(x)
    x = x.squeeze(dim=1) 
    x = x.view(batch, seq_len, -1) 
    x, (h, c) = self.lstm(x) 
    x = x [:, -1, :] 
    x = self.fc(x) 
    return x 