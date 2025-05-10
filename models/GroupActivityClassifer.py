import torch.nn as nn
import albumentations as A
import torchvision.models as models
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchsummary import summary


class Group_Activity_Classifer(nn.Module):
    def __init__(self, num_classes):
        super(Group_Activity_Classifer, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=self.in_features, out_features=num_classes)
        )
    
    def forward(self, x):
        return self.resnet50(x)



    