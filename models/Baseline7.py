import os 
import torch 
import torch.nn as nn
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

class Person_Activity_Temporal(nn.Module):
    def __init__(self , num_classes , hidden_size , num_layers):
        super(Person_Activity_Temporal, self).__init__()

        self.resnet50 = nn.Sequential(
            *list(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).children())[:-1]
        )

        self.norm = nn.LayerNorm(2048)

        self.lstm = nn.LSTM(
            input_size = 2048,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch , bounding_box, seq_len, c, h, w = x.shape
        x = x.view(batch * bounding_box * seq_len, c, h, w)
        x = self.resnet50(x)
        x = x.view(batch * bounding_box, seq_len, -1)
        x = self.norm(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1])
        return x
    

class Group_Activity_Temporal(nn.Module):
    def __init__(self , person_feature_extraction, hidden_size, num_classes , num_layers):
        super(Group_Activity_Temporal, self).__init__()

        self.resnet50 = person_feature_extraction.resnet50
        self.frist_lstm = person_feature_extraction.lstm

        for m in [self.resnet50, self.frist_lstm]:
            for param in m.parameters():
                param.requires_grad = False
        
        self.pool = nn.AdaptiveMaxPool2d((1, 2048))
        self.norm = nn.LayerNorm(2048)

        self.second_lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
         
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
         batch, bounding_box, seq_len, c, h, w = x.shape
         x = x.view(batch * bounding_box * seq_len, c, h, w)
         x1 = self.resnet50(x)

         x1 = x.view(batch*bounding_box, seq_len, -1)
         x1 = self.norm(x1)
         x2, _ = self.frist_lstm(x1)
         x= torch.concat([x1 , x2] , dim=2).contiguous()

         x = x.view(batch *seq_len, bounding_box, -1)
         x = self.pool(x)

         x = x.view(batch, seq_len, -1)
         x = self.norm(x)

         x , _ = self.second_lstm(x)
         x = self.fc(x[:, -1])
         return x
    

"""class Person_Activity_Temporal(nn.Module):
    def __init__(self , num_classes , hidden_size , num_layers):
        super().__init__()
        self.resnet50 = nn.Sequential(
            *list(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).children())[:-1]
        )
        self.norm = nn.LayerNorm(2048)
        self.lstm = nn.LSTM(2048, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch , bounding_box, seq_len, c, h, w = x.shape
        x = x.view(batch * bounding_box * seq_len, c, h, w)
        x = self.resnet50(x).squeeze(-1).squeeze(-1)
        x = x.view(batch, bounding_box, seq_len, -1)
        x = self.norm(x)
        x, _ = self.lstm(x.view(batch*bounding_box, seq_len, -1))
        x = self.fc(x[:, -1])
        return x


class Group_Activity_Temporal(nn.Module):
    def __init__(self , person_feature_extraction, hidden_size, num_classes , num_layers):
        super().__init__()
        self.resnet50 = person_feature_extraction.resnet50
        self.frist_lstm = person_feature_extraction.lstm
        for m in [self.resnet50, self.frist_lstm]:
            for param in m.parameters():
                param.requires_grad = False
        self.norm = nn.LayerNorm(2048)
        self.second_lstm = nn.LSTM(2048, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch, bounding_box, seq_len, c, h, w = x.shape
        x = x.view(batch * bounding_box * seq_len, c, h, w)
        x1 = self.resnet50(x).squeeze(-1).squeeze(-1)
        x1 = x1.view(batch*bounding_box, seq_len, -1)
        x1 = self.norm(x1)
        x2, _ = self.frist_lstm(x1)
        x = torch.cat([x1 , x2] , dim=2)
        x = x.view(batch, bounding_box, seq_len, -1)
        x, _ = torch.max(x, dim=1)   # (batch, seq_len, feature_dim)
        x = self.norm(x)
        x , _ = self.second_lstm(x)
        x = self.fc(x[:, -1])
        return x
"""