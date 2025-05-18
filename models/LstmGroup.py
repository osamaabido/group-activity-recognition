import torch.nn as nn
import torch
import torchvision.models as models
class LSTMGroup(nn.Module):
    def __init__(self , input_size , hidden_size , num_layers , num_classes):
        super(LSTMGroup, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
       
        self.feature_extraction = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        self.fc = nn.Sequential(
            nn.Linear(input_size + hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        def forward(self, x):
            batch, seq, c, h, w = x.shape
            x1 = x.view(batch * seq, c, h, w)
            x1 = self.feature_extraction(x1)
            x1 = x1.view(batch, seq, -1) 
            x2, (h, c) = self.lstm(x1)  
        
            x = torch.cat([x1, x2], dim=2)
                                        
            x = x[:, -1, :]  
            x = self.fc(x)  
            
            return x 
