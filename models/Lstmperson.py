import torch.nn as nn
import torch
import torchvision.models as models

class LSTMPerson(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers):
        super(LSTMPerson , self).__init__()
        
        self.resnet50 = nn.Sequential(
            *list(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).children())[:-1]
        )
        self.lstm = nn.LSTM(
                            input_size=2048,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                        )
        
        self.fc =  nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        b, bb, seq, c, h, w = x.shape #
        x = x.view(b*bb*seq, c, h, w) 
        x = self.resnet50(x) 

        x = x.view(b*bb, seq, -1) 
        x, (h , c) = self.lstm(x) 
        x = x[:, -1, :] 
        x = self.fc(x) 
        return x