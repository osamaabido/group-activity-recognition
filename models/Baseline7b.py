import os 
import torch 
import torch.nn as nn
import torchvision.models as models

class Group_Activity_Temporal(nn.Module):
    def __init__(self , person_feature_extraction, hidden_size, num_classes , num_layers):
        super(Group_Activity_Temporal, self).__init__()
        self.resnrt50 = person_feature_extraction.resnet50
        self.frist_lstm = person_feature_extraction.lstm

        for module in [self.resnrt50, self.frist_lstm]:
            for param in module.parameters():
                param.requires_grad = False
        
        self.pool = nn.AdaptiveMaxPool2d((1, 2048))

        self.norm = nn.LayerNorm(2048)
        
        self.second_lstm = nn.LSTM(
            input_size = 2048,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )


    def forward(self, x):
        batch , bounding_box, seq_len, c, h, w = x.shape
        x= x.view(batch * bounding_box * seq_len, c, h, w)
        x1 = self.resnrt50(x)
        x1 = x1.view(batch * bounding_box, seq_len, -1)

        x1 = self.norm(x1)
        x2, _ = self.frist_lstm(x1)
        x = torch.cat((x1, x2), dim=2).contiguous()

        x = x.view(batch * seq_len, bounding_box, -1)

        x = self.pool(x)
        x = x.view(batch, seq_len, -1)
        x = self.norm(x)
        x, _ = self.second_lstm(x)
        x = self.fc(x[:, -1])
        return x
    