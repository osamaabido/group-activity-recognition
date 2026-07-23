import torch.nn as nn
import torch
import torchvision.models as models

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