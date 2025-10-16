import torch
import torch.nn as nn
import torchvision.models as models


class Two_stage_Hierarchical(nn.Module):
    def __init__(self, num_classes_person, num_classes_group, hidden_size, num_layers):
        super(Two_stage_Hierarchical, self).__init__()

        self.resnet50 = nn.Sequential(
            *list(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).children())[:-1]
        )

        # stage 1 normalization (person-level)
        self.norm_feat = nn.LayerNorm(2048)

        # stage 2 normalization (group-level)
        self.norm_group = nn.LayerNorm(4096)

        # LSTM stage 1 (person-level)
        self.lstm_person = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5
        )

        # LSTM stage 2 (group-level)
        self.lstm_group = nn.LSTM(
            input_size=4096,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5
        )

        # Person-level classifier
        self.fc_person = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes_person)
        )

        self.pool = nn.AdaptiveMaxPool2d((1, 2048))

        # Group-level classifier
        self.fc_group = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes_group)
        )

    def forward(self, x):
        batch, bounding_box, seq_len, c, h, w = x.shape
        x = x.view(batch * bounding_box * seq_len, c, h, w)

        # Stage 1: extract features per person
        x1 = self.resnet50(x)
        x1 = x1.view(batch * bounding_box, seq_len, -1)
        x1 = self.norm_feat(x1)

        x2, _ = self.lstm_person(x1)
        x_person = self.fc_person(x2[:, -1, :])

        # Stage 2: group representation
        x = torch.cat((x1, x2), dim=2).contiguous()
        x = x.view(batch * seq_len, bounding_box, -1)
        first_team = x[:, :6, :]
        second_team = x[:, 6:, :]

        first_team = self.pool(first_team)
        second_team = self.pool(second_team)

        x = torch.cat((first_team, second_team), dim=2).contiguous()
        x = x.view(batch, seq_len, -1)
        x = self.norm_group(x)
        x, _ = self.lstm_group(x)
        x_group = self.fc_group(x[:, -1, :])

        return x_person, x_group
