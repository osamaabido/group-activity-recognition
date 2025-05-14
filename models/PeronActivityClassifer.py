import torch.nn as nn

class ClassiferNN(nn.Module):
    def __init__(self, person_feature_extraction, num_classes):
        super(ClassiferNN, self).__init__()
        self.feature_extraction = nn.Sequential(*list(person_feature_extraction.resnet50.children())[:-2])

        for param in self.feature_extraction.parameters():
            param.requires_grad = False
        
        self.pool = nn.AdaptiveMaxPool2d((1, 1))  
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x):
        batch, bounding_box, c, h, w = x.shape 
        x = x.view(batch*bounding_box, c, h, w) 
        x = self.feature_extraction(x) 
        x = self.pool(x) 
        x = x.view(batch, bounding_box, -1) 
        x = x.mean(dim=1) 
        x = self.fc(x) 
        return x