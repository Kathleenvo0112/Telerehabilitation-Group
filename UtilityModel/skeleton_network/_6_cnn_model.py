import torch
import torch.nn as nn
import torch.nn.functional as F

class SkeletonCNN(nn.Module):
    def __init__(self, num_classes=60, dropout_p=0.3):
        super().__init__()

        self.model = nn.Sequential(

            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 1)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 1)),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 1)),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_p),

            nn.AdaptiveAvgPool3d((1, 1, 1))  # -> [B, 256, 1, 1, 1]            
        )

        self.dropout_fc = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(256, num_classes)


    def forward(self, x):
        # x: [B, C, T, V, M]
        x = x.max(dim=-1)[0]  # -> [B, C, T, V]
        x = x.unsqueeze(-1)  # -> [B, C, T, V, 1]

        x = self.model(x)

        x = x.view(x.size(0), -1) # Flatten to 1D vector

        x = self.dropout_fc(x)
        x = self.fc(x)

        return x

    
# test
if __name__ == '__main__':
    model = SkeletonCNN(num_classes=60)
    x = torch.randn(32, 3, 300, 25, 2)
    out = model(x)
    print(out.shape)  # Should print [32, 60]