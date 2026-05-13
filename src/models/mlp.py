import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    Research-grade CNN Architecture.
    Tích hợp chuẩn: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
    Đảm bảo trích xuất đặc trưng tốt và chống overfitting bằng Dropout.
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        in_channels = params.get('input_channel', 3)
        
        # Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1)) # Ép về 1x1 chuẩn GAP

        self.relu = nn.ReLU(inplace=True)

        # Classifier
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

        self.to(self.device)

    def forward(self, x):
        x = x.float()

        # Feature Extractor
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))

        x = torch.flatten(x, 1)

        # Classifier
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class MLP(CNN):
    """
    Giữ lại class MLP kế thừa từ CNN để tương thích ngược với code cũ.
    Thực chất nó là CNN.
    """
    pass