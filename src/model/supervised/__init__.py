import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


# resnet50 for 2 classes
class ResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50, self).__init__()
        self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT)
        # self.resnet50.fc = nn.Linear(2048, num_classes)
        # drop out layer
        self.dropout = nn.Dropout(0.5)
        # fully connected layer
        self.resnet50.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            self.dropout,
            nn.Linear(512, num_classes)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet50(x)
        return x
    


class GlaucomaModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(GlaucomaModel, self).__init__()
        print("=> Initializing GlaucomaModel")

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 32 * 32, 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        return x