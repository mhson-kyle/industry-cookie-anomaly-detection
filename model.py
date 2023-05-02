from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50

class ResNetDefectClassifier(nn.Module):
    def __init__(self, trainable=False):
        super(ResNetDefectClassifier, self).__init__()
        self.resnet = resnet50(weights='IMAGENET1K_V2')
        self.fc1 = nn.Linear(1000, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 4)
        for param in self.resnet.parameters():
            param.requires_grad = trainable
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class CNNDefectClassifier(nn.Module):
    def __init__(self):
        super(CNNDefectClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 32, 1, padding=0)
        
        self.fc1 = nn.Linear(6272, 2048)
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

