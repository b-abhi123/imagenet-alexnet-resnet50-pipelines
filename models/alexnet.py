import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=11, stride=4, padding=2) #96x55x55
        self.bn1 = nn.BatchNorm2d(96) #can implement LRN here as well
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2) #96x27x27

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2) #256x27x27
        self.bn2 = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2) #256x13x13

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1) #384x13x13
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1) #384x13x13
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1) #256x13x13

        self.mp3 = nn.MaxPool2d(kernel_size=3, stride=2) #256x6x6
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,num_classes)

        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.mp1(self.relu(self.bn1(self.conv1(x))))
        x = self.mp2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        x = self.mp3(x)
        x = x.reshape(x.shape[0], -1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x



        



