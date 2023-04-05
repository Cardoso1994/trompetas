#!/usr/bin/env python3

from torch import nn

class segmentation_cnn(nn.Module):
    def __init__(self, num_classes):
        super(segmentation_cnn, self).__init__()
        self.fc1 = nn.Linear(2048, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
