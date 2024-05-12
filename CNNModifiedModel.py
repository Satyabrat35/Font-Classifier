import torch
import torch.nn as nn


class CNNModifiedModel(nn.Module):
    def __init__(self, num_classes=10):
        """
        Model definition
        :param num_classes: 10
        """
        # Inheritance
        super(CNNModifiedModel, self).__init__()

        # Sequential Layers
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Dropout Layer
        self.dropout = nn.Dropout(p=0.2)

        # Flatten Layer
        self.fl = nn.Flatten()

        # Linear Layer
        self.layer_fc = nn.Linear(128 * 18 * 31, num_classes)

    def forward(self, input):
        """
        Forward pass
        :param input:
        :return:
        """
        output = self.layer_1(input)
        output = self.layer_2(output)
        output = self.layer_3(output)
        output = self.layer_4(output)
        output = self.dropout(output)
        output = self.fl(output)
        final = self.layer_fc(output)
        return final