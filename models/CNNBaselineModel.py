import torch
import torch.nn as nn

class CNNBaselineModel(nn.Module):
  def __init__(self, num_classes = 10):
    """
    Model definition
    :param num_classes:
    """
    # Inheritance
    super(CNNBaselineModel, self).__init__()

    # Sequential Layers
    self.layer_1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.layer_2 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    # Flatten Layer
    self.fl = nn.Flatten()

    # Linear Layer
    # self.fc = nn.Linear(32 * 75 * 125, num_classes)
    self.layer_fc = nn.Linear(16 * 75 * 125, num_classes)

  def forward(self, input):
    """
    Forward pass
    :param input:
    :return:
    """
    output = self.layer_1(input)
    output = self.layer_2(output)
    output = self.fl(output)
    final = self.layer_fc(output)
    return final