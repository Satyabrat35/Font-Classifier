import torch
import torchvision.models as models

def CustomResnet18(num_classes=10):
    """
    Custom ResNet18 model
    :param num_classes: 10
    :return: Modified ResNet18 model
    """
    model = models.resnet18()

    # Modify the FC layer
    features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(features, num_classes)
    )

    return model