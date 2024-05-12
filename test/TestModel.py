import os
import torch
import torch
import torchvision.models as models
import json
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.CNNBaselineModel import CNNBaselineModel
from models.CNNModifiedModel import CNNModifiedModel
from models.ImageProcessing import ImageProcessor
from models.CustomResnet18 import CustomResnet18

def main(label_mapping):

    checkpoint_dir = '../model checkpoint'

    ################################################################################
    # Use the appropriate model's ckpt file
    ################################################################################

    # checkpoint_file = 'cnnbaseline.ckpt'
    # checkpoint_file = 'cnnmodified.ckpt'
    checkpoint_file = 'resnet.ckpt'

    ckpt_path = os.path.join(checkpoint_dir, checkpoint_file)

    ################################################################################
    # Uncomment the model that you wish to test your data against
    ################################################################################
    # model = CNNBaselineModel()
    # model = CNNModifiedModel()
    model = CustomResnet18()


    ################################################################################
    # Load model weights
    ################################################################################
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))

    ################################################################################
    # Process image
    ################################################################################
    sample_image_path = '../sample.png'
    image = ImageProcessor(width=500, height=300)
    img = image.processImage(sample_image_path)
    batch_img = image.transposeImage(img)
    batch_img = batch_img.to(torch.float32)

    model.eval()

    with torch.no_grad():
        output = model(batch_img)
        _, prediction = torch.max(output.data, 1)
        print(f'Font predicted -> {label_mapping[str(prediction.item())]}')


if __name__ == "__main__":
    with open('../label_mapping.json', 'r') as f:
        label_mapping = json.load(f)

    main(label_mapping)