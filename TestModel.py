import os
import torch
import torch
import torchvision.models as models
import json
import matplotlib.pyplot as plt
from CNNBaselineModel import CNNBaselineModel
from CNNModifiedModel import CNNModifiedModel
from ImageProcessing import ImageProcessor

def main(label_mapping):

    checkpoint_dir = 'model checkpoint'

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

    # Resnet model
    model = models.resnet18()
    features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(features, 10)
    )

    ################################################################################
    # Load model weights
    ################################################################################
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))

    ################################################################################
    # Process image
    ################################################################################
    sample_image_path = 'sample.png'
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
    with open('label_mapping.json', 'r') as f:
        label_mapping = json.load(f)

    main(label_mapping)