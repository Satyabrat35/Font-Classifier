import os
import cv2
import warnings
import json
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor, ViTImageProcessor


def main(label_mapping):
    ################################################################################
    # Download the ViT fine-tuned model and call the model
    ################################################################################
    model_path = '../model checkpoint/vit'
    vit_model = ViTForImageClassification.from_pretrained(model_path)

    ################################################################################
    # Load image and generate embeddings using ViTImageProcessor
    ################################################################################
    sample_image_path = '../sample.png'
    img = cv2.imread(sample_image_path)

    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    inputs = feature_extractor(images=img, return_tensors='pt')

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # vit_model.to(device)
    # inputs.to(device)

    with torch.no_grad():
        outputs = vit_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1)
        print(f'Font predicted using ViT-> {label_mapping[str(prediction.item())]}')


if __name__ == '__main__':
    with open('../label_mapping.json', 'r') as f:
        label_mapping = json.load(f)

    warnings.filterwarnings("ignore", category=FutureWarning)
    main(label_mapping)