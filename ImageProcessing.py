import cv2
import torch

class ImageProcessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def processImage(self, image_path):
        original_img = cv2.imread(image_path)

        # Upscale or downscale based on dimension
        if original_img.shape[1] < self.width or original_img.shape[0] < self.height:
            resized_img = cv2.resize(original_img, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        else:
            resized_img = cv2.resize(original_img, (self.width, self.height), interpolation=cv2.INTER_AREA)

        return resized_img

    def transposeImage(self, image):
        # Transpose the image
        img_tensor = torch.tensor(image)
        img_tensor = img_tensor.permute(2, 0, 1)

        # All of my models expect input in batch format
        batch = 16
        batched_img = img_tensor.unsqueeze(0)

        return batched_img
