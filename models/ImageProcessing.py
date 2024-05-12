import cv2
import torch

class ImageProcessor:
    def __init__(self, width, height):
        """
        Define width and height of image
        :param width: 300
        :param height: 500
        """
        self.width = width
        self.height = height

    def processImage(self, image_path):
        """
        Resize the image to width and height, either upscale or downscale
        :param image_path:
        :return: resized image
        """
        original_img = cv2.imread(image_path)

        # Upscale or downscale based on dimension
        if original_img.shape[1] < self.width or original_img.shape[0] < self.height:
            resized_img = cv2.resize(original_img, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        else:
            resized_img = cv2.resize(original_img, (self.width, self.height), interpolation=cv2.INTER_AREA)

        return resized_img

    def transposeImage(self, image):
        """
        Transpose the image from WxHxC to CxWxH, also increase one dimnesion for batch input
        :param image:
        :return: transposed image
        """
        img_tensor = torch.tensor(image)
        img_tensor = img_tensor.permute(2, 0, 1)

        # All of the models expect input in batch format
        batch = 16
        batched_img = img_tensor.unsqueeze(0)

        return batched_img
