import numpy as np
import cv2
import rasterio
import matplotlib.pyplot as plt

def read_tiff(file_path):
    with rasterio.open(file_path) as src:
        image = src.read([1, 2, 3])  # Read RGB bands
        image = image.transpose((1, 2, 0))  # Reorder dimensions
    return image

def display_image(image, title="Image"):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def save_image(image, output_path):
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
