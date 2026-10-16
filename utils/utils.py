import os
import cv2
import numpy as np

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP')

def is_image_file(filename):
    return filename.endswith(IMG_EXTENSIONS)

def srgb_to_linear(img):
    img = np.clip(img, 0.0, 1.0)
    return np.where(img <= 0.04045, img / 12.92, np.power((img + 0.055) / 1.055, 2.4)).astype(np.float32)

def linear_to_srgb(img):
    img = np.clip(img, 0.0, 1.0)
    return np.where(img <= 0.0031308, img * 12.92, 1.055 * np.power(img, 1.0 / 2.4) - 0.055).astype(np.float32)