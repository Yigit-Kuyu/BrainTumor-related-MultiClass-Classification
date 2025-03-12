
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from os import listdir
import os
import numpy as np
from albumentations import Compose, Rotate, HorizontalFlip, VerticalFlip, ShiftScaleRotate
from tqdm import tqdm


 # To apply data augmentation on training data



# Offline augmentation
def augment_data(file_dir, n_generated_samples, save_to_dir):
    # Define augmentation pipeline
    augmentation = Compose([
        Rotate(limit=10, border_mode=cv2.BORDER_REPLICATE, p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.0, 
            rotate_limit=0, 
            border_mode=cv2.BORDER_REPLICATE, 
            p=0.5
        )
    ])

    # Create output directory if it doesn't exist
    os.makedirs(save_to_dir, exist_ok=True)

    # Process each image
    for filename in tqdm(os.listdir(file_dir)):
        image_path = os.path.join(file_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read {image_path}")
            continue

        # Convert BGR to RGB (optional, depends on your use case)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate n_generated_samples augmented images
        for i in range(n_generated_samples):
            augmented = augmentation(image=image_rgb)
            augmented_image = augmented["image"]

            # Convert back to BGR for saving (if needed)
            augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

            # Save the image
            output_path = os.path.join(
                save_to_dir, 
                f"aug_{i}_{filename}"
            )
            cv2.imwrite(output_path, augmented_image_bgr)


# path of cropped images
glioma_train= '/home/yck/Desktop/GITHUB/Bayesian Reinforcement Learning/MULTICLASS_CLASSIFICATION/multi-class-brain-tumor-classification/Train/glioma'
meningioma_train= '/home/yck/Desktop/GITHUB/Bayesian Reinforcement Learning/MULTICLASS_CLASSIFICATION/multi-class-brain-tumor-classification/Train/meningioma'
pituitary_train= '/home/yck/Desktop/GITHUB/Bayesian Reinforcement Learning/MULTICLASS_CLASSIFICATION/multi-class-brain-tumor-classification/Train/pituitary'

# Offline augmentation
augment_data(
    file_dir=glioma_train, # base dataset to be augmented from the given path
    n_generated_samples=1,
    save_to_dir=glioma_train #  generated dataset to save the path
)


augment_data(
    file_dir=meningioma_train, # base dataset to be augmented from the given path
    n_generated_samples=1,
    save_to_dir=meningioma_train #  generated dataset to save the path
)


augment_data(
    file_dir=pituitary_train, # base dataset to be augmented from the given path
    n_generated_samples=1,
    save_to_dir=pituitary_train #  generated dataset to save the path
)

print('stop')