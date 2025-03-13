import numpy as np 
import cv2
from os import listdir
import shutil
import itertools
import imutils
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import splitfolders
import os





# To crop jpg images based on Canny edge detection algorihm on train and test data



# Edge detection with Canny Algorithm
def crop_brain_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges
    edges = cv2.Canny(blurred, 30, 100)
    
    # Dilate edges to close gaps
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    
    x, y, w, h = cv2.boundingRect(c)
    return image[y:y+h, x:x+w]




def save_new_images(x_set, y_set, folder_name):
    """
    x_set: List of images (NumPy arrays)
    y_set: List of corresponding labels (strings)
    folder_name: Folder in which subfolders for each class exist or will be created
    """

    # Create subdirectories if they don't exist
    glioma_path = os.path.join(folder_name, 'glioma')
    meningioma_path = os.path.join(folder_name, 'meningioma')
    pituitary_path = os.path.join(folder_name, 'pituitary')

    os.makedirs(glioma_path, exist_ok=True)
    os.makedirs(meningioma_path, exist_ok=True)
    os.makedirs(pituitary_path, exist_ok=True)

    for i, (img, imclass) in enumerate(zip(x_set, y_set)):
        # Skip if image is None
        if img is None:
            print(f"[WARNING] Skipping index {i} because image is None.")
            continue

        if imclass == 'glioma':
            out_path = os.path.join(glioma_path, f"{i}.png")
        elif imclass == 'meningioma':
            out_path = os.path.join(meningioma_path, f"{i}.png")
        else:
            out_path = os.path.join(pituitary_path, f"{i}.png")

        success = cv2.imwrite(out_path, img)
        if not success:
            print(f"[ERROR] Failed to save image {i} -> {out_path}")



path_to_splitdataset="/Test_Val_Data/"


labels = ['glioma','meningioma','pituitary']
img_size=240

X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []

print("For Train Set")
for i in labels:
    folderPath = os.path.join(path_to_splitdataset,'train',i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = crop_brain_canny(img)
        img = cv2.resize(img, (img_size, img_size))
        X_train.append(img)
        y_train.append(i)
                
X_train = np.array(X_train)
y_train = np.array(y_train)

print("For Validation Set")
for i in labels:
    folderPath = os.path.join(path_to_splitdataset,'val',i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = crop_brain_canny(img)
        img = cv2.resize(img, (img_size, img_size))
        X_val.append(img)
        y_val.append(i)

X_val = np.array(X_val)
y_val = np.array(y_val)

print("For Test Set")
for i in labels:
    folderPath = os.path.join(path_to_splitdataset,'test',i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = crop_brain_canny(img)
        img = cv2.resize(img, (img_size, img_size))
        X_test.append(img)
        y_test.append(i)
                
X_test = np.array(X_test)
y_test = np.array(y_test)

# path to save cropped image
base_train_folder="/Train/"
base_test_folder='/Test/'
base_val_folder='/Val/'
save_new_images(X_train, y_train, folder_name=base_train_folder)
save_new_images(X_val, y_val, folder_name=base_val_folder)
save_new_images(X_test, y_test, folder_name=base_test_folder)


