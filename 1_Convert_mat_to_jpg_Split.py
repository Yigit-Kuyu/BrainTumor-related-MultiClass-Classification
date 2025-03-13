import os
import h5py
import numpy as np
from PIL import Image


# To convert mat file-based dataset into jpg, and splting jpg-based dataset into train/val/test data.


# Define paths
input_folder = "/Dataset"
output_folder = "/Processed_Dataset"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all .mat files in the input folder
file_list = [f for f in os.listdir(input_folder) if f.endswith('.mat')]

# Process each .mat file
for i, file_name in enumerate(file_list, 1):
    file_path = os.path.join(input_folder, file_name)
    
    # Load .mat file with h5py
    with h5py.File(file_path, 'r') as f:
        # Access the 'cjdata' group (struct in MATLAB)
        cjdata = f['cjdata']
        
        # Extract image data (referenced dataset)
        im1 = np.array(cjdata['image']).astype(float)  # Convert to numpy array
        min1 = np.min(im1)
        max1 = np.max(im1)
        
        # Normalize to 0-255 and convert to uint8
        if max1 != min1:  # Avoid division by zero
            im = ((im1 - min1) * 255 / (max1 - min1)).astype(np.uint8)
        else:
            im = np.zeros_like(im1, dtype=np.uint8)  # If max=min, output a black image
        
        # Extract label (assuming it's a scalar)
        label = np.array(cjdata['label']).item()  # Convert to scalar
        
        # Map label to class name (adjust if your labels differ)
        label_map = {1: 'glioma', 2: 'meningioma', 3: 'pituitary'}
        class_name = label_map.get(label, str(label))  # Fallback to str(label) if not in map
        
        # Create class folder if it doesn't exist
        label_folder = os.path.join(output_folder, class_name)
        if not os.path.exists(label_folder):
            os.makedirs(label_folder, exist_ok=True)
        
        # Define output file path (remove .mat extension)
        file_name_base = os.path.splitext(file_name)[0]
        output_file_path = os.path.join(label_folder, f"{file_name_base}.jpg")
        
        # Save image
        img = Image.fromarray(im)
        img.save(output_file_path)
        
        print(f"Processed {i}/{len(file_list)}: {file_name} -> {output_file_path}")

# Split the dataset using splitfolders
import splitfolders
path_to_splitdataset = "/Test_Val_Data"
splitfolders.ratio(output_folder, output=path_to_splitdataset, seed=3, ratio=(0.8, 0.1, 0.1))
print("Dataset split into train/val/test.")