import numpy as np 
import cv2
import shutil
import itertools
import imutils
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import splitfolders
import os
from sklearn.utils import shuffle
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns


labels = ['glioma', 'meningioma', 'pituitary']



def build_effNet(image_size=240):
    # Transfer learning from EfficientNetB1 trained on "ImageNet" dataset.
    effnet = EfficientNetB1(weights='imagenet',include_top=False,input_shape=(image_size,image_size,3))
    model = effnet.output
    model = tf.keras.layers.GlobalAveragePooling2D()(model)
    model = tf.keras.layers.Dropout(rate=0.5)(model)
    model = tf.keras.layers.Dense(3,activation='softmax')(model)    
    model = tf.keras.models.Model(inputs=effnet.input, outputs = model)
    # model.summary()
    return model

def load_images_from_folder(base_folder):
    X = []
    y = []
    
    for label in labels:
        folder_path = os.path.join(base_folder, label)
        if not os.path.exists(folder_path):
            print(f"[WARNING] Folder {folder_path} does not exist. Skipping.")
            continue
        
        print(f"Loading {label} images...")
        for filename in tqdm(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[ERROR] Failed to read {img_path}")
                continue
            
            X.append(img)
            y.append(label)
    
    return np.array(X), np.array(y)


def display_sample(X, y, n_samples=5, title="Sample Images"):
    # Select n_samples random indices from the dataset
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices):
        img = X[idx]
        label = y[idx]
        # Convert image from BGR to RGB for correct color display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, n_samples, i + 1)
        plt.imshow(img_rgb)
        plt.title(label)
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

def print_class_distribution(y, dataset_name="Dataset", imbalance_threshold=2.0):
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    total_samples = len(y)
    
    print(f"\nClass Distribution for {dataset_name}:")
    for cls, count in class_dist.items():
        print(f"{cls}: {count} samples ({count/total_samples*100:.2f}%)")
    
    # Calculate imbalance ratio (majority class : minority class)
    max_count = np.max(counts)
    min_count = np.min(counts)
    imbalance_ratio = max_count / min_count
    
    print(f"\nImbalance Ratio (Majority:Minority): {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > imbalance_threshold:
        print(f"Warning: Significant class imbalance detected (threshold = {imbalance_threshold}:1).")

def train_model(X_train, y_train_onehot, 
                X_val, y_val_onehot, 
                saved_checkpoints='best_model.h5',
                epochs=50, 
                batch_size=32):
    """
    Train an EfficientNet model with callbacks and return model + history
    """
    # Initialize callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
    checkpoint = ModelCheckpoint(saved_checkpoints, save_best_only=True, monitor='val_loss')

    # Build and compile model
    model = build_effNet()
    model.compile(loss='categorical_crossentropy', 
                 optimizer='Adam', 
                 metrics=['accuracy'])

    # Train model
    history = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint, reduce_lr],
        verbose=1
    )
    
    return model, history


def test_model(saved_model_path, X_test, y_test_onehot, class_names):
    # Load the saved model
    model = tf.keras.models.load_model(saved_model_path)
    
    # Convert one-hot encoded labels to class indices
    y_true = np.argmax(y_test_onehot, axis=1)
    
    # Predict probabilities and class labels
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Accuracy
    test_loss, test_acc = model.evaluate(X_test, y_test_onehot, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # ROC-AUC Scores
    macro_roc_auc = roc_auc_score(y_test_onehot, y_pred_probs, multi_class='ovo', average='macro')
    print(f"Macro-Average ROC AUC (One-vs-One): {macro_roc_auc:.4f}")
    
    # Per-class AUC (One-vs-Rest)
    for i, cls in enumerate(class_names):
        auc = roc_auc_score(y_test_onehot[:, i], y_pred_probs[:, i])
        print(f"ROC AUC for {cls}: {auc:.4f}")




base_train_folder = "/home/yck/Desktop/GITHUB/Bayesian Reinforcement Learning/MULTICLASS_CLASSIFICATION/multi-class-brain-tumor-classification/Train/"
base_val_folder = "/home/yck/Desktop/GITHUB/Bayesian Reinforcement Learning/MULTICLASS_CLASSIFICATION/multi-class-brain-tumor-classification/Val/"
base_test_folder = "/home/yck/Desktop/GITHUB/Bayesian Reinforcement Learning/MULTICLASS_CLASSIFICATION/multi-class-brain-tumor-classification/Test/"


X_train, y_train = load_images_from_folder(base_train_folder)
X_val, y_val= load_images_from_folder(base_val_folder)
X_test, y_test = load_images_from_folder(base_test_folder)
X_train, y_train = shuffle(X_train,y_train, random_state=101)
X_val, y_val = shuffle(X_test,y_test, random_state=101)
X_test, y_test = shuffle(X_test,y_test, random_state=101)
print("Shapes:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")


# Display samples from training, validation, and test sets
display_sample(X_train, y_train, n_samples=5, title="Training Samples")
display_sample(X_val, y_val, n_samples=5, title="Validation Samples")
display_sample(X_test, y_test, n_samples=5, title="Test Samples")


print_class_distribution(y_train, "Training Set")
saved_checkpoint= '/home/yck/Desktop/GITHUB/Bayesian Reinforcement Learning/MULTICLASS_CLASSIFICATION/multi-class-brain-tumor-classification/Checkpoints/best_checkpoint_model.h5'

# Encode labels to integers and one-hot format
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Convert string labels to integers
y_train_encoded = label_encoder.transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

# Convert to one-hot, because categorical_crossentropy expects labels in a one-hot encoded format
y_train_onehot = to_categorical(y_train_encoded, num_classes=3)
y_val_onehot = to_categorical(y_val_encoded, num_classes=3)
y_test_onehot = to_categorical(y_test_encoded, num_classes=3)



# Train model
trained_model, training_history = train_model(
    X_train=X_train,
    y_train_onehot=y_train_onehot,
    X_val=X_val,
    y_val_onehot=y_val_onehot,
    saved_checkpoints=saved_checkpoint,
    epochs=100,
    batch_size=32
)


# Test saved model
test_model(
    saved_model_path=saved_checkpoint,
    X_test=X_test,
    y_test_onehot=y_test_onehot,
    class_names=labels
)

print('stop')




   



print('stop')