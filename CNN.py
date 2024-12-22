import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Define the path to your dataset
image_dir = r"C:\Users\santh\DATASETT"


# Initialize data and labels
X = []  # Feature matrix (images)
Y = []  # Labels

# Load and preprocess data
for class_name in os.listdir(image_dir):
    class_path = os.path.join(image_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # Resize the image to a fixed size (e.g., 100x100)
        img_resized = cv2.resize(img, (100, 100))
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_resized / 255.0
        
        # Add the image to the feature matrix
        X.append(img_normalized)
        
        # Add the corresponding label
        Y.append(class_name)

# Convert data to NumPy arrays
X = np.array(X).reshape(-1, 100, 100, 1)  # Add channel dimension for grayscale
Y = np.array(Y)

# Encode labels to integers
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Convert labels to one-hot encoding
Y_one_hot = to_categorical(Y_encoded)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_one_hot, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)