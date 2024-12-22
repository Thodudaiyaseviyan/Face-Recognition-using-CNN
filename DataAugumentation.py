import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img
import numpy as np

# Paths
input_folder = r"C:\Users\santh\preprocessdataset\class12"  # Folder containing the original 100 images
output_folder = r"C:\Users\santh\augpredata\class12"  # Folder to save augmented images


# Data Augmentation settings
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Augment and save images
num_augmented_images_per_original = 10  # Number of augmented images per original image
image_size = (200, 200)  # Target image size (same as your model input size)

for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Ensure only image files are processed
        # Load image
        img_path = os.path.join(input_folder, filename)
        img = load_img(img_path, target_size=image_size)
        img_array = img_to_array(img)  # Convert to numpy array
        img_array = img_array.reshape((1,) + img_array.shape)  # Add batch dimension

        # Generate augmented images
        count = 0
        for batch in datagen.flow(img_array, batch_size=1):
            # Save augmented image
            augmented_img_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_aug_{count}.jpg")
            save_img(augmented_img_path, batch[0])
            count += 1
            if count >= num_augmented_images_per_original:
                break  # Stop after generating the desired number of augmentations

print(f"Data augmentation completed. Augmented images saved in {output_folder}")
