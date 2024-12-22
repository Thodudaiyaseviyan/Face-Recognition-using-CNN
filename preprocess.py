import cv2
import numpy as np
import os

def preprocess_image(image_path):
    """
    Preprocess the input image: Apply filters, enhance contrast, and prepare for face detection.
    """
   
    image = cv2.imread(image_path)
    
   
    bilateral_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

   
    gray_image = cv2.cvtColor(bilateral_image, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.equalizeHist(gray_image)

    # Increase sharpness using a kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(enhanced_image, -1, kernel)

    return sharpened_image

def detect_and_resize_face(image, target_size=(100, 100)):
    """
    Detect the face in the image and resize it to the target size.
    """
    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Crop the first detected face
        x, y, w, h = faces[0]
        cropped_face = image[y:y+h, x:x+w]
        
        # Resize the cropped face to the target size
        resized_face = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_AREA)
        
        # Ensure the resized image has 3 channels (RGB)
        if len(resized_face.shape) == 2:  # If grayscale, convert to BGR
            resized_face = cv2.cvtColor(resized_face, cv2.COLOR_GRAY2BGR)
        
        return resized_face
    else:
        print("No face detected in the image.")
        return None

def process_folder(input_folder, output_folder, target_size=(200, 200)):
    """
    Process all images in the input folder, detect faces, resize, and save to the output folder.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image formats
            image_path = os.path.join(input_folder, filename)
            print(f"Processing: {image_path}")
            
            # Preprocess and detect face
            processed_image = preprocess_image(image_path)
            face_image = detect_and_resize_face(processed_image, target_size)
            
            if face_image is not None:
                # Save the processed face image
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, face_image, [cv2.IMWRITE_JPEG_QUALITY, 95])  # Save with high quality
                print(f"Processed and saved: {output_path}")
            else:
                print(f"Face not found in: {filename}")

# Example usage
input_folder = r"C:\Users\santh\OneDrive\Desktop\fall sem 24-25\biometric\chumma"
output_folder = r"C:\Users\santh\simpletest"

process_folder(input_folder,output_folder)