import cv2

def generate_dataset():
    face_classifier = cv2.CascadeClassifier(r"C:\Users\santh\OneDrive\Desktop\face reg\face reg\haarcascade_frontalface_default.xml")
    #Add path for your haarcascade classifier
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
        return cropped_face
    
    cap = cv2.VideoCapture(0)
    img_id = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret: 
            print("Failed to capture image. Please check the camera.")
            break
        
        cropped_face = face_cropped(frame)
        if cropped_face is not None:
            img_id += 1
            face = cv2.resize(cropped_face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = r"C:\Users\santh\dataset\class9/" + "santhi." + str(img_id) + '.jpg'
            #add your file path consisting sample images
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Cropped_Face", face)
        
        if cv2.waitKey(1) == 13 or img_id == 100:  
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed!")

generate_dataset()
