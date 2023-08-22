import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd

# Load pre-trained emotion classification model
emotion_classification_model = load_model('deployment\model\model_fine_tune.h5')  # Replace with actual path

# Load Haarcascades face detection classifier
face_cascade = cv2.CascadeClassifier('deployment\model\haarcascade_frontalface_default.xml')

class_labels = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Process each detected face
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        
        # Preprocess the face image
        resized_face_img = cv2.resize(face_img, (48, 48))
        img_array = image.img_to_array(resized_face_img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= (255*117)  # Normalize the image
        
        # Perform emotion prediction using the loaded model
        inf_pred_single = emotion_classification_model.predict(img_array)
        
        max_pred_single = np.argsort(inf_pred_single[0])[-2:][::-1]
        
        data_inf_single = []
        rank = []
        
        for i in inf_pred_single[0]:
            value = i * 100
            rank.append(value)
            data_inf_single.append(f'{value.round(2)}%')
        
        rank = (-np.array(rank)).argsort()[:2]
        
        pred_class_single = pd.DataFrame(class_labels).loc[rank][0].tolist()
        
        # Draw bounding box and emotion label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'Emotion: {pred_class_single[0]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow('Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
