import numpy as np
import cv2
from tensorflow.keras.models import load_model

import helper


if __name__ == "__main__":
    model_name = 'attractiveNet_mnv2'
    model_path = 'models/' + model_name + '.h5'

    # Load the face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    model = load_model(model_path)
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect faces directly on color image
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        
        if len(faces) > 0:
            # Draw rectangles around all detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Get the first detected face (you can change this to select a different face)
            (x, y, w, h) = faces[0]
            
            # Highlight the face being scored with a different color
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
            
            # Get face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Predict attractiveness score
            score = model.predict(np.expand_dims(helper.preprocess_image(face_roi,(350,350)), axis=0))
            
            # Display score and face count
            text1 = f'AttractiveNet Score: {str(round(score[0][0],1))}'
            text2 = f'Faces detected: {len(faces)}'
            if len(faces) > 1:
                text2 += " (scoring first face)"
            cv2.putText(frame, text1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, text2, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        else:
            # No face detected
            text1 = "No face detected"
            cv2.putText(frame, text1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Display instructions
        text3 = "press 'Q' to exit"
        cv2.putText(frame, text3, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('AttractiveNet', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()