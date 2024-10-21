import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("E:\\He_is_enough03 X UniqoXTech X Dreams\\Click_here\\Machine Learning\\CIFER 10\\CIFER10.h5")

# Define the labels (same as in training)
labels = '''airplane automobile bird cat deer dog frog horse ship truck'''.split()

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (32, 32))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb / 255.0
    frame_reshaped = frame_normalized.reshape(1, 32, 32, 3)
    
    return frame_reshaped

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert the frame to grayscale (required for the cascade detector)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect objects in the frame (using the Haar Cascade here for demonstration)
    objects = cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in objects:
        # Extract the region of interest (ROI) from the frame
        region = frame[y:y+h, x:x+w]
        
        # Preprocess the region
        processed_region = preprocess_frame(region)
        
        # Predict the label using the trained model
        prediction = model.predict(processed_region)
        predicted_label = labels[np.argmax(prediction)]
        
        # Draw a rectangle around the detected object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Put the predicted label text on the frame (above the rectangle)
        cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    # Display the frame with detections
    cv2.imshow('Webcam Object Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
