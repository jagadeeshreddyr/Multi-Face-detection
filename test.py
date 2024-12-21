import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError, BinaryAccuracy

# Register custom loss functions and metrics
custom_objects = {
    'mse': MeanSquaredError(),  # Register 'mse' loss as MeanSquaredError
    'binary_crossentropy': 'binary_crossentropy',  # 'binary_crossentropy' should be recognized by default
    'mae': MeanAbsoluteError(),  # Register 'mae' metric
    'accuracy': BinaryAccuracy()  # Register 'accuracy' metric
}

# Load the model with the custom objects
model = load_model('model/freeze_model.h5', custom_objects=custom_objects)

# Load the pre-trained OpenCV face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up the camera capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert the image to grayscale (for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate through detected faces
    for (x, y, w, h) in faces:
        # Draw a bounding box around each face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the face region
        face_region = frame[y:y + h, x:x + w]

        # Preprocess the face region (resize and normalize)
        face_region_resized = cv2.resize(face_region, (128, 128))  # Resize to match model input
        face_region_resized = face_region_resized / 255.0  # Normalize the image

        # Expand dimensions to match model input shape (batch size, height, width, channels)
        face_region_resized = np.expand_dims(face_region_resized, axis=0)

        # Predict the age and gender
        age_pred, gender_pred = model.predict(face_region_resized)

        # Get predicted age and gender (we round the age and map gender to Male/Female)
        predicted_age = int(age_pred[0][0])
        predicted_gender = 'Male' if gender_pred[0][0] < 0.5 else 'Female'  # Apply sigmoid threshold

        # Display the predictions on the frame
        cv2.putText(frame, f'Age: {predicted_age}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f'Gender: {predicted_gender}', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection and Prediction', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
