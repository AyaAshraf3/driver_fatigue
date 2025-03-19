import cv2
import torch
import torch.nn as nn
from PIL import Image

import numpy as np
from torchvision import transforms
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained model
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
in_features = model.classifier[-1].in_features  # Access last layer dynamically
model.classifier[-1] = nn.Linear(in_features, 5)

model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)

model.load_state_dict(torch.load('./Models/FER2013/Models FER with AffectNet Greyscale/Model_E45.pth', map_location=device))
model.eval()
model = model.to(device)

# Define transformations
transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Adjusted for grayscale images
])


# Load face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define class labels (adjust to your dataset)
emotion_labels = ["Anger", "Fear", "Happy", "Neutral", "Sad"]

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
   # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_img = frame[y:y+h, x:x+w]
        
        # Convert to RGB and apply transformations
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)


        # Convert to PIL image for proper transformation handling
        face_pil = Image.fromarray(face_rgb)
        transformed = transform(face_pil).unsqueeze(0).to(device)

        #transformed = transform(face_rgb).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(transformed)
            _, predicted = torch.max(outputs, 1)
            emotion = emotion_labels[predicted.item()]
            probabilities = torch.softmax(outputs, dim=1)[0] * 100
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_class = emotion_labels[predicted_idx.item()]
        
        # Draw rectangle and emotion text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

         # Add text (class name and confidence score)
        label = f"{predicted_class}: {confidence.item() :.2f}%"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
