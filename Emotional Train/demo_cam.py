import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import torchvision.models as models


# Load your trained model
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 8)  # Adjust 7 to your number of classes
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
model = model.to('cuda')

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define class labels (adjust to your dataset)
emotion_labels = ['Angry','Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_img = frame[y:y+h, x:x+w]
        
        # Convert to RGB and apply transformations
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        device = torch.device("cuda")
        transformed = transform(face_rgb).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(transformed)
            _, predicted = torch.max(outputs, 1)
            emotion = emotion_labels[predicted.item()]
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        
        # Draw rectangle and emotion text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{emotion} {probabilities[predicted].item():.1f}%', 
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()