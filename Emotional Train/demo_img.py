import torch
import torch.nn as nn  # Neural network module for building models.
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
from PIL import Image
import os

# Configuration
image_folder = "./Test_Final"  # Folder containing images for testing
device = torch.device("cuda")

# Load Model
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

in_features = model.classifier[-1].in_features  # Access last layer dynamically
model.classifier[-1] = nn.Linear(in_features, 5)

model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)

model.load_state_dict(torch.load('./Models/FER2013/Models FER with AffectNet Greyscale/Model_E34.pth', map_location=device))
model.to(device)
model.eval()

# Define Image Preprocessing
transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Adjusted for grayscale images
])

# Load Class Names
emotion_labels = ["Anger", "Fear", "Happy", "Neutral", "Sad"]


# Process Images in Folder
for img_name in os.listdir(image_folder):
    if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(image_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

            probabilities = torch.nn.functional.softmax(output[0], dim=0)  # Convert to probabilities
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_class = emotion_labels[predicted_idx.item()]
        
        
        # Convert image to OpenCV format
        image_cv2 = cv2.imread(img_path)
        image_cv2 = cv2.resize(image_cv2, (600, 600))  # Resize for better visualization
        
        # Add text (class name and confidence score)
        label = f"{predicted_class}: {confidence.item() * 100:.2f}%"
        cv2.putText(image_cv2, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show Image
        cv2.imshow("Prediction", image_cv2)
        # Display image with predicted label
        print(f"Image: {img_name} → Predicted: {predicted_class} → Confidence: {confidence.item():.2f}")
        cv2.waitKey(0)  # Wait for key press

# Close OpenCV windows
cv2.destroyAllWindows()
    
