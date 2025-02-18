import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
from PIL import Image
import os

# Configuration
model_path = "The_Best_Model.pth"  # Update with the actual best model path
image_folder = "./Final_Test"  # Folder containing images for testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(num_features, 8)  # Update class count if different
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Class Names
emotion_labels = ['Angry','Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


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
    
