import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from yolov5 import YOLOv5 

# Load YOLOv5 Model
MODEL_PATH = r"D:\grad project\driver_fatigue\models\best_ours.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# Load YOLOv5 Model
model = YOLOv5("yolov5s.pt", device=device)  # Load a pre-trained YOLOv5 model
model.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # Load custom weights
model.model.eval()  # Set to evaluation mode

print("Model loaded successfully!")


# Define class names (Adjust if needed)
CLASS_NAMES = ["Open Eyes", "Closed Eyes", "Yawn"]

# Load the test image
IMAGE_PATH = r"D:\grad project\download (1).jpg"  
image = Image.open(IMAGE_PATH).convert("RGB")

# Preprocess Image (Resize, Normalize)
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # YOLO input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

# Perform Inference
with torch.no_grad():
    results = model(input_tensor)

# Extract Detection Results
detections = results[0]  # YOLOv5 returns a list, take the first element
if detections is not None:
    boxes = detections[:, :4]  # Bounding boxes
    scores = detections[:, 4]  # Confidence scores
    class_ids = detections[:, 5].int()  # Class indices

    # Load Image with OpenCV for visualization
    image_cv = cv2.imread(IMAGE_PATH)
    h, w, _ = image_cv.shape

    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box.tolist())

        # Draw bounding box
        color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # Green for open eyes, Red for closed/yawn
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)

        # Add label text
        label = f"{CLASS_NAMES[class_id]} ({score:.2f})"
        cv2.putText(image_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save and Display the Output
    output_path = "output.jpg"
    cv2.imwrite(output_path, image_cv)
    cv2.imshow("Detection Results", image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No objects detected.")
