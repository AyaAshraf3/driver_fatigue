import cv2
import os
import yaml
import matplotlib.pyplot as plt

# Load class names from data.yaml
data_yaml_path = r"D:\grad project\customed dataset\HOW object detection dataset\HOW dataset roboflow\data.yaml"
with open(data_yaml_path, 'r') as file:
    data_yaml = yaml.safe_load(file)
    class_names = data_yaml.get("names", [])

# Path to images and labels
image_folder = r"D:\grad project\customed dataset\HOW object detection dataset\HOW dataset roboflow\split dataset\valid\images"  # Change this to your image directory
label_folder = r"D:\grad project\customed dataset\HOW object detection dataset\HOW dataset roboflow\split dataset\valid\labels"  # Path to label directory

# Get all image filenames
image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]  # Adjust for PNG or other formats

# Function to draw bounding boxes
def draw_bboxes(image_path, label_path, image_name):
    image = cv2.imread(image_path)
    h, w, _ = image.shape  # Get image dimensions

    if os.path.exists(label_path):
        with open(label_path, "r") as file:
            lines = file.readlines()

        for line in lines:
            data = line.strip().split()
            class_id = int(data[0])
            x_center, y_center, bbox_width, bbox_height = map(float, data[1:])

            # Convert YOLO format to pixel coordinates
            x1 = int((x_center - bbox_width / 2) * w)
            y1 = int((y_center - bbox_height / 2) * h)
            x2 = int((x_center + bbox_width / 2) * w)
            y2 = int((y_center + bbox_height / 2) * h)

            # Draw rectangle
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Put class label
            label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert BGR to RGB for displaying with Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display image
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.title(image_name)  # Display image name as title
    plt.axis("off")
    plt.show()

# Loop through all images in the folder
for img in image_files:
    label_file = img.replace(".jpg", ".txt")  # Adjust for PNG if needed
    label_path = os.path.join(label_folder, label_file)
    image_path = os.path.join(image_folder, img)
    draw_bboxes(image_path, label_path, img)

'''
import cv2
import os
import matplotlib.pyplot as plt

# Path to images and labels
image_folder = r"D:\grad project\customed dataset\HOW object detection dataset\kaggle\kaggle\working\out_handsON_aya_images" # Change this to your image directory
label_folder = r"D:\grad project\customed dataset\HOW object detection dataset\kaggle\working\out_handsON_aya_labels"# Replace with your actual class names
class_names = ['hands off wheel','hands on wheel']

# Get all image filenames
image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]  # Adjust for PNG or other formats

# Function to draw bounding boxes
def draw_bboxes(image_path, label_path, image_name):
    image = cv2.imread(image_path)
    h, w, _ = image.shape  # Get image dimensions

    if os.path.exists(label_path):
        with open(label_path, "r") as file:
            lines = file.readlines()

        for line in lines:
            data = line.strip().split()
            class_id = int(data[0])
            x_center, y_center, bbox_width, bbox_height = map(float, data[1:])

            # Convert YOLO format to pixel coordinates
            x1 = int((x_center - bbox_width / 2) * w)
            y1 = int((y_center - bbox_height / 2) * h)
            x2 = int((x_center + bbox_width / 2) * w)
            y2 = int((y_center + bbox_height / 2) * h)

            # Draw rectangle
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Put class label
            label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert BGR to RGB for displaying with Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display image
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.title(image_name)  # Display image name as title
    plt.axis("off")
    plt.show()

# Loop through all images in the folder
for img in image_files:
    label_file = img.replace(".jpg", ".txt")  # Adjust for PNG if needed
    label_path = os.path.join(label_folder, label_file)
    image_path = os.path.join(image_folder, img)
    draw_bboxes(image_path, label_path, img)
'''