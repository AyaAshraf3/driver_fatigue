import os
import shutil
import random

# Set the paths
dataset_path = r"D:\grad project\customed dataset\HOW object detection dataset\HOW dataset roboflow\train"
output_path = r"D:\grad project\customed dataset\HOW object detection dataset\HOW dataset roboflow\split dataset"

# Create output directories
splits = ["train", "valid", "test"]
for split in splits:
    os.makedirs(os.path.join(output_path, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, split, "labels"), exist_ok=True)

# Collect all images and labels
image_folder = os.path.join(dataset_path, "images")
label_folder = os.path.join(dataset_path, "labels")

images = sorted([f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png", ".jpeg"))])

# Shuffle images randomly
random.seed(42)
random.shuffle(images)

# Define split sizes
total_images = len(images)
train_size = int(0.80 * total_images)
valid_size = int(0.10 * total_images)

# Split dataset
train_images = images[:train_size]
valid_images = images[train_size:train_size + valid_size]
test_images = images[train_size + valid_size:]

# Function to move images and labels
def move_files(file_list, split_name):
    for img_file in file_list:
        img_src = os.path.join(image_folder, img_file)
        label_src = os.path.join(label_folder, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))

        img_dest = os.path.join(output_path, split_name, "images", img_file)
        label_dest = os.path.join(output_path, split_name, "labels", os.path.basename(label_src))

        shutil.copy(img_src, img_dest)  # Copy image
        if os.path.exists(label_src):  # Copy label if it exists
            shutil.copy(label_src, label_dest)

# Move images and labels to their respective folders
move_files(train_images, "train")
move_files(valid_images, "valid")
move_files(test_images, "test")

print("Dataset split completed successfully!")
