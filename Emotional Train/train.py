"""
1. Imports
"""
import random
import torch  # PyTorch library for deep learning.
import torch.nn as nn  # Neural network module for building models.
import torch.optim as optim  # Optimizers for training models (e.g., SGD, Adam).
import cv2
import numpy as np
from IPython.display import display
from torch.utils.data import DataLoader
from torchvision import (
    datasets,
    models,
    transforms,
)
import cv2
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    T_num_batches = len(data_loader)
    batch_count = 1
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            print(f"Training... Batch {batch_count}/{T_num_batches}")
            batch_count += 1
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def fit_model(model, model_name, train_loader, valid_loader, optimizer, criterion, device, num_epochs):
    model = model.to(device)
    train_stats = {"train_loss": [], "valid_loss": [], "train_acc": [], "valid_acc": []}
    T_num_batches = len(train_loader)
    V_num_batches = len(valid_loader)
    best_val_accuracy = 0
    
    # Open log file
    log_file = open("./Models/FER2013/Models FER with AffectNet Greyscale/training_log.txt", "a")

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch = 1
    

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            tot_bat = f"Training Batch {batch}/{T_num_batches}"
            print(tot_bat)
            batch = batch + 1

        train_loss = running_loss / total
        train_acc = 100 * correct / total

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        batch = 1

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                tot_bat = f"Validating Batch {batch}/{V_num_batches}"
                print(tot_bat)
                batch = batch + 1

        valid_loss = val_loss / val_total
        valid_acc = 100 * val_correct / val_total

        # Save best model
        if valid_acc > best_val_accuracy:
            best_val_accuracy = valid_acc
            model_path = f"./Models/FER2013/Models FER with AffectNet Greyscale//Model_E{epoch}.pth"
            torch.save(model.state_dict(), model_path)
            #C_Matrix_2.get_c_matrix(epoch, model_path, "./Models/FER2013/C-Matix")

            # Save log entry
            log_file.write(f"Saved a Better Model E{epoch}\n")
            print("Saved a Better Model E{epoch}")  # Keep printing to console

        # Log epoch results
        log_message = f"Epoch [{epoch+1}/{num_epochs}] -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%\n"
        log_file.write(log_message)
        print(log_message.strip())  # Keep printing to console

        # Store statistics
        train_stats["train_loss"].append(train_loss)
        train_stats["valid_loss"].append(valid_loss)
        train_stats["train_acc"].append(train_acc)
        train_stats["valid_acc"].append(valid_acc)

    # Close the log file after training
    log_file.close()

    return train_stats

SEED = 47
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

"""
3. Image Preprocessing
"""


# Define paths to your dataset(train set / test set)
train_dir = "./Dataset/Full Dataset"
test_dir =  "./Dataset/FER2013/test"

classes = {
    0: "Anger",
    1: "Fear",
    2: "Happy",
    3: "Neutral",
    4: "Sad"
}

#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data_transforms = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Adjusted for grayscale images

        #normalize,
    ]
)
train_valid_data = datasets.ImageFolder(root=train_dir, transform = data_transforms)
Test_data = datasets.ImageFolder(root=test_dir, transform = data_transforms)

RATIO = 0.8

n_train_examples = int(len(train_valid_data) * RATIO)
n_Valid_examples = len(train_valid_data) - n_train_examples

Train_data, Valid_data = torch.utils.data.random_split(
    train_valid_data, [n_train_examples, n_Valid_examples]
)


print(f"Number of Training examples: {len(Train_data)}")
print(f"Number of Validation examples: {len(Valid_data)}")
print(f"Number of Training examples: {len(Test_data)}")

"""
4. Loading Datasets
"""

# Load the dataset

# Create DataLoader to load batches of images
batch_size = 64
train_loader = DataLoader(Train_data, batch_size = batch_size * 2, shuffle=True)
valid_loader = DataLoader(Valid_data, batch_size = batch_size * 2, shuffle= False)
test_loader = DataLoader(Test_data, batch_size = batch_size * 2, shuffle= False)
"""
5. Model Customization
"""
## model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
## model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

# Modify the final classifier layer
#model.classifier[3] = nn.Linear(model.classifier[3].in_features, 5)  # 5 classes
in_features = model.classifier[-1].in_features  # Access last layer dynamically
model.classifier[-1] = nn.Linear(in_features, 5)

model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)

# Move model to device
device = torch.device("cuda")
model.to(device)

"""
6. Training the Model
"""

# Define the loss function
criterion = nn.CrossEntropyLoss()  # suitable for multi-class classification.

# Use Adam optimizer (you can adjust learning rate)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
model_name = "MobileNet"
print(f"The model has {count_parameters(model):,} trainable parameters")

train_stats = fit_model(
    model,
    model_name=model_name,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    num_epochs=num_epochs
)

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

PATH = "./Models/FER2013/Models FER with AffectNet Greyscale/Model_E50.pth"
torch.save(model.state_dict(), PATH)
