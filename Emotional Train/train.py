import datetime
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torchvision.models as models

# Load Images with Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_path = "./Dataset"
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Split dataset (80% train, 10% validation, 10% test)
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size  # Ensure full coverage

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
train_dataset.dataset.transform = train_transform

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Print dataset sizes
print(f"Train set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# Load Pre-trained ResNet Model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Modify Classifier
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # 50% dropout
    nn.Linear(num_features, 8))


# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Move Model to Device
device = torch.device("cuda")
model = model.to(device)
best_val_accuracy = 0
counter = 0


# Training
num_epochs = 100  # Adjust as needed
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print(datetime.datetime.now().time())
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 0:  # Print every 10 batches
            print(f"Epoch {epoch}/{num_epochs}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

    # Validation Loop
    model.eval()
    correct, total = 0, 0
    val_loss = 0.0
    best_val_loss = 0.0
    patience = 5

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    val_accuracy = 100 * correct / total

    print(f"Epoch {epoch}, "
            f"Train Loss: {running_loss/len(train_loader):.4f}, "
            f"Validation Loss: {val_loss/len(val_loader):.4f}, "
            f"Validation Accuracy: {val_accuracy:.2f}%")

    # Save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), f"The_Best_Model_{epoch}.pth")
        print("Saved new best model")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            

print("Training complete!")

# Final Test Evaluation
model.eval()
correct, total = 0, 0
test_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        test_loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total

print(f"\nFinal Test Set Performance: "
      f"Test Loss: {test_loss/len(test_loader):.4f}, "
      f"Test Accuracy: {test_accuracy:.2f}%")
