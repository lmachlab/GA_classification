import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from load_data import DataGen
import matplotlib.pyplot as plt

## TODO: get list of image paths and labels
## TODO: split into train and test sets
from sklearn.model_selection import train_test_split

# Assuming 'images_path' and 'img_labels' are your data
X_trainval, X_test, y_trainval, y_test = train_test_split(images_path, img_labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)

train_dataset = DataGen(X_train, y_train, image_height=1536, image_width=500)
val_dataset = DataGen(X_val, y_val, image_height=1536, image_width=500)
test_dataset = DataGen(X_test, y_test, image_height=1536, image_width=500)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)

# Step 3: Load Pre-trained Model and Feature Extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")

# Modify the final classification layer for your custom number of classes
num_classes = 2
model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

# Define loss function and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10


train_losses = []
val_losses = []
val_accuracies = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_dataloader:
        inputs = feature_extractor.pad(images, return_tensors="pt")
        inputs = inputs.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(**inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            inputs = feature_extractor.pad(images, return_tensors="pt")
            inputs = inputs.to(device)

            labels = labels.to(device)

            outputs = model(**inputs).logits
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_accuracy = 100 * correct / total
    val_loss /= len(val_dataloader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Store metrics
    train_losses.append(loss.item())
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

# Save the fine-tuned model
model.save_pretrained("fine_tuned_resnet18")

# Plot the metrics
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Val Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

