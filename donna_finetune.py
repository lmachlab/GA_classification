import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from make_dataset import DataGen
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import torchvision
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import lr_scheduler

print('check 1')
## TODO: get list of image paths and labels
img_labels = pd.read_csv('data\per_scan_data.csv')
## TODO: split into train and test sets

all_slices_path = r'\\fsmresfiles.fsm.northwestern.edu\fsmresfiles\Ophthalmology\Mirza_Images\AMD\dAMD_GA\all_slices_3'
all_slices = os.listdir(all_slices_path)
print('check 2')

all_imgs = [item for item in all_slices if item.endswith('.jpg')]

def find_label(img_name, img_labels):
   
    name_column = 'scan_name'

    found_row = img_labels[img_labels[name_column] == img_name]
    return found_row['status'].item()

labels = []
for i in range(len(all_imgs)):
    img_name = all_imgs[i]

    label = find_label(img_name, img_labels)
    if label == True:
        labels.append(1)
    else:
        labels.append(0)


print('check 3')
X_trainval, X_test, y_trainval, y_test = train_test_split(all_imgs, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)
print('check 4')
train_dataset = DataGen(X_train, y_train, all_slices_path, image_height=1536, image_width=500)
val_dataset = DataGen(X_val, y_val, all_slices_path, image_height=1536, image_width=500)
test_dataset = DataGen(X_test, y_test, all_slices_path, image_height=1536, image_width=500)
print('check 5')
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)
print('check 6')
# Step 3: Load Pre-trained Model and Feature Extractor
model = torchvision.models.resnet18(pretrained=True)
for params in model.parameters():
    params.requires_grad_ = False
# Add new final layer 
nr_filters = model.fc.in_features
model.fc = nn.Linear(nr_filters, 1)
print('check 7')


# Define loss function and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print('check 8')
criterion = BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.fc.parameters())
num_epochs = 10
print('check 9')

train_losses = []
val_losses = []
val_accuracies = []

# Training loop
for epoch in range(num_epochs):
    print('starting epoch ', epoch)
    model.train()
    for images, labels in train_dataloader:
        # inputs = feature_extractor(images, return_tensors="pt")
        inputs = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(**inputs).logits
        print('outputs: ', outputs.shape)
        print('labels: ', labels)
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
            # inputs = feature_extractor.pad(images, return_tensors="pt")
            inputs = images.to(device)

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
model.save_pretrained("fine_tuned_resnet18_donna")

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

