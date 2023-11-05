from torchvision import models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss
import torch.optim as optim
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd
from make_dataset import DataGen

print('check 1')
img_labels = pd.read_csv('data/per_scan_data.csv')

all_slices_path = r'/Volumes/fsmresfiles/Ophthalmology/Mirza_Images/AMD/dAMD_GA/all_slices_3'
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if device.type != "cuda":
#     raise Exception("GPU not available, please ensure you have a compatible GPU and PyTorch with CUDA support installed.")
print(f"Training on device {device}.")

# Load pre-trained model and modify the last layer
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
# model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)  # Assuming binary classification
model = model.to(device) # Send model to device (GPU if available, else CPU)
# Loss function and optimizer
criterion = BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Initialize lists to save the losses
train_losses = []
val_losses = []
n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_dataloader):
        # Move data and labels to device
        inputs, labels = inputs.float().to(device), labels.to(device)
        optimizer.zero_grad()
 
        inputs = inputs.permute(0, 3, 1, 2).to(device)
 
       
        outputs = model(inputs).squeeze()
     
        labels = labels.float()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
 
        running_loss += loss.item()
    # Save the training loss for this epoch
    train_losses.append(running_loss / len(train_dataloader))
    print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {running_loss / len(train_dataloader)}")
    # Validate
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for i, (inputs, masks, labels) in enumerate(val_dataloader):
            inputs, labels = inputs.float().to(device), labels.to(device)
            inputs = inputs.permute(0, 3, 1, 2).to(device)
            outputs = model(inputs).squeeze()
            labels = labels.float()
            val_loss = criterion(outputs, labels)
            running_val_loss += val_loss.item()
    # Save the validation loss for this epoch
    val_losses.append(running_val_loss / len(val_dataloader))
    print(f"Epoch {epoch + 1}/{n_epochs}, Val Loss: {running_val_loss / len(val_dataloader)}")
# Plotting
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
# Save the model
model_save_path = "./cnn_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
# test

model.eval()  # Set the model to evaluation mode
true_labels = []
pred_labels = []
outputs_list = []
# Loop through the test data
for inputs,labels in test_dataloader:
    inputs, labels = inputs.float().to(device), labels.to(device)
    inputs = inputs.permute(0, 3, 1, 2).to(device)
    # Forward pass
    outputs = model(inputs).squeeze()
    _, preds = torch.max(outputs, 1)  # Get the predicted classes
    true_labels.extend(labels.cpu().numpy())
    pred_labels.extend(preds.cpu().numpy())
    outputs_list.extend(outputs.detach().cpu().numpy())
# Convert to numpy arrays for use with sklearn
true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)
# Compute ROC AUC
roc_auc = roc_auc_score(label_binarize(true_labels, classes=[0,1]),
                        label_binarize(pred_labels, classes=[0,1]),
                        average='macro')
# Compute accuracy
accuracy = accuracy_score(true_labels, pred_labels)
# Compute confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
# Plot confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
print(f"Accuracy: {accuracy}")
print(f"ROC AUC: {roc_auc}")