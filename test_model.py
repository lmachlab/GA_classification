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
import csv


torch.cuda.empty_cache()

print('check 1')
img_labels = pd.read_csv('data\per_scan_data.csv')

all_slices_path = r'\\fsmresfiles.fsm.northwestern.edu\fsmresfiles\Ophthalmology\Mirza_Images\AMD\dAMD_GA\all_slices_3'
all_slices = os.listdir(all_slices_path)
print('check 2')

all_imgs = [item for item in all_slices if item.endswith('.jpg')]

def find_label(img_name, img_labels):
   
    name_column = 'scan_name'
    found_row = img_labels[img_labels[name_column] == img_name]
    return found_row['status'].item()




print('check 3')
file = open('testset_values.csv', 'r')
X_test = list(csv.reader(file, delimiter=","))[0]
file.close()
y_test = []
for i in range(len(X_test)):
    img_name = X_test[i]

    label = find_label(img_name, img_labels)
    if label == True:
        y_test.append(1)
    else:
        y_test.append(0)
# X_trainval, X_test, y_trainval, y_test = train_test_split(all_imgs, labels, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)
print('check 4')
# train_dataset = DataGen(X_train, y_train, all_slices_path, image_height=1536, image_width=500)
# val_dataset = DataGen(X_val, y_val, all_slices_path, image_height=1536, image_width=500)
test_dataset = DataGen(X_test, y_test, all_slices_path, image_height=1536, image_width=500)
print('check 5')
# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)
print('check 6')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if device.type != "cuda":
#     raise Exception("GPU not available, please ensure you have a compatible GPU and PyTorch with CUDA support installed.")
print(f"Training on device {device}.")

# Load pre-trained model and modify the last layer
model = models.resnet18()

num_features = model.fc.in_features
# model.fc = nn.Linear(num_features, 1)
model.fc = nn.Sequential(
    nn.Linear(num_features, 1),  # Assuming binary classification
    nn.Sigmoid())
model.load_state_dict(torch.load(r'finetuned_torchvision_model_8epoch.pth'))
model = model.to(device) # Send model to device (GPU if available, else CPU)

model.eval()  # Set the model to evaluation mode
true_labels = []
preds = []
print('number of loops needed: ', len(test_dataloader))
# Loop through the test data
with torch.no_grad():
    for i, (inputs,labels) in enumerate(test_dataloader):
        print('loop number: ', i)
        inputs, labels = inputs.float().to(device), labels.to(device)
        inputs = inputs.permute(0, 3, 1, 2).to(device)
        # Forward pass
        outputs = model(inputs).squeeze()
        true_labels.extend(labels.cpu().numpy())
        preds.extend(outputs.cpu().detach().numpy())

    
# Convert to numpy arrays for use with sklearn
true_labels = np.array(true_labels)
preds = np.array(preds)
pred_binary = np.where(preds >= 0.5, 1, 0)
# Compute ROC AUC
roc_auc = roc_auc_score(true_labels, preds)
# Compute accuracy
accuracy = accuracy_score(true_labels, pred_binary)
# Compute confusion matrix
cm = confusion_matrix(true_labels, pred_binary)
# Plot confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
print(f"Accuracy: {accuracy}")
print(f"ROC AUC: {roc_auc}")