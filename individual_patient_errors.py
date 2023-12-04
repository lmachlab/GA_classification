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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on device {device}.")


model = models.resnet18(pretrained=False)  
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)  


model_checkpoint_path = 'finetune_model_epoch49.pth'
model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))


model = model.to(device)

test_df = pd.read_csv('testset_values_50epoch.csv')


img_labels = pd.read_csv('data\per_scan_data.csv')



def find_label(img_name, img_labels):
   
    name_column = 'scan_name'

    found_row = img_labels[img_labels[name_column] == img_name]
    return found_row['status'].item()

labels = []
for i in range(len(test_df)):
    img_name = test_df[i]

    label = find_label(img_name, img_labels)
    if label == True:
        labels.append(1)
    else:
        labels.append(0)

all_slices_path = r'\\fsmresfiles.fsm.northwestern.edu\fsmresfiles\Ophthalmology\Mirza_Images\AMD\dAMD_GA\all_slices_3'

test_dataset = DataGen(test_df, labels, all_slices_path, image_height=1536, image_width=500)

test_dataloader = DataLoader(test_dataset, batch_size=32)


model.eval()

with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_dataloader):
        inputs, labels = inputs.float().to(device), labels.to(device)
        inputs = inputs.permute(0, 3, 1, 2).to(device)
        outputs = model(inputs).squeeze()

        preds = outputs.cpu().detach().numpy()
        pred_binary = np.where(preds >= 0.5, 1, 0)

        labels = labels.float()
        val_loss = criterion(outputs, labels)
        running_val_loss += val_loss.item()
        tn, fp, fn, tp = confusion_matrix(labels.cpu().numpy(), pred_binary, labels=[0,1]).ravel()
        TP_count += tp
        TN_count += tn
        FP_count += fp
        FN_count += fn
