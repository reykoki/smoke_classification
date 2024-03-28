import pickle
import sys
import json
from torchmetrics import JaccardIndex
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchsummary import summary
from SmokeDataset import SmokeDataset
from torchvision import transforms
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

with open('subsample.pkl', 'rb') as handle:
    data_dict = pickle.load(handle)
#print(data_dict)

data_transforms = transforms.Compose([transforms.ToTensor()])

train_set = SmokeDataset(data_dict['train'], data_transforms)
val_set = SmokeDataset(data_dict['val'], data_transforms)
test_set = SmokeDataset(data_dict['test'], data_transforms)

print('there are {} training samples in this dataset'.format(len(train_set)))

def val_model(dataloader, model, BCE_loss):
    model.eval()
    torch.set_grad_enabled(False)
    total_loss = 0.0
    for data in dataloader:
        batch_data, batch_labels = data
        batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
        preds = model(batch_data)
        loss = BCE_loss(preds, batch_labels).to(device)
        test_loss = loss.item()
        total_loss += test_loss
    final_loss = total_loss/len(dataloader)
    print("Validation Loss: {}".format(round(final_loss,8)), flush=True)
    return final_loss

def train_model(train_dataloader, val_dataloader, model, n_epochs, start_epoch, exp_num):
    history = dict(train=[], val=[])
    best_loss = 10000.0
    BCE_loss = nn.BCEWithLogitsLoss()

    for epoch in range(start_epoch, n_epochs):
        total_loss = 0.0
        print('--------------\nStarting Epoch: {}'.format(epoch), flush=True)
        model.train()
        torch.set_grad_enabled(True)
        #for batch_data, batch_labels in train_dataloader:
        for data in train_dataloader:
            batch_data, batch_labels = data
            batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
            optimizer.zero_grad() # zero the parameter gradients
            preds = model(batch_data)
            loss = BCE_loss(preds, batch_labels).to(device)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            total_loss += train_loss
        epoch_loss = total_loss/len(train_dataloader)
        print("Training Loss:   {0}".format(round(epoch_loss,8), epoch+1), flush=True)
        val_loss = val_model(val_dataloader, model, BCE_loss)
        history['val'].append(val_loss)
        history['train'].append(epoch_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }
            torch.save(checkpoint, './models/checkpoint_exp{}_n.pth'.format(exp_num))
    print(history)
    return model, history

if len(sys.argv) < 2:
    print('\n YOU DIDNT SPECIFY EXPERIMENT NUMBER! ', flush=True)
if len(sys.argv) > 2:
    test_mode = sys.argv[2]

exp_num = str(sys.argv[1])

with open('configs/exp{}.json'.format(exp_num)) as fn:
    hyperparams = json.load(fn)

use_ckpt = False
#use_ckpt = True
BATCH_SIZE = int(hyperparams["batch_size"])
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

n_epochs = 100
start_epoch = 0

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model = model.to(device)

lr = hyperparams['lr']
optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
if use_ckpt == True:
    checkpoint=torch.load('./models/checkpoint_exp{}_n.pth'.format(exp_num))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']

train_model(train_loader, val_loader, model, n_epochs, start_epoch, exp_num)

