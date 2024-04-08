import pickle
from torcheval.metrics import BinaryRecall, BinaryPrecision, BinaryAccuracy
import json
import skimage
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from TestSmokeDataset import SmokeDataset
from torchvision import transforms
import segmentation_models_pytorch as smp
from metrics import compute_iou, display_iou
#from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

with open('subsample.pkl', 'rb') as handle:
    data_dict = pickle.load(handle)

data_transforms = transforms.Compose([transforms.ToTensor()])

train_set = SmokeDataset(data_dict['train'], data_transforms)
val_set = SmokeDataset(data_dict['val'], data_transforms)
test_set = SmokeDataset(data_dict['test'], data_transforms)

print('there are {} training samples in this dataset'.format(len(train_set)))
print('there are {} testing samples in this dataset'.format(len(test_set)))

def save_test_results(data_fn, preds, dir_num):
    save_loc = '../plot_results/test_results/{}/'.format(dir_num)
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    data_fn = data_fn[0]
    coords_fn = data_fn.replace('/data/', '/coords/')
    if 'None' in data_fn:
        label = 0
    else:
        label = 1
    if preds >= .5:
        pred = 1
    else:
        pred = 0

    fn_info = {'data_fn': data_fn,
               'coords_fn': coords_fn,
               'label': label,
               'preds': pred,
               }

    json_object = json.dumps(fn_info, indent=4)
    with open(save_loc + "fn_info.json", "w") as outfile:
        outfile.write(json_object)

def test_model(dataloader, model, BCE_loss):
    model.eval()
    torch.set_grad_enabled(False)
    total_loss = 0.0
    recall_metric = BinaryRecall()
    accuracy_metric = BinaryAccuracy()
    precision_metric = BinaryPrecision()
    iou_dict= {'high': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}, 'medium': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}, 'low': {'int': 0, 'union':0, 'prev_int': 0, 'prev_union': 0}}
    max_num = 1000
    max_num = 0
    for idx, data in enumerate(dataloader):
        batch_data, batch_labels, data_fn = data
        batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
        preds = model(batch_data)
        preds = preds.squeeze(1)
        preds = torch.sigmoid(preds)
        loss = BCE_loss(preds, batch_labels).to(device)
        test_loss = loss.item()
        total_loss += test_loss
        batch_labels = batch_labels.int()
        recall_metric.update(preds, batch_labels)
        accuracy_metric.update(preds, batch_labels)
        precision_metric.update(preds, batch_labels)
        if len(data_fn)==1:
            if idx < max_num:
                print(idx)
                save_test_results(data_fn, preds.detach().to('cpu').numpy(), idx)
            else:
                break
    recall = recall_metric.compute()
    acc = accuracy_metric.compute()
    prec = precision_metric.compute()
    final_loss = total_loss/len(dataloader)
    print("Testing Loss: {}".format(round(final_loss,8)), flush=True)
    print("Testing Recall: {}".format(recall), flush=True)
    print("Testing Accuracy: {}".format(acc), flush=True)
    print("Testing Precision: {}".format(prec), flush=True)
    return final_loss

def val_model(dataloader, model, BCE_loss):
    model.eval()
    torch.set_grad_enabled(False)
    total_loss = 0.0

    iou_dict= {'high': {'int': 0, 'union':0}, 'medium': {'int': 0, 'union':0}, 'low': {'int': 0, 'union':0}}
    for data in dataloader:
        batch_data, batch_labels = data
        batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
        preds = model(batch_data)

        high_loss = BCE_loss(preds[:,0,:,:], batch_labels[:,0,:,:]).to(device)
        med_loss = BCE_loss(preds[:,1,:,:], batch_labels[:,1,:,:]).to(device)
        low_loss = BCE_loss(preds[:,2,:,:], batch_labels[:,2,:,:]).to(device)
        loss = 3*high_loss + 2*med_loss + low_loss
        #loss = high_loss + med_loss + low_loss
        test_loss = loss.item()
        total_loss += test_loss
        iou_dict= compute_iou(preds[:,0,:,:], batch_labels[:,0,:,:], 'high', iou_dict)
        iou_dict= compute_iou(preds[:,1,:,:], batch_labels[:,1,:,:], 'medium', iou_dict)
        iou_dict= compute_iou(preds[:,2,:,:], batch_labels[:,2,:,:], 'low', iou_dict)
    display_iou(iou_dict)
    final_loss = total_loss/len(dataloader)
    print("Validation Loss: {}".format(round(final_loss,8)), flush=True)
    return final_loss

def train_model(train_dataloader, val_dataloader, model, n_epochs, start_epoch, exp_num, BCE_loss):
    history = dict(train=[], val=[])
    best_loss = 10000.0

    for epoch in range(start_epoch, n_epochs):
        total_loss = 0.0
        print('--------------\nStarting Epoch: {}'.format(epoch), flush=True)
        model.train()
        torch.set_grad_enabled(True)
        #for batch_data, batch_labels in train_dataloader:
        for data in train_dataloader:
            batch_data, batch_labels = data
            batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
            #print(torch.isnan(batch_data).any())
            optimizer.zero_grad() # zero the parameter gradients
            preds = model(batch_data)
            high_loss = BCE_loss(preds[:,0,:,:], batch_labels[:,0,:,:]).to(device)
            med_loss = BCE_loss(preds[:,1,:,:], batch_labels[:,1,:,:]).to(device)
            low_loss = BCE_loss(preds[:,2,:,:], batch_labels[:,2,:,:]).to(device)
            loss = 3*high_loss + 2*med_loss + low_loss
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
            torch.save(checkpoint, '/scratch/alpine/mecr8410/semantic_segmentation_smoke/scripts/deep_learning/models/checkpoint_exp{}.pth'.format(exp_num))
            #torch.save(model, './scratch/alpine/mecr8410/semantic_segmentation_smoke/scripts/deep_learning/models/best_model.pth')
    print(history)
    return model, history

if len(sys.argv) < 2:
    print('\n YOU DIDNT SPECIFY EXPERIMENT NUMBER! ', flush=True)
if len(sys.argv) > 2:
    print("IN TEST MODE!")
    test_mode = sys.argv[2]
else:
    test_mode = False


exp_num = str(sys.argv[1])

with open('configs/exp{}.json'.format(exp_num)) as fn:
    hyperparams = json.load(fn)

use_ckpt = False
#use_ckpt = True
BATCH_SIZE = int(hyperparams["batch_size"])
BATCH_SIZE = 128
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
#test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=True, drop_last=True)

n_epochs = 100
start_epoch = 0

#model = resnet50(weights=ResNet50_Weights.DEFAULT)
#model.fc = nn.Linear(model.fc.in_features, 1)
model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model = model.to(device)

lr = hyperparams['lr']
optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)

BCE_loss = nn.BCEWithLogitsLoss()
if test_mode:
    print("IN TEST MODE!")
    checkpoint=torch.load('./models/checkpoint_exp{}_en.pth'.format(exp_num))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_model(test_loader, model, BCE_loss)

else:
    if use_ckpt == True:
        checkpoint=torch.load('./models/checkpoint_exp{}_en.pth'.format(exp_num))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    train_model(train_loader, val_loader, model, n_epochs, start_epoch, exp_num, BCE_loss)

