import pickle
import shutil
import glob
import json
import skimage
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from SmokeDataset import SmokeDataset
from torchvision import transforms
import segmentation_models_pytorch as smp

def get_file_list(dn_dir, idx):
    truth_file_list = []
    truth_file_list = glob.glob('{}truth/*/*/*_{}.tif'.format(dn_dir, idx))
    truth_file_list.sort()
    print(truth_file_list)
    data_file_list = [s.replace('truth','data') for s in truth_file_list]
    print('number of samples for idx:', len(truth_file_list))
    data_dict = {'find': {'truth': truth_file_list, 'data': data_file_list}}
    return data_dict

def run_model(dn_dir, idx):
    data_dict = get_file_list(dn_dir, idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    data_transforms = transforms.Compose([transforms.ToTensor()])

    test_set = SmokeDataset(data_dict['find'], data_transforms)

    print('there are {} images for this annotation'.format(len(test_set)))

    def get_best_file(dataloader, model, BCE_loss):
        model.eval()
        torch.set_grad_enabled(False)
        best_loss = 10000000.0
        losses = []
        best_truth_fn = None
        for idx, data in enumerate(dataloader):
            batch_data, batch_labels, truth_fn = data
            batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
            preds = model(batch_data)
            high_loss = BCE_loss(preds[:,0,:,:], batch_labels[:,0,:,:]).to(device)
            med_loss = BCE_loss(preds[:,1,:,:], batch_labels[:,1,:,:]).to(device)
            low_loss = BCE_loss(preds[:,2,:,:], batch_labels[:,2,:,:]).to(device)
            loss = 3*high_loss + 2*med_loss + low_loss
            losses.append(loss.item())
            if loss < best_loss:
                best_loss = loss
                best_truth_fn = truth_fn
        print("Losses: {}".format(losses, flush=True))
        best_loss_idx = losses.index(min(losses))
        print('best loss index', best_loss_idx)
        if truth_fn:
            print(truth_fn)
        if truth_fn:
            return truth_fn[0]
        return None


    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    model = smp.DeepLabV3Plus(
            encoder_name="timm-efficientnet-b2",
            encoder_weights="imagenet", # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3, # model input channels
            classes=3, # model output channels
    )
    model = model.to(device)

    #lr = hyperparams['lr']
    #optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
    BCE_loss = nn.BCEWithLogitsLoss()
    chkpt_path = './model/chkpt.pth'
    checkpoint = torch.load(chkpt_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    best_fn = get_best_file(test_loader, model, BCE_loss)
    return best_fn

def get_indices(dn_dir):
    file_list = glob.glob('{}truth/*/*/*'.format(dn_dir))
    indices = []
    for fn in file_list:
        idx = fn.split('_')[-1].split('.')[0]
        indices.append(idx)
    indices = list(set(indices))
    indices.sort()
    return indices

def mv_files(truth_src, yr_dn, idx):
    coords_src = truth_src.replace('truth','coords')
    data_src = truth_src.replace('truth','data')
    truth_dst = truth_src.replace('temp_data/{}'.format(yr_dn),'')
    coords_dst = coords_src.replace('temp_data/{}'.format(yr_dn),'')
    data_dst = data_src.replace('temp_data/{}'.format(yr_dn),'')
    shutil.copyfile(truth_src, truth_dst)
    shutil.copyfile(coords_src, coords_dst)
    shutil.copyfile(data_src, data_dst)
    if os.path.exists(truth_dst) and os.path.exists(coords_dst) and os.path.exists(data_dst):
        for f in glob.glob(dn_dir+'*/*/*/*_{}.tif'.format(idx)):
            os.remove(f)

def main(dn, yr):
    global dn_dir
    yr_dn = yr+dn
    dn_dir = '/scratch/alpine/mecr8410/semantic_segmentation_smoke/filtered_data/temp_data/{}/'.format(yr_dn)
    indices = get_indices(dn_dir)
    for idx in indices:
        best_fn = run_model(dn_dir, idx)
        mv_files(best_fn, yr_dn, idx)

if __name__ == '__main__':
    dn = sys.argv[1]
    yr = sys.argv[2]
    main(dn, yr)
