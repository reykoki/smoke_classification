import shutil
import glob
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from SmokeDataset import SmokeDataset
from torchvision import transforms
import segmentation_models_pytorch as smp
import matplotlib
from MakeDirs import MakeDirs
import pyproj
import ray
import sys
import logging
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pyresample import create_area_def
import geopandas
import pandas as pd
from satpy import Scene
from PIL import Image, ImageOps
import os
import random
import glob
import skimage
from datetime import datetime
import numpy as np
import time
import s3fs
import pytz
import multiprocessing
import shutil
import wget
from suntime import Sun
from datetime import timedelta

def get_file_list(idx):
    truth_file_list = []
    truth_file_list = glob.glob('{}truth/*/*/*_{}.tif'.format(dn_dir, idx))
    truth_file_list.sort()
    print(truth_file_list)
    data_file_list = [s.replace('truth','data') for s in truth_file_list]
    print('number of samples for idx:', len(truth_file_list))
    data_dict = {'find': {'truth': truth_file_list, 'data': data_file_list}}
    return data_dict

def compute_iou(preds, truths):
    densities = ['high', 'medium', 'low']
    intersection = 0
    union = 0
    for idx, level in enumerate(densities):
        pred = preds[:,idx,:,:]
        true = truths[:,idx,:,:]
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5) * 1
        intersection += (pred + true == 2).sum()
        union += (pred + true >= 1).sum()
    try:
        iou = intersection / union
        return iou
    except Exception as e:
        print(e)
    return 0

def run_model(idx):
    data_dict = get_file_list(idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    data_transforms = transforms.Compose([transforms.ToTensor()])

    test_set = SmokeDataset(data_dict['find'], data_transforms)

    print('there are {} images for this annotation'.format(len(test_set)))

    def get_best_file(dataloader, model, BCE_loss):
        model.eval()
        torch.set_grad_enabled(False)
        # iou has to be more than .01
        best_loss = .01
        losses = []
        best_truth_fn = None
        for idx, data in enumerate(dataloader):
            batch_data, batch_labels, truth_fn = data
            batch_data, batch_labels = batch_data.to(device, dtype=torch.float), batch_labels.to(device, dtype=torch.float)
            preds = model(batch_data)
            #high_loss = BCE_loss(preds[:,0,:,:], batch_labels[:,0,:,:]).to(device)
            #med_loss = BCE_loss(preds[:,1,:,:], batch_labels[:,1,:,:]).to(device)
            #low_loss = BCE_loss(preds[:,2,:,:], batch_labels[:,2,:,:]).to(device)
            #loss = 3*high_loss + 2*med_loss + low_loss
            #losses.append(loss.item())
            loss = compute_iou(preds, batch_labels)
            losses.append(loss)
            #if loss < best_loss:
            if loss > best_loss:
                best_loss = loss
                best_truth_fn = truth_fn
        print("Losses: {}".format(losses, flush=True))
        best_loss_idx = losses.index(max(losses))
        print('best loss index', best_loss_idx)
        if best_truth_fn:
            print('best_truth_fn: ', best_truth_fn)
            return  best_truth_fn[0]

        fn = truth_fn[0].split('/')[-1]
        bad_fn = "/scratch/alpine/mecr8410/semantic_segmentation_smoke/filtered_data/bad_fns/{}".format(fn)
        with open(bad_fn, 'w') as fp:
                pass
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
    chkpt_path = '/scratch/alpine/mecr8410/semantic_segmentation_smoke/find_best/model/chkpt.pth'
    checkpoint = torch.load(chkpt_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    best_fn = get_best_file(test_loader, model, BCE_loss)
    return best_fn

def get_indices():
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

def find_best_data(yr, dn):
    yr_dn = yr+dn
    indices = get_indices()
    print(indices)
    for idx in indices:
        best_fn = run_model(idx)
        if best_fn:
            mv_files(best_fn, yr_dn, idx)

def doesnt_already_exists(yr, fn_heads, idx, density):
    final_dest = "/scratch/alpine/mecr8410/semantic_segmentation_smoke/filtered_data/"
    for fn_head in fn_heads:
        file_list = glob.glob('{}truth/{}/{}/{}_{}.tif'.format(final_dest, yr, density, fn_head, idx))
        if len(file_list) > 0:
            print("FILE THAT ALREADY EXIST:", file_list[0], flush=True)
            return False
        file_list = glob.glob('{}bad_fns/{}_{}.tif'.format(final_dest, fn_head, idx))
        if len(file_list) > 0:
            print("THIS ANNOTATION FAILED:", file_list[0], flush=True)
            return False
    return True

def check_bounds(x, y, bounds):
    if bounds['minx'] > np.min(x) and bounds['maxx'] < np.max(x) and bounds['miny'] > np.min(y) and bounds['maxy'] < np.max(y):
        return True
    else:
        return False

def pick_temporal_smoke(smoke_shape, t_0, t_f):
    use_idx = []
    bounds = smoke_shape.bounds
    for idx, row in smoke_shape.iterrows():
        end = row['End']
        start = row['Start']
        # the ranges overlap if:
        if t_0-timedelta(minutes=10)<= end and start-timedelta(minutes=10) <= t_f:
            use_idx.append(idx)
    rel_smoke = smoke_shape.loc[use_idx]
    return rel_smoke

def reshape(A, idx, size=256):
    print('before reshape: ', np.sum(A))
    d = int(size/2)
    A =A[idx[0]-d:idx[0]+d, idx[1]-d:idx[1]+d]
    print('after reshape: ', np.sum(A))
    return A

def get_norm(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def save_data(R, G, B, idx, fn_data, size=256):
    R = reshape(R, idx, size)
    G = reshape(G, idx, size)
    B = reshape(B, idx, size)
    layers = np.dstack([R, G, B])
    total = np.sum(R).compute() + np.sum(G).compute() + np.sum(B).compute()
    print('========')
    print("total:", total)
    print('========')
    #print('SUM TOTAL: ', int(np.sum((total))))
    if total > 100 and total < 1e5:
        skimage.io.imsave(fn_data, layers)
        return True
    return False
    #else:
    #    return False

def normalize(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

def get_rand_center(idx, rand_xy):
    x_o = idx[0] + rand_xy[0]
    y_o = idx[1] + rand_xy[1]
    return (x_o, y_o)

def find_closest_pt(pt_x, pt_y, x, y):
    x_diff = np.abs(x - pt_x)
    y_diff = np.abs(y - pt_y)
    x_diff2 = x_diff**2
    y_diff2 = y_diff**2
    sum_diff = x_diff2 + y_diff2
    dist = sum_diff**(1/2)
    idx = np.unravel_index(dist.argmin(), dist.shape)
    #if distance is less than 1km away
    if np.min(dist) < 1000:
        return idx
    else:
        print("not close enough")
        return None

def get_centroid(center, x, y, img_shape, rand_xy):
    pt_x = center.x
    pt_y = center.y
    idx = find_closest_pt(pt_x, pt_y, x, y)
    if idx:
        rand_idx = get_rand_center(idx, rand_xy)
        return idx, rand_idx
    else:
        return None, None
def plot_coords(lat, lon, idx, tif_fn):
    lat_coords = reshape(lat, idx)
    lon_coords = reshape(lon, idx)
    coords_layers = np.dstack([lat_coords, lon_coords])
    skimage.io.imsave(tif_fn, coords_layers)
    #print(coords_layers)

def plot_truth(x, y, lcc_proj, smoke, png_fn, idx, img_shape):
    fig = plt.figure(figsize=(img_shape[2]/100, img_shape[1]/100), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection=lcc_proj)
    smoke.plot(ax=ax, facecolor='black')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig(png_fn, dpi=100)
    plt.close(fig)
    img = Image.open(png_fn)
    bw = img.convert('1')
    bw = ImageOps.invert(bw)

    truth = np.asarray(bw).astype('i')
    truth = reshape(truth, idx)
    os.remove(png_fn)
    return truth

def get_truth(x, y, lcc_proj, smoke, idx, png_fn, tif_fn, center, img_shape):

    low_smoke = smoke.loc[smoke['Density'] == 'Light']
    med_smoke = smoke.loc[smoke['Density'] == 'Medium']
    high_smoke = smoke.loc[smoke['Density'] == 'Heavy']

    # high = [1,1,1], med = [0, 1, 1], low = [0, 0, 1]
    low_truth = plot_truth(x, y, lcc_proj, low_smoke, png_fn, idx, img_shape)
    med_truth = plot_truth(x, y, lcc_proj, med_smoke, png_fn, idx, img_shape)
    high_truth = plot_truth(x, y, lcc_proj, high_smoke, png_fn, idx, img_shape)
    low_truth += med_truth + high_truth
    low_truth = np.clip(low_truth, 0, 1)
    med_truth += high_truth
    med_truth = np.clip(med_truth, 0, 1)

    truth_layers = np.dstack([high_truth, med_truth, low_truth])
    print('---------------------------')
    print(tif_fn)
    print(np.sum(truth_layers))
    print('---------------------------')
    if np.sum(truth_layers) > 0:
        skimage.io.imsave(tif_fn, truth_layers)
        return True
    return False

def get_extent(center):
    x0 = center.x - 2.5e5
    y0 = center.y - 2.5e5
    x1 = center.x + 2.5e5
    y1 = center.y + 2.5e5
    return [x0, y0, x1, y1]

def get_scn(fns, extent):
    scn = Scene(reader='abi_l1b', filenames=fns)

    scn.load(['cimss_true_color_sunz_rayleigh'], generate=False)
    my_area = create_area_def(area_id='lccCONUS',
                              description='Lambert conformal conic for the contiguous US',
                              projection="+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs",
                              resolution=1000,
                              #full us
                              #area_extent=[-2.4e6, -1.5e6, 2.3e6, 1.4e6])
                              #western us
                              #area_extent=[-2.4e6, -1.5e6, 3.5e5, 1.4e6])
                              area_extent=extent)

    new_scn = scn.resample(my_area)
    return scn, new_scn

def get_get_scn(sat_fns, extent, sleep_time=0):
    time.sleep(sleep_time)
    old_scn, tmp_scn = get_scn(sat_fns, extent)
    return old_scn, tmp_scn

def create_data_truth(sat_fns, smoke, idx0, yr, density, rand_xy):
    print('idx: ', idx0)
    fn_head = sat_fns[0].split('C01_')[-1].split('.')[0].split('_c2')[0]

    lcc_str = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
    lcc_proj = pyproj.CRS.from_user_input(lcc_str)
    smoke_lcc = smoke.to_crs(lcc_proj)
    centers = smoke_lcc.centroid
    center = centers.loc[idx0]
    try:
        extent = get_extent(center)
    except:
        return fn_head

    try:
        old_scn, scn = get_get_scn(sat_fns, extent)
    except:
        if 'G17' in sat_fns[0]:
            print('G17 data not done downloading!')
            print('wait 15 seconds')
            try:
                old_scn, scn = get_get_scn(sat_fns, extent, 15)
            except:
                print('G17 data STILL not done downloading!')
                print('wait 60 seconds')
                try:
                    old_scn, scn = get_get_scn(sat_fns, extent, 60)
                except:
                    try:
                        print('wait ANOTHER 120 seconds')
                        old_scn, scn = get_get_scn(sat_fns, extent, 120)
                    except:
                        print('{} wouldnt download, moving on'.format(sat_fns[0]))
                        return fn_head
        else:
            print('G16 data not done downloading!')
            print('wait 10 seconds')
            try:
                old_scn, scn = get_get_scn(sat_fns, extent, 15)
            except:
                print('G16 data STILL not done downloading!')
                print('wait 60 seconds')
                try:
                    old_scn, scn = get_get_scn(sat_fns, extent, 60)
                except:
                    print('{} wouldnt download, moving on'.format(sat_fns[0]))
                    return fn_head

    lcc_proj = scn['cimss_true_color_sunz_rayleigh'].attrs['area'].to_cartopy_crs()
    smoke_lcc = smoke.to_crs(lcc_proj)
    scan_start = pytz.utc.localize(scn['cimss_true_color_sunz_rayleigh'].attrs['start_time'])
    scan_end = pytz.utc.localize(scn['cimss_true_color_sunz_rayleigh'].attrs['end_time'])
    rel_smoke = pick_temporal_smoke(smoke_lcc, scan_start, scan_end)

    # make sure the smoke shape is within the bounds of the
    x = scn['cimss_true_color_sunz_rayleigh'].coords['x']
    y = scn['cimss_true_color_sunz_rayleigh'].coords['y']
    lon, lat = scn['cimss_true_color_sunz_rayleigh'].attrs['area'].get_lonlats()

    corr_data = scn.save_dataset('cimss_true_color_sunz_rayleigh', compute=False)
    img_shape = scn['cimss_true_color_sunz_rayleigh'].shape

    R = corr_data[0][0]
    G = corr_data[0][1]
    B = corr_data[0][2]

    R = get_norm(R)
    G = get_norm(G)
    B = get_norm(B)

    xx = np.tile(x, (len(y),1))
    yy = np.tile(y, (len(x),1)).T

    cent, idx = get_centroid(center, xx, yy, img_shape, rand_xy)
#    print(center)
#    print(cent)
#    print(idx)

    if cent:
        png_fn_truth = dn_dir + 'temp_png/truth_' + fn_head + '_{}'.format(idx0) + '.png'
        tif_fn_truth = dn_dir + 'truth/{}/{}/{}_{}.tif'.format(yr, density, fn_head, idx0)
        print(tif_fn_truth)
        tif_fn_data = dn_dir + 'data/{}/{}/{}_{}.tif'.format(yr, density, fn_head, idx0)
        tif_fn_coords = dn_dir + 'coords/{}/{}/{}_{}.tif'.format(yr, density, fn_head, idx0)
        data_saved = save_data(R, G, B, idx, tif_fn_data)
        if data_saved:
            truth_saved  = get_truth(x, y, lcc_proj, rel_smoke, idx, png_fn_truth, tif_fn_truth, center, img_shape)
            if truth_saved:
                plot_coords(lat, lon, idx, tif_fn_coords)
    return fn_head

def closest_to_sunrise(st,et,actual_sunrise,bounds, density):
    west_lon = bounds['maxx']
    #print("WEST_LON: ", west_lon)
    if west_lon > -85:
        #sat = '16'
        #delay_time = 30
        return '16', None
    else:
        delay_time = 1.1 * west_lon + 209
        sat = '17'
    sunrise = actual_sunrise + timedelta(minutes=delay_time)
    #print('effective sunrise: ', sunrise)
    #print('actual sunrise: ', actual_sunrise)
    if st - sunrise > timedelta(hours=2.5) and density == 'Light':
        print('more than 2.5 hrs from sunrise and sunset')
        return '17', None
    elif et == st:
        return sat, st
    elif st >= sunrise:
        return sat, st
    elif st < sunrise and et >= sunrise:
        return sat, sunrise
    elif et <= sunrise and et >= actual_sunrise:
        return sat, et
    else:
        print('THERE IS AN ERROR FOR SUNRISE')
        print('et: ', et)
        print('st: ', st)
        print('sunrise: ', sunrise)
        return None, None

def closest_to_sunset(st, et, sunset, density):
    sunset = sunset - timedelta(minutes=45)
    if sunset - et > timedelta(hours=2.5) and density == 'Light':
        print('more than 2.5 hrs from sunrise and sunset')
        return '16', None
    elif et == st or et <= sunset:
        return '16', et
    elif et > sunset and st <= sunset:
        return '16', sunset
    else:
        print('THERE IS AN ERROR FOR SUNSET')
        print('et: ', et)
        print('st: ', st)
        print('sunset: ', sunset)
        return None, None

def closer_east_west(bounds, st, et):
    # if closer to west coast:
    if bounds['minx'] < -98:
        sat = '16'
        best_time = st
    else:
        sat = '17'
        best_time = et
    return sat, best_time

def get_ss(bounds, st, et):
    try:
        east = Sun(bounds['miny'], bounds['minx'])
        sr_dt_st = {-1: abs(st - east.get_sunset_time(st+timedelta(days=-1))),
                     0: abs(st - east.get_sunset_time(st+timedelta(days=0))),
                     1: abs(st - east.get_sunset_time(st+timedelta(days=1)))}
        sr_dt_et = {-1: abs(et - east.get_sunset_time(et+timedelta(days=-1))),
                     0: abs(et - east.get_sunset_time(et+timedelta(days=0))),
                     1: abs(et - east.get_sunset_time(et+timedelta(days=1)))}
    except Exception as e:
        print(e)
        try:
            # actually west
            east = Sun(bounds['maxy'], bounds['maxx'])
            sr_dt_st = {-1: abs(st - east.get_sunset_time(st+timedelta(days=-1))),
                         0: abs(st - east.get_sunset_time(st+timedelta(days=0))),
                         1: abs(st - east.get_sunset_time(st+timedelta(days=1)))}
            sr_dt_et = {-1: abs(et - east.get_sunset_time(et+timedelta(days=-1))),
                         0: abs(et - east.get_sunset_time(et+timedelta(days=0))),
                         1: abs(et - east.get_sunset_time(et+timedelta(days=1)))}
        except Exception as e:
            print(e)
            return None, None
    st_dt = min(sr_dt_st, key=sr_dt_st.get)
    et_dt = min(sr_dt_et, key=sr_dt_et.get)
    if sr_dt_st[st_dt] > sr_dt_et[et_dt]:
        return east.get_sunset_time(et+timedelta(days=et_dt)), sr_dt_et[et_dt]
    return east.get_sunset_time(et+timedelta(days=st_dt)), sr_dt_st[st_dt]

def get_sr(bounds, st, et):
    try:
        west = Sun(bounds['maxy'], bounds['maxx'])
        sr_dt_st = {-1: abs(st - west.get_sunrise_time(st+timedelta(days=-1))),
                     0: abs(st - west.get_sunrise_time(st+timedelta(days=0))),
                     1: abs(st - west.get_sunrise_time(st+timedelta(days=1)))}
        sr_dt_et = {-1: abs(et - west.get_sunrise_time(et+timedelta(days=-1))),
                     0: abs(et - west.get_sunrise_time(et+timedelta(days=0))),
                     1: abs(et - west.get_sunrise_time(et+timedelta(days=1)))}
    except Exception as e:
        print(e)
        try:
            #actually east
            west = Sun(bounds['miny'], bounds['minx'])
            sr_dt_st = {-1: abs(st - west.get_sunrise_time(st+timedelta(days=-1))),
                         0: abs(st - west.get_sunrise_time(st+timedelta(days=0))),
                         1: abs(st - west.get_sunrise_time(st+timedelta(days=1)))}
            sr_dt_et = {-1: abs(et - west.get_sunrise_time(et+timedelta(days=-1))),
                         0: abs(et - west.get_sunrise_time(et+timedelta(days=0))),
                         1: abs(et - west.get_sunrise_time(et+timedelta(days=1)))}
        except Exception as e:
            print(e)
            return None, None
    st_dt = min(sr_dt_st, key=sr_dt_st.get)
    et_dt = min(sr_dt_et, key=sr_dt_et.get)
    if sr_dt_st[st_dt] > sr_dt_et[et_dt]:
        return west.get_sunrise_time(et+timedelta(days=et_dt)), sr_dt_et[et_dt]
    return west.get_sunrise_time(st+timedelta(days=st_dt)), sr_dt_st[st_dt]

def get_best_time(st, et, bounds, density):
    sunrise, sr_dt = get_sr(bounds, st, et)
    sunset, ss_dt  = get_ss(bounds, st, et)
    # no sunrise or sunset (assuming sun isnt setting)
    if sr_dt == None or ss_dt == None:
        sat, best_time = closer_east_west(bounds, st, et)
    # times are closer to sunset
    elif sr_dt >= ss_dt:
        sat, best_time = closest_to_sunset(st,et,sunset, density)
    else:
        sat, best_time = closest_to_sunrise(st,et,sunrise,bounds,density)
    return sat, best_time


def get_closest_file(fns, best_time, sat_num):
    diff = timedelta(days=100)
    use_fns = []
    for fn in fns:
        starts = []
        if 'C01' in fn:
            s_e = fn.split('_')[3:5]
            start = s_e[0]
            end = s_e[1]
            C02_fn = 'C02_G{}_{}_{}'.format(sat_num, start, end)
            C03_fn = 'C03_G{}_{}_{}'.format(sat_num, start, end)
            for f in fns:
                if C02_fn in f:
                   C02_fn = f
                elif C03_fn in f:
                   C03_fn = f
            if 'nc' in C02_fn and 'nc' in C03_fn:
                start = s_e[0][1:-3]
                s_dt = pytz.utc.localize(datetime.strptime(start, '%Y%j%H%M'))
                if diff > abs(s_dt - best_time):
                    diff = abs(s_dt - best_time)
                    use_fns = [fn, C02_fn, C03_fn]
    return use_fns

def get_smoke(yr, month, day):
    fn = 'hms_smoke{}{}{}.zip'.format(yr, month, day)
    print('DOWNLOADING SMOKE:')
    print(fn)
    out_dir = dn_dir + 'smoke/'
    if os.path.exists(out_dir+fn):
        print("{} already exists".format(fn))
        smoke_shape_fn = dn_dir + 'smoke/hms_smoke{}{}{}.shp'.format(yr,month,day)
        smoke = geopandas.read_file(smoke_shape_fn)
        return smoke
    else:
        try:
            url = 'https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/Shapefile/{}/{}/{}'.format(yr, month, fn)
            filename = wget.download(url, out=out_dir)
            shutil.unpack_archive(filename, out_dir)
            smoke_shape_fn = dn_dir + 'smoke/hms_smoke{}{}{}.shp'.format(yr,month,day)
            smoke = geopandas.read_file(smoke_shape_fn)
            return smoke
        except Exception as e:
            print(e)
            print('NO SMOKE DATA FOR THIS DATE')
            return None

def get_sat_files(smoke_row):
    smoke = smoke_row['smoke']
    idx = smoke_row['idx']
    bounds = smoke_row['bounds']
    density = smoke_row['density']
    row = smoke.loc[idx]

    fs = s3fs.S3FileSystem(anon=True)
    s_dt = row['Start']
    e_dt = row['End']
    tt = s_dt.timetuple()
    dn = tt.tm_yday
    dn = str(dn).zfill(3)

    all_fn_heads = []
    all_sat_fns = []
    t = s_dt
    time_list = [t]
    while t < e_dt:
        t += timedelta(minutes=15)
        time_list.append(t)
    sat_num, best_time = get_best_time(s_dt, e_dt, bounds, density)
    print('best time; ', best_time)
    if best_time:
        if best_time not in time_list:
            time_list.append(best_time)
    for curr_time in time_list:
        hr = curr_time.hour
        hr = str(hr).zfill(2)
        yr = curr_time.year
        dn = curr_time.strftime('%j')

        full_filelist = []
        if sat_num == '17':
            view = 'F'
        else:
            view = 'C'
        try:
            full_filelist = fs.ls("noaa-goes{}/ABI-L1b-Rad{}/{}/{}/{}/".format(sat_num, view, yr, dn, hr))
        except Exception as e:
            print("ERROR WITH FS LS")
            print(sat_num, view, yr, dn, hr)
            print(e)
        if len(full_filelist) == 0:
            if yr <= 2018:
                sat_num = '16'
                print("YOU WANTED 17 BUT ITS NOT LAUNCHED")
            elif yr >= 2022:
                sat_num = '18'
            try:
                full_filelist = fs.ls("noaa-goes{}/ABI-L1b-Rad{}/{}/{}/{}/".format(sat_num, view, yr, dn, hr))
            except Exception as e:
                print("ERROR WITH FS LS")
                print(sat_num, view, yr, dn, hr)
                print(e)
        if len(full_filelist) > 0:
            sat_fns = get_closest_file(full_filelist, curr_time, sat_num)
            if sat_fns:
                fn_head = sat_fns[0].split('C01_')[-1].split('.')[0].split('_c2')[0]
                all_fn_heads.append(fn_head)
                all_sat_fns.append(sat_fns)

    if len(all_sat_fns)>0:
        all_sat_fns = [list(item) for item in set(tuple(row) for row in all_sat_fns)]
        all_fn_heads = list(set(all_fn_heads))
        return all_fn_heads, all_sat_fns
    return None, None

def get_file_locations(use_fns):
    file_locs = []
    fs = s3fs.S3FileSystem(anon=True)
    goes_dir = dn_dir + 'goes_temp/'
    for file_path in use_fns:
        fn = file_path.split('/')[-1]
        dl_loc = goes_dir+fn
        file_locs.append(dl_loc)
        if os.path.exists(dl_loc):
            print("{} already exists".format(fn))
        else:
            print('downloading {}'.format(fn))
            fs.get(file_path, dl_loc)
    return file_locs

@ray.remote
def iter_rows(smoke_row):
    smoke = smoke_row['smoke']
    idx = smoke_row['idx']
    bounds = smoke_row['bounds']
    density = smoke_row['density']
    yr = smoke_row['Start'].strftime('%Y')
    use_fns = smoke_row['sat_fns']
    rand_xy = smoke_row['rand_xy']
    file_locs = get_file_locations(use_fns)

    if len(file_locs) > 0:
        fns = create_data_truth(file_locs, smoke, idx, yr, density, rand_xy)
        return fns
    else:
        print('ERROR NO FILES FOUND FOR best_time: ', best_time)

def run_no_ray(smoke_rows):
    for smoke_row in smoke_rows:
        fn_heads = iter_rows(smoke_row)
    return fn_heads

def run_remote(smoke_rows):
    try:
        fn_heads = ray.get([iter_rows.remote(smoke_row) for smoke_row in smoke_rows])
        return fn_heads
    except Exception as e:
        print("ERROR WITH RAY GET")
        print(e)
        print(smoke_rows)
        fn_heads = []
        for smoke_row in smoke_rows:
            sat_fns = smoke_row['sat_fns']
            fn_head = sat_fns[0].split('C01_')[-1].split('.')[0].split('_c2')[0]
            fn_heads.append(fn_head)
        return fn_heads

# we need a consistent random shift in the dataset per each annotation
def get_random_xy(size=256):
    d = int(size/4)
    x_shift = random.randint(int(-1*d), d)
    y_shift = random.randint(int(-1*d), d)
    return (x_shift, y_shift)
# create object that contians all the smoke information needed
def create_smoke_rows(smoke):
    fmt = '%Y%j %H%M'
    smoke_fns = []
    bounds = smoke.bounds
    smoke_rows = []
    smoke_lcc = smoke.to_crs(3857)
    smoke_lcc_area = smoke_lcc['geometry'].area
    for idx, row in smoke.iterrows():
        rand_xy = get_random_xy()
        ts_start = pytz.utc.localize(datetime.strptime(smoke.loc[idx]['Start'], fmt))
        ts_end = pytz.utc.localize(datetime.strptime(smoke.loc[idx]['End'], fmt))
        print(ts_start)
        row_yr = ts_start.strftime('%Y')
        smoke.at[idx, 'Start'] =  ts_start
        smoke.at[idx, 'End'] =  ts_end
        if row['Density'] == 'Medium' or row['Density'] == 'Heavy':
            if smoke_lcc_area.loc[idx] > 1e7 and smoke_lcc_area.loc[idx] < 4e11:
                print('high or med idx:', idx)
                print("density area:", smoke_lcc.loc[idx]['geometry'].area)
                smoke_row = {'smoke': smoke, 'idx': idx, 'bounds': bounds.loc[idx], 'density': row['Density'], 'sat_fns': [], 'Start': ts_start, 'rand_xy': rand_xy}
                fn_heads, sat_fns = get_sat_files(smoke_row)
                if sat_fns:
                    if doesnt_already_exists(row_yr, fn_heads, idx, row['Density']):
                        for sat_fn_entry in sat_fns:
                            smoke_row = {'smoke': smoke, 'idx': idx, 'bounds': bounds.loc[idx], 'density': row['Density'], 'sat_fns': sat_fn_entry , 'Start': ts_start, 'rand_xy': rand_xy}
                            smoke_rows.append(smoke_row)
            else:
                print('high or med idx:', idx)
                print('too high or low area:', smoke_lcc_area.loc[idx])

        elif row['Density'] == 'Light' and smoke_lcc_area.loc[idx] > 1e8 and smoke_lcc_area.loc[idx] < 4e10:
            print('light idx:', idx)
            print("density area:", smoke_lcc_area.loc[idx])
            smoke_row = {'smoke': smoke, 'idx': idx, 'bounds': bounds.loc[idx], 'density': row['Density'], 'sat_fns': [], 'Start': ts_start, 'rand_xy': rand_xy}
            fn_heads, sat_fns = get_sat_files(smoke_row)
            if sat_fns:
                if doesnt_already_exists(row_yr, fn_heads, idx, row['Density']):
                    for sat_fn_entry in sat_fns:
                        smoke_row = {'smoke': smoke, 'idx': idx, 'bounds': bounds.loc[idx], 'density': row['Density'], 'sat_fns': sat_fn_entry , 'Start': ts_start, 'rand_xy': rand_xy}
                        smoke_rows.append(smoke_row)
        else:
            print('wrong sized light idx:', idx)
            print('too high or low area:', smoke_lcc_area.loc[idx])

    return smoke_rows

# remove large satellite files and the tif files created during corrections
def remove_files(fn_heads):
    fn_heads = list(set(fn_heads))
    print("REMOVING FILES")
    print(fn_heads)
    for head in fn_heads:
        for fn in glob.glob(dn_dir + 'goes_temp/*{}*'.format(head)):
            os.remove(fn)
        s = head.split('s')[1][:13]
        dt = pytz.utc.localize(datetime.strptime(s, '%Y%j%H%M%S'))
        tif_fn = glob.glob('cimss_true_color_sunz_rayleigh_{}{}{}_{}{}{}.tif'.format(dt.strftime('%Y'), dt.strftime('%m'), dt.strftime('%d'), dt.strftime('%H'), dt.strftime('%M'), dt.strftime('%S')))
        if tif_fn:
            os.remove(tif_fn[0])

# analysts can only label data that is taken during the daytime, we want to filter for geos data that was within the timeframe the analysts are looking at
def iter_smoke(date):

    dn = date[0]
    yr = date[1]
    s = '{}/{}'.format(yr, dn)
    fmt = '%Y/%j'
    dt = pytz.utc.localize(datetime.strptime(s, fmt))
    month = dt.strftime('%m')
    day = dt.strftime('%d')
    print('------')
    print(dt)
    print('------')
    smoke = get_smoke(yr, month, day)

    if smoke is not None:
        smoke_rows = create_smoke_rows(smoke)
        print(smoke_rows)
        ray_dir = "/scratch/alpine/mecr8410/tmp/{}{}".format(yr,dn)
        if not os.path.isdir(ray_dir):
            os.mkdir(ray_dir)
        ray.init(num_cpus=8, _temp_dir=ray_dir, include_dashboard=False, ignore_reinit_error=True, dashboard_host='127.0.0.1')
        fn_heads = run_remote(smoke_rows)
        #fn_heads = run_no_ray(smoke_rows)
        ray.shutdown()
        shutil.rmtree(ray_dir)
        find_best_data(yr, dn)
        if fn_heads:
            remove_files(fn_heads)


def main(start_dn, end_dn, yr):
    global dn_dir
    dates = []
    dns = list(range(int(start_dn), int(end_dn)+1))
    for dn in dns:
        dn = str(dn).zfill(3)
        dates.append([dn, yr])
    for date in dates:
        dn_dir = '/scratch/alpine/mecr8410/semantic_segmentation_smoke/filtered_data/temp_data/{}{}/'.format(date[1], date[0])
        if not os.path.isdir(dn_dir):
            os.mkdir(dn_dir)
            MakeDirs(dn_dir, yr)
        start = time.time()
        print(date)
        iter_smoke(date)
        shutil.rmtree(dn_dir)
        print("Time elapsed for day {}: {}s".format(date, int(time.time() - start)), flush=True)

if __name__ == '__main__':
    start_dn = sys.argv[1]
    end_dn = sys.argv[2]
    yr = sys.argv[3]
    main(start_dn, end_dn, yr)
