import shutil
from shapely.geometry import Polygon, LineString, Point
import glob
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
import shutil
import wget
from suntime import Sun
from datetime import timedelta

def get_file_list(idx):
    truth_file_list = []
    truth_file_list = glob.glob('{}truth/*/*/*_{}.tif'.format(dn_dir, idx))
    truth_file_list.sort()
    data_file_list = [s.replace('truth','data') for s in truth_file_list]
    data_dict = {'find': {'truth': truth_file_list, 'data': data_file_list}}
    return data_dict

def pick_temporal_smoke(smoke_shape, t_0, t_f):
    use_idx = []
    bounds = smoke_shape.bounds
    for idx, row in smoke_shape.iterrows():
        end = row['End']
        start = row['Start']
        fmt = '%Y%j %H%M'
        start = pytz.utc.localize(datetime.strptime(row['Start'], fmt))
        end = pytz.utc.localize(datetime.strptime(row['End'], fmt))
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
    print("R:", np.sum(R).compute())
    print("total:", total)
    print('========')
    #print('SUM TOTAL: ', int(np.sum((total))))
    if total > 10 and total < 1e5:
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
    skimage.io.imsave(tif_fn, truth_layers)
    return True
    if np.sum(truth_layers) == 0:
        skimage.io.imsave(tif_fn, truth_layers)
        return True
    else:
        print("there was smoke in this one!!!")
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

    fn_head = sat_fns[0].split('C01_')[-1].split('.')[0].split('_c2')[0]

    lcc_str = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
    lcc_proj = pyproj.CRS.from_user_input(lcc_str)

    centers = smoke.centroid
    center = centers.loc[idx0]
    try:
        extent = get_extent(center)
    except:
        return fn_head

    try:
        old_scn, scn = get_get_scn(sat_fns, extent)
    except Exception as e:
        print(e)
        if 'G17' in sat_fns[0]:
            print('G17 data not done downloading!')
            print('wait 15 seconds')
            try:
                old_scn, scn = get_get_scn(sat_fns, extent, 15)
            except Exception as e:
                print(e)
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
            except Exception as e:
                print(e)
                print('G16 data STILL not done downloading!')
                print('wait 60 seconds')
                try:
                    old_scn, scn = get_get_scn(sat_fns, extent, 60)
                except:
                    print('{} wouldnt download, moving on'.format(sat_fns[0]))
                    return fn_head

    lcc_proj = scn['cimss_true_color_sunz_rayleigh'].attrs['area'].to_cartopy_crs()
    scan_start = pytz.utc.localize(scn['cimss_true_color_sunz_rayleigh'].attrs['start_time'])
    scan_end = pytz.utc.localize(scn['cimss_true_color_sunz_rayleigh'].attrs['end_time'])
    rel_smoke = pick_temporal_smoke(smoke, scan_start, scan_end)

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

def get_sunrise_sunset(dt):
    west_lon = -124.8
    west_lat = 24.5
    east_lon = -71.1
    east_lat = 45.93
    east = Sun(east_lat, east_lon)
    west = Sun(west_lat, west_lon)
    sunset = east.get_sunset_time(dt)
    sunrise = west.get_sunrise_time(dt)
    if sunrise > sunset:
        sunset = west.get_sunset_time(dt + timedelta(days=1))
    return sunrise, sunset

def get_random_dt(dt):
    sunrise, sunset = get_sunrise_sunset(dt)
    sun_out = sunset - sunrise
    sun_out_secs = sun_out.total_seconds()
    secs_after_sunrise = np.random.uniform(0, sun_out_secs)
    random_time = sunrise + timedelta(seconds=secs_after_sunrise)
    if np.abs(sunset - random_time) < np.abs(random_time - sunrise):
        sat_num = 16
    else:
        sat_num = 17
    return random_time, sat_num


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

def get_sat_files(dt):
    all_fn_heads = []
    all_sat_fns = []
    best_time, sat_num = get_random_dt(dt)
    time_list = [best_time]
    print('best time; ', best_time)
    fs = s3fs.S3FileSystem(anon=True)
    for curr_time in time_list:
        hr = curr_time.hour
        hr = str(hr).zfill(2)
        yr = curr_time.year
        dn = curr_time.strftime('%j')

        full_filelist = []
        if sat_num == '17':
            view = 'C'
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
        all_sat_fns = all_sat_fns[0]
        all_fn_heads = list(set(all_fn_heads))
        return all_sat_fns, best_time
    return None, best_time

def get_file_locations(use_fns):
    file_locs = []
    fs = s3fs.S3FileSystem(anon=True)
    goes_dir = dn_dir + 'goes_temp/'
    print(use_fns)
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

#@ray.remote
def iter_rows(smoke_row):
    smoke = smoke_row['smoke']
    idx = smoke_row['idx']
    yr = smoke_row['time'].strftime('%Y')
    file_locs = smoke_row['file_locs']
    density = 'None'

    if len(file_locs) > 0:

        fns = create_data_truth(file_locs, smoke, idx, yr, density, (0,0))
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

def check_overlap(center_x, center_y, smoke):
    x0 = center_x - 2.5e5
    y0 = center_y - 2.5e5
    x1 = center_x + 2.5e5
    y1 = center_y + 2.5e5
    poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    overlap = smoke.intersects(poly)
    print(overlap)
    if ((overlap==True).any()):
        return True
    else:
        return False

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
        lcc_str = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
        lcc_proj = pyproj.CRS.from_user_input(lcc_str)
        states = geopandas.read_file('/projects/mecr8410/semantic_segmentation_smoke/data/shape_files/contiguous_states.shp')
        smoke = smoke.to_crs(lcc_proj)
        states = states.to_crs(lcc_proj)
        x_min, y_min, x_max, y_max = states.total_bounds
        num_samples = len(smoke)
        num_samples = 1
        xs = np.random.uniform(x_min, x_max, num_samples)
        ys = np.random.uniform(y_min, y_max, num_samples)
        smoke_rows = []
        for idx, x in enumerate(xs):
            y = ys[idx]
            print('x and y:', x, y)
            overlap = check_overlap(x, y, smoke)
            if not overlap:
                use_fns, best_time = get_sat_files(dt)
                if use_fns:
                    file_locs = get_file_locations(use_fns)
                    smoke_rows.append({'x': x, 'y': y, 'smoke': smoke, 'idx': idx, 'file_locs': file_locs, 'time': best_time})
        #ray_dir = "/projects/rey/smoke_classification/tmp/{}{}".format(yr,dn)
        #if not os.path.isdir(ray_dir):
        #    os.mkdir(ray_dir)
        #ray.init(num_cpus=8, _temp_dir=ray_dir, include_dashboard=False, ignore_reinit_error=True, dashboard_host='127.0.0.1')
        #fn_heads = run_remote(smoke_rows)
        if len(smoke_rows) > 0:
            fn_heads = run_no_ray(smoke_rows)
        #ray.shutdown()
        #shutil.rmtree(ray_dir)
        #if fn_heads:
        #    remove_files(fn_heads)


def main(start_dn, end_dn, yr):
    global dn_dir
    dates = []
    dns = list(range(int(start_dn), int(end_dn)+1))
    for dn in dns:
        dn = str(dn).zfill(3)
        dates.append([dn, yr])
    for date in dates:
        dn_dir = './null_data/temp_data/{}{}/'.format(date[1], date[0])
        if not os.path.isdir(dn_dir):
            os.mkdir(dn_dir)
            MakeDirs(dn_dir, yr)
        start = time.time()
        print(date)
        iter_smoke(date)
        #DELshutil.rmtree(dn_dir)
        print("Time elapsed for day {}: {}s".format(date, int(time.time() - start)), flush=True)

if __name__ == '__main__':
    start_dn = sys.argv[1]
    end_dn = sys.argv[2]
    yr = sys.argv[3]
    main(start_dn, end_dn, yr)
