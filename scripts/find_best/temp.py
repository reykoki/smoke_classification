import shutil
import glob
from SmokeDataset import SmokeDataset
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

def get_file_locations(use_fns):
    file_locs = []
    fs = s3fs.S3FileSystem(anon=True)
    goes_dir = './goes_temp/'
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

fs = s3fs.S3FileSystem(anon=True)
full_filelist = fs.ls("noaa-goes{}/ABI-L1b-Rad{}/{}/{}/{}/".format(16, 'C', 2023, 100, 20))
C01_list = list(filter(lambda x: 'C01' in x, full_filelist))
print(len(C01_list))
print(random.choice(C01_list))
fns = [random.choice(C01_list)]
fns = get_file_locations(fns)
print(fns)
scn = Scene(reader='abi_l1b', filenames=fns)
name = 'C01'
scn.load([name], generate=False)
my_area = create_area_def(area_id='lccCONUS',
                          description='Lambert conformal conic for the contiguous US',
                          projection="+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs",
                          resolution=1000)



lon, lat = scn[name].attrs['area'].get_lonlats()
min_lon = np.min(lon)
max_lon = np.nanmax(lon[lon != np.inf])
min_lat = np.min(lat)
max_lat = np.nanmax(lat[lat != np.inf])
print(min_lon, max_lon, min_lat, max_lat)
states = geopandas.read_file('/projects/mecr8410/semantic_segmentation_smoke/data/shape_files/contiguous_states.shp')
x_min, y_min, x_max, y_max = states.total_bounds
print(x_min, y_min, x_max, y_max)
x = np.random.uniform(x_min, x_max)
y = np.random.uniform(y_min, y_max)
print(x, y )

