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

def get_smoke(yr, month, day):
    fn = 'hms_smoke{}{}{}.zip'.format(yr, month, day)
    print('DOWNLOADING SMOKE:')
    print(fn)
    out_dir = '/scratch/alpine/mecr8410/semantic_segmentation_smoke/new_data_copy/smoke/'
    smoke_shape_fn = '{}hms_smoke{}{}{}.shp'.format(out_dir, yr,month,day)
    if os.path.exists(out_dir+fn):
        print("{} already exists".format(fn))
        smoke = geopandas.read_file(smoke_shape_fn)
        return smoke
    else:
        try:
            url = 'https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/Shapefile/{}/{}/{}'.format(yr, month, fn)
            filename = wget.download(url, out=out_dir)
            shutil.unpack_archive(filename, out_dir)
            smoke = geopandas.read_file(smoke_shape_fn)
            return smoke
        except Exception as e:
            print(e)
            print('NO SMOKE DATA FOR THIS DATE')
            return None


def check_overlap(center_x, center_y, smoke):
    x0 = center_x - 2.5e3
    y0 = center_y - 2.5e3
    x1 = center_x + 2.5e3
    y1 = center_y + 2.5e3
    poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    overlap = smoke.intersects(poly)
    if (overlap==True.any()):
        return True
    else:
        return False

smoke = get_smoke(2023, '01', 21)
print(smoke)
smoke0 = smoke
lcc_str = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
lcc_proj = pyproj.CRS.from_user_input(lcc_str)
smoke = smoke.to_crs(lcc_proj)
states = geopandas.read_file('/projects/mecr8410/semantic_segmentation_smoke/data/shape_files/contiguous_states.shp')
#states = states.to_crs(lcc_proj)
states = states.to_crs(smoke.crs)
x_min, y_min, x_max, y_max = states.total_bounds
print(x_min, y_min, x_max, y_max)
x = np.random.uniform(x_min, x_max)
y = np.random.uniform(y_min, y_max)
centers = smoke.centroid
center = centers.loc[5]
print(center.x, center.y)

s = get_polygon(center.x, center.y, lcc_proj, smoke)
s2 = get_polygon(x, y, lcc_proj, smoke)
print(smoke.intersects(s))
print(smoke.intersects(s2))
a = smoke.intersects(s2)

if (a==True).any():
    print('randomly seleted point overlaps with smoke!')


#df = pd.DataFrame({'longitude': [-140, 0, 123], 'latitude': [-65, 1, 48]})
#pd.DataFrame({'longitude': [], 'latitude': [-65, 1, 48]})
#lcc_str = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
#lcc_proj = pyproj.CRS.from_user_input(lcc_str)
#smoke_lcc = smoke.to_crs(lcc_proj)
#centers = smoke_lcc.centroid



