import matplotlib.pyplot as plt
import numpy as np
import sys
import glob


def get_ids(fns):
    new_fns = []
    for fn_loc in fns:
        fn = fn_loc.split('/')[-1]
        a_s = fn.split('_')
        a_n = '*' + a_s[1][0:8]+ '*_' + a_s[-1]
        new_fns.append(a_n)
    return new_fns

def get_list_overlap(dir_1, dir_2):
    fns_1 = glob.glob('{}*/*/*tif'.format(dir_1))
    fns_2 = glob.glob('{}*/*/*tif'.format(dir_2))
    ids_1 = get_ids(fns_1)
    ids_2 = get_ids(fns_2)
    print('length of new_data files: ', len(ids_1))
    print('length of filtered files: ', len(ids_2))
    overlap_ids = list(set(ids_1).intersection(ids_2))
    print('length of overlapping files', len(overlap_ids))
    return overlap_ids

def get_fns(fn_id, dir_1, dir_2):
    fn_1 = glob.glob('{}truth/*/*/{}'.format(dir_1, fn_id))
    fn_2 = glob.glob('{}truth/*/*/{}'.format(dir_2, fn_id))
    return fn_1[0].split("/")[-1], fn_2[0].split("/")[-1]

def get_data_dict(fn_id, dir_1, dir_2, data_dict):
    t_fn_1 = glob.glob('{}truth/*/*/{}'.format(dir_1, fn_id))[0]
    t_fn_2 = glob.glob('{}truth/*/*/{}'.format(dir_2, fn_id))[0]
    d_fn_1 = t_fn_1.replace('truth','data')
    d_fn_2 = t_fn_2.replace('truth','data')
    data_dict['test']['truth'].append(t_fn_1)
    data_dict['test']['truth'].append(t_fn_2)
    data_dict['test']['data'].append(d_fn_1)
    data_dict['test']['data'].append(d_fn_2)
    return data_dict

dir_1 = "/scratch/alpine/mecr8410/semantic_segmentation_smoke/new_data/"
dir_2 = "/scratch/alpine/mecr8410/semantic_segmentation_smoke/filtered_data/"
fns = get_list_overlap("{}truth/".format(dir_1), "{}truth/".format(dir_2))

same_count = 0
diff_count = 0
for i in range(len(fns)):
    fn_1, fn_2 = get_fns(fns[i], dir_1, dir_2)
    if fn_1 != fn_2:
        diff_count += 1
    else:
        same_count += 1

print(diff_count)
print(same_count)
