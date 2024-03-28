import os
import pickle
import glob
import random

yrs = ['2018', '2019', '2020', '2021', '2022', '2023']
yrs = ['2020', '2021', '2022']
val_test_yr = '2022'

data_dir = '/scratch/alpine/mecr8410/semantic_segmentation_smoke/filtered_data/data/'

cats = ['Light', 'Medium', 'Heavy', 'None']
cat_count = {'Light':100, 'Medium':100, 'Heavy': 100, 'None': 100}
#cat_count = {'Light':None, 'Medium':None, 'Heavy': None, 'None': None}


def make_class(data_fns, cat):
    if cat == 'None':
        return [0]*len(data_fns)
    else:
        return [1]*len(data_fns)

def list_every_ten_days(dataset):
    dns = list(range(0,37))
    dns_filled = [str(item).zfill(2) for item in dns]
    if dataset == 'val':
        val_days = dns_filled[::2]
        return val_days
    if dataset == 'test':
        test_days = dns_filled[1::2]
        return test_days

def get_val_test_fns(yr, cat_count, dataset):
    truth = []
    data_fns = []
    days_oi = list_every_ten_days(dataset)
    yr_data_dir = data_dir + yr + '/'
    cat_num_files = 0
    for cat in cat_count:
        cat_data_fns = []
        for days in days_oi:
            cat_data_fns.extend(glob.glob('{}{}/*_s{}{}*.tif'.format(yr_data_dir, cat, yr, days)))
        if len(cat_data_fns) > cat_count[cat]:
            random.shuffle(cat_data_fns)
            cat_data_fns = cat_data_fns[:cat_count[cat]]
        cat_truth = make_class(cat_data_fns, cat)
        truth.extend(cat_truth)
        data_fns.extend(cat_data_fns)
    return truth, data_fns

def get_train_fns(yrs, cat_count):
    truth = []
    data_fns = []
    for yr in yrs:
        yr_data_dir = data_dir + yr + '/'
        cat_num_files = 0
        for cat in cat_count:
            cat_data_fns = glob.glob('{}{}/*.tif'.format(yr_data_dir, cat))
            if len(cat_data_fns) > cat_count[cat]:
                random.shuffle(cat_data_fns)
                cat_data_fns = cat_data_fns[:cat_count[cat]]
            cat_truth = make_class(cat_data_fns, cat)
            truth.extend(cat_truth)
            data_fns.extend(cat_data_fns)
    return truth, data_fns

train_truth, train_data_fns = get_train_fns(yrs, cat_count)
val_truth, val_data_fns = get_val_test_fns(val_test_yr, cat_count, 'val')
test_truth, test_data_fns = get_val_test_fns(val_test_yr, cat_count, 'test')

data_dict = {'train': {'truth': train_truth, 'data': train_data_fns},
             'val': {'truth': val_truth, 'data': val_data_fns},
             'test': {'truth': test_truth, 'data': test_data_fns}}

print('number of train samples:', len(train_truth))
print('number of val samples:', len(val_truth))
print('number of test samples:', len(test_truth))

with open('subsample.pkl', 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

