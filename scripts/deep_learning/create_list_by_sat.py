import pickle
import os
import glob
global sat_num
sat_num = 16
train_yrs = ['2018', '2019', '2020', '2021', '2023']
val_test_yr = '2022'

#truth_dir = '/scratch/alpine/mecr8410/semantic_segmentation_smoke/data/subset/truth/'
truth_dir = '/scratch/alpine/mecr8410/semantic_segmentation_smoke/new_data/truth/'
truth_dir = '/scratch/alpine/mecr8410/semantic_segmentation_smoke/filtered_data/truth/'

def check_list(filelist):
    for fn in filelist:
        if not os.path.isfile(fn):
            print('FILE DOESNT EXIST')
            print(fn)

def check_files_exists(filelists):
    check_list(filelists['truth'])
    check_list(filelists['data'])

def get_training_file_list(truth_dir):
    train_truth_file_list = []
    for yr in train_yrs:
        yr_truth_dir = truth_dir + yr + '/'
        truth_file_list = glob.glob('{}*/G{}*.tif'.format(yr_truth_dir, sat_num))
        train_truth_file_list.extend(truth_file_list)
    train_truth_file_list.sort()
    train_data_file_list = [s.replace('truth','data') for s in train_truth_file_list]
    return train_truth_file_list, train_data_file_list

train_truth_file_list, train_data_file_list = get_training_file_list(truth_dir)

def list_every_ten_days():
    dns = list(range(0,37))
    dns_filled = [str(item).zfill(2) for item in dns]
    val_days = dns_filled[::2]
    print(val_days)
    test_days = dns_filled[1::2]
    return val_days, test_days

def create_truth_file_list_every_ten_days(yr_truth_dir, val_test_yr, dns):
    truth_file_list = []
    for dn in dns:
        truth_file_list.extend(glob.glob('{}/*/G{}*s{}{}*'.format(yr_truth_dir,
                                                                  sat_num,
                                                              val_test_yr,
                                                              dn)))
    truth_file_list.sort()
    data_file_list = [s.replace('truth','data') for s in truth_file_list]
    return truth_file_list, data_file_list

def get_val_test_file_list(truth_dir, val_test_yr):
    yr_truth_dir = truth_dir + val_test_yr + '/'
    val_days, test_days = list_every_ten_days()
    val_truth_file_list, val_data_file_list = create_truth_file_list_every_ten_days(yr_truth_dir, val_test_yr, val_days)
    test_truth_file_list, test_data_file_list = create_truth_file_list_every_ten_days(yr_truth_dir, val_test_yr, test_days)

    return val_truth_file_list, val_data_file_list, test_truth_file_list, test_data_file_list

val_truth_file_list, val_data_file_list, test_truth_file_list, test_data_file_list = get_val_test_file_list(truth_dir, val_test_yr)


print('number of train samples:', len(train_truth_file_list))
print('number of val samples:', len(val_truth_file_list))
print('number of test samples:', len(test_truth_file_list))

data_dict = {'train': {'truth': train_truth_file_list, 'data': train_data_file_list},
             'val': {'truth': val_truth_file_list, 'data': val_data_file_list},
             'test': {'truth': test_truth_file_list, 'data': test_data_file_list}}

check_files = True
if check_files:
    check_files_exists(data_dict['train'])
    check_files_exists(data_dict['val'])
    check_files_exists(data_dict['test'])

with open('data_dict_G{}.pkl'.format(sat_num), 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
