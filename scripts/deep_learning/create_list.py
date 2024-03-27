import pickle
import os
import glob

train_yrs = ['2018', '2019', '2020', '2021', '2023']

val_test_yr = '2022'

#truth_dir = '/scratch/alpine/mecr8410/semantic_segmentation_smoke/data/subset/truth/'
#truth_dir = '/scratch/alpine/mecr8410/semantic_segmentation_smoke/new_data/truth/'
truth_dir = '/scratch/alpine/mecr8410/semantic_segmentation_smoke/filtered_data/truth/'

def check_list(filelist):
    for fn in filelist:
        if not os.path.isfile(fn):
            print('FILE DOESNT EXIST')
            print(fn)

def check_files_exists(filelists):
    check_list(filelists['truth'])
    check_list(filelists['data'])

def get_training_fns(truth_dir, num_samples=None):
    train_truth_fns = []
    train_truth_light_fns = []
    train_truth_none_fns = []
    for yr in train_yrs:
        yr_truth_dir = truth_dir + yr + '/'
        train_truth_light_fns.extend(glob.glob('{}/Light/*.tif'.format(yr_truth_dir)))
        train_truth_fns.extend(glob.glob('{}/Heavy/*.tif'.format(yr_truth_dir)))
        train_truth_fns.extend(glob.glob('{}/Medium/*.tif'.format(yr_truth_dir)))
        train_truth_fns.extend(glob.glob('{}/None/*.tif'.format(yr_truth_dir)))
    len_all = len(train_truth_fns) + len(train_truth_light_fns) + len(train_truth_none_fns)
    if num_samples and num_samples < len_all:
        if len(train_truth_fns) > num_samples:
            random.shuffle(train_truth_fns)
            train_truth_fns = train_truth_fns[:num_samples]
        else:
            random.shuffle(train_truth_light_fns)
            num_light_samples = num_samples - len(train_truth_fns)
            train_truth_light_fns = train_truth_light_fns[:num_light_samples]
            train_truth_fns.extend(train_truth_light_fns)
    train_truth_fns.sort()
    train_data_fns = [s.replace('truth','data') for s in train_truth_fns]
    return train_truth_fns, train_data_fns

train_truth_fns, train_data_fns = get_training_fns(truth_dir, num_samples=60000)

def list_every_ten_days():
    dns = list(range(0,37))
    dns_filled = [str(item).zfill(2) for item in dns]
    val_days = dns_filled[::2]
    print(val_days)
    test_days = dns_filled[1::2]
    return val_days, test_days

def create_truth_fns_every_ten_days(yr_truth_dir, val_test_yr, dns):
    truth_fns = []
    truth_light_fns = []
    for dn in dns:
        truth_fns.extend(glob.glob('{}/None/*s{}{}*'.format(yr_truth_dir,
                                                              val_test_yr,
                                                              dn)))
        truth_fns.extend(glob.glob('{}/Heavy/*s{}{}*'.format(yr_truth_dir,
                                                              val_test_yr,
                                                              dn)))
        truth_fns.extend(glob.glob('{}/Medium/*s{}{}*'.format(yr_truth_dir,
                                                              val_test_yr,
                                                              dn)))
        truth_light_fns.extend(glob.glob('{}/Light/*s{}{}*'.format(yr_truth_dir,
                                                              val_test_yr,
                                                              dn)))
    if num_samples and num_samples < len(truth_fns)+ len(truth_light_fns):
        if len(truth_fns) > num_samples:
            random.shuffle(truth_fns)
            truth_fns = truth_fns[:num_samples]
        else:
            random.shuffle(truth_light_fns)
            num_light_samples = num_samples - len(train_truth_fns)
            train_truth_light_fns = train_truth_light_fns[:num_light_samples]
            train_truth_fns.extend(train_truth_light_fns)
    truth_fns.sort()
    data_fns = [s.replace('truth','data') for s in truth_fns]
    return truth_fns, data_fns

def get_val_test_fns(truth_dir, val_test_yr):
    yr_truth_dir = truth_dir + val_test_yr + '/'
    val_days, test_days = list_every_ten_days()
    val_truth_fns, val_data_fns = create_truth_fns_every_ten_days(yr_truth_dir, val_test_yr, val_days)
    test_truth_fns, test_data_fns = create_truth_fns_every_ten_days(yr_truth_dir, val_test_yr, test_days)

    return val_truth_fns, val_data_fns, test_truth_fns, test_data_fns

val_truth_fns, val_data_fns, test_truth_fns, test_data_fns = get_val_test_fns(truth_dir, val_test_yr)


print('number of train samples:', len(train_truth_fns))
print('number of val samples:', len(val_truth_fns))
print('number of test samples:', len(test_truth_fns))

def make_class(truth_fns):
    class_list = []
    for fn in truth_fns:
        if 'None' in truth_fns:
            class_list.append(0)
        else:
            class_list.append(1)
    return class_list

train_class_list = make_class(train_truth_fns)
val_class_list = make_class(val_truth_fns)
test_class_list = make_class(test_truth_fns)

data_dict = {'train': {'truth': train_class_list, 'data': train_data_fns},
             'val': {'truth': val_class_list, 'data': val_data_fns},
             'test': {'truth': test_class_list, 'data': test_data_file_list}}

check_files = False
if check_files:
    check_files_exists(data_dict['train'])
    check_files_exists(data_dict['val'])
    check_files_exists(data_dict['test'])


with open('class_data_dict.pkl', 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
