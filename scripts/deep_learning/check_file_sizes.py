import pickle
import skimage


def iter_list_size(filelist):
    for fn in filelist:
        try:
            data_img = skimage.io.imread(fn, plugin='tifffile')
        except:
            print('corrupted fn: ', fn)
        #if data_img.shape != (512, 512, 3):
        if data_img.shape != (256, 256, 3):
            print(fn, 'is wrong shape!!', data_img.shape)

def check_size(filelists):
    iter_list_size(filelists['truth'])
    iter_list_size(filelists['data'])

with open('data_dict_new.pkl', 'rb') as handle:
    data_dict = pickle.load(handle)

check_size(data_dict['train'])
check_size(data_dict['val'])
check_size(data_dict['test'])
