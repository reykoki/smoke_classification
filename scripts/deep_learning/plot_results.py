import numpy as np
import torch
import skimage
import shelve
import matplotlib.pyplot as plt
import numpy as np

ds = dict(shelve.open("/projects/mecr8410/history/other_stuff/test_results.shlv"))
data = np.array(ds['batch_data'])
labels = np.array(ds['batch_labels'])
losses = np.array(ds['losses'])
pred = np.array(ds['predictions'])


print(len(pred))
pred = np.array(pred)

print(pred.shape)
print(pred[0].shape)



def plot_stuff(img, truth, pred):
    print(img.shape)
    R = img[0]
    G = img[1]
    B = img[2]
    LOS = img[3]
    print(R.shape)

    label = ['R', 'G', 'B', 'LOS']
    cmaps = ['Reds', 'Greens', 'Blues', 'Greys']

    for idx in range(4):
        f = plt.figure(figsize=(16,16))
        plt.imshow(img[idx], cmap=cmaps[idx])
        plt.title(label[idx])
        plt.savefig('plots/{}.png'.format(label[idx]))
        #plt.show()
        #plt.close(f)

    RGB = np.dstack([R, G, B])
    f1 = plt.figure(1)
    plt.imshow(RGB)
    plt.title('RGB')
    plt.savefig('plots/RGB.png', dpi=300)

    label = ['high', 'med', 'low']
    colors = ['Reds', 'Blues', 'Greys']
    for idx in range(3):
        f = plt.figure(figsize=(16,16))
        plt.imshow(pred[idx], cmap=colors[idx])
        plt.title(label[idx] + 'predictions')
        plt.savefig('plots/{}_pred.png'.format(label[idx]))

    for idx in range(3):
        f = plt.figure(figsize=(16,16))
        plt.imshow(truth[idx], cmap=colors[idx])
        plt.title(label[idx] + 'truth')
        plt.savefig('plots/{}_truth.png'.format(label[idx]))


for idx in range(1):
    plot_stuff(data[idx], labels[idx], pred[idx])




