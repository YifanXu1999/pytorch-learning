import pickle
import os
import cv2
import numpy as np



def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

label_name = ["airplane",
              "automobile",
              "bird",
              "cat",
              "deer",
              "dog",
              "frog",
              "horse",
              "ship",
              "truck"
              ]

import glob

train_list = glob.glob("/Users/yifanxu/Programming/ML/Pytorch/Cifar-10/pythonProject1/data/cifar-10-batches-py/data_batch_*")
print(train_list)

for l in train_list:
    print(l)
    l_dict = unpickle(l)
    print(l_dict)
    print(l_dict.keys())
    for im_idx, im_data in enumerate(l_dict[b'data']):
        print(im_idx)
        print(im_data)

        im_label = l_dict[b'labels'][im_idx]
        im_name = l_dict[b'filenames'][im_idx]
        print(im_label)
        print(im_name)

        im_label_name = label_name[im_label]
        im_data = np.reshape(im_data, [3, 32, 32])

        im_data = np.transpose(im_data, (1, 2, 0))

