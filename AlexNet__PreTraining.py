"""
Created on Mon Feb 25 09:36:56 2019

@author: yh
"""

import pickle
import numpy as np
# Import os and use os.path. Do not import os.path directly.
import os
# io.open() works in Python 2.6 and all later versions
# import codecs

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import cv2

import config
import RCNN_preprocessing as prep

def load_data(datafile, num_class, save = False, save_path = 'dataset.pkl'):
    # fr = open(datafile, 'r', encoding = 'utf-8')
    with open(datafile, 'r', encoding = 'utf-8') as fr:
        labels = []
        images = []
        data_list = fr.readlines()
        
    for line in data_list:
        tmp = line.strip().split(' ')
        fpath = tmp[0]
        img = cv2.imread(fpath)
        img = prep.resize_image(img, config.IMAGE_SIZE, config.IMAGE_SIZE)
        # asarray 不生成新的array
        np_img = np.asarray(img, dtype = "float32")
        images.append(np_img)
        
        index = int(tmp[1])
        label = np.zeros(num_class)
        label[index] = 1
        labels.append(label)
        
    if save:
        with open(save_path, 'wb') as inputFile:
            pickle.dump((images, labels), inputFile)
    print(np.shape(images), np.shape(labels))
    return images, labels

def load_from_pkl(dataset_file):
    with open(dataset_file, 'rb') as outputFile:
        X, Y = pickle.load(outputFile)
    return X, Y

def create_AlexNet(num_classes):
    AlexNet = input_data(shape = [None, config.IMAGE_SIZE, config.IMAGE_SIZE, 3])
    # 原论文中， padding = valid 
    AlexNet = conv_2d(AlexNet, 96, 11, strides = 4, padding = "valid", activation = 'relu')
    # 原论文中， padding = valid 
    # Pooling和LRN的顺序问题：caffe的model zoo里有个caffenet，结构基本与Alexnet一样，
    # 只是LRN和pooling的顺序换了一下，据考证这是caffenet作者在复现网络时的失误caffe issues，
    # 作者本来是想实现个一样的，不怪他，我把论文仔仔细细看了，说得的确不直观。
    # 尽管如此caffenet依然被广泛的使用，为什么呢？
    # 从实现上来看，LRN与pooling顺序的调换并没有影响到整体的网络结构；
    # 相反，从另一种角度来看，顺序的调换反而节约了计算量，因为Alexnet先LRN计算完后，
    # 许多值被后面的pooling丢掉了；在GoogLeNet中就是先池化后LRN。
    # https://blog.csdn.net/youmy1111/article/details/59704841
    AlexNet = max_pool_2d(AlexNet, 3, strides = 2, padding = "valid")
    # LRN 的效果不如 BN
    AlexNet = local_response_normalization(AlexNet)
    AlexNet = conv_2d(AlexNet, 256, 5, activation = 'relu')
    AlexNet = max_pool_2d(AlexNet, 3, strides = 2, padding = "valid")
    AlexNet = conv_2d(AlexNet, 384, 3, activation = 'relu')
    AlexNet = conv_2d(AlexNet, 384, 3, activation = 'relu')
    AlexNet = conv_2d(AlexNet, 256, 3, activation = 'relu')
    AlexNet = max_pool_2d(AlexNet, 3, strides = 2, padding = "valid")
    # If not 2D, input will be flatten.
    AlexNet = fully_connected(AlexNet, 4096, activation = 'relu')
    AlexNet = dropout(AlexNet, 0.5)
    AlexNet = fully_connected(AlexNet, 4096, activation = 'relu')
    AlexNet = dropout(AlexNet, 0.5)
    AlexNet = fully_connected(AlexNet, num_classes, activation = 'softmax')
    AlexNet = regression(AlexNet, optimizer = 'momentum', 
                         loss = 'categorical_crossentropy', 
                         learning_rate = 0.001)
    return AlexNet

def train(network, X, Y, save_model_path):
    # Training
    model = tflearn.DNN(network, checkpoint_path = 'model_AlexNet',
                        max_checkpoints = 1, tensorboard_verbose = 2,
                        tensorboard_dir = 'PreTraining_output')
    if os.path.isfile(save_model_path + '.index'):
        model.load(save_model_path)
        print('loading model...')
    for _ in range(5):
        # epoch = 1000
        model.fit(X, Y, n_epoch = 60, validation_set = 0.1, shuffle = True,
                  show_metric = True, batch_size = 64, snapshot_step = 200,
                  snapshot_epoch = False, run_id = 'alexnet_oxflowers17') 
        model.save(save_model_path)
        print('save model...')
        
def predict(network, modelfile, images):
    model = tflearn.DNN(network)
    model.load(modelfile)
    return model.predict(images)

if __name__ == '__main__':
    X, Y = load_data(config.TRAIN_LIST, config.TRAIN_CLASS)
    net = create_AlexNet(config.TRAIN_CLASS)
    train(net, X, Y, config.SAVE_MODEL_PATH)