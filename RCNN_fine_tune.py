"""
Created on Fri Mar  1 16:58:29 2019

@author: yh
"""

import os
import RCNN_preprocessing as prep
import config
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


def create_AlexNet(num_classes, restore = False):
    # Building 'AlexNet'
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
    AlexNet = fully_connected(AlexNet, num_classes, activation = 'softmax', restore = restore)
    AlexNet = regression(AlexNet, optimizer = 'momentum', 
                         loss = 'categorical_crossentropy', 
                         learning_rate = 0.001)
    return AlexNet

def fine_tune_AlexNet(net, X, Y, save_model_path, fine_tune_model_path):
    # Training
    model = tflearn.DNN(net, checkpoint_path = 'rcnn_model_alexnet',
                        max_checkpoints = 1, tensorboard_verbose = 2,
                        tensorboard_dir = 'output_RCNN')
    if os.path.isfile(fine_tune_model_path + '.index'):
        print("Loading the fine tuned model")
        model.load(fine_tune_model_path)
    elif os.path.isfile(save_model_path + '.index'):
        print("Loading the alexnet")
        model.load(save_model_path)
    else:
        print("No file to load, error")
        return False
    # epoch = 1000
    model.fit(X, Y, n_epoch = 50, validation_set = 0.1, shuffle = True,
              show_metric = True, batch_size = 64, snapshot_step = 200,
              snapshot_epoch = False, run_id = 'alexnet_rcnnflowers')
    # save the model
    model.save(fine_tune_model_path)
    
if __name__ == '__main__':
    data_set = config.FINE_TUNE_DATA
    if len(os.listdir(config.FINE_TUNE_DATA)) == 0:
        print("Reading Data")
        prep.load_train_RegionProposals(config.FINE_TUNE_LIST, 2, save = True,
                                        save_path = data_set)
    print("Loading Data")
    X, Y = prep.load_from_npy(data_set)
    restore = False
    if os.path.isfile(config.FINE_TUNE_MODEL_PATH + '.index'):
        restore = True
        print("Continue fine-tune")
    # three classes include background
    net = create_AlexNet(config.FINE_TUNE_CLASS, restore = restore)
    fine_tune_AlexNet(net, X, Y, 
                      config.SAVE_MODEL_PATH, config.FINE_TUNE_MODEL_PATH)


