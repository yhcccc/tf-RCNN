"""
Created on Fri Mar  1 14:52:05 2019

@author: yh
"""

import numpy as np
import SelectiveSearch
import os.path
from sklearn import svm
from sklearn.externals import joblib
import RCNN_preprocessing as prep
import os
import tools
import cv2
import config
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


def image_proposal(img_path):
    img = cv2.imread(img_path)
    img_lbl, regions = SelectiveSearch.selective_search(
            img, scale = 500, sigma = 0.9, min_size = 10)
    
    candidates = set()
    images = []
    vertices = []
    for r in regions:
        # Excluding same rectangle (with different segments)
        if r['rect'] in candidates: continue
        
        # Excluding small regions
        if r['size'] < 220: continue
        if (r['rect'][2] * r['rect'][3]) < 500: continue
        
        # Crop image_orig to get the image inside the box
        proposal_img, proposal_vertice = prep.clip_pic(img, r['rect'])
        
        # Delete Empty array
        if len(proposal_img) == 0: continue
        
        # Ignore things contain 0 or not C contiguous array
        x, y, w, h = r['rect']
        if w == 0 or h == 0: continue
        
        # Check if any 0-dimension exist
        [a, b, c] = np.shape(proposal_img)
        if a == 0 or b == 0 or c == 0: continue
        
        # padding clip
        proposal_img, proposal_vertice = prep.clip_pic(img, r['rect'], padding = 16)
        # Object proposal transformations
        # Resize to 227 * 227 for input
        resized_proposal_img = prep.resize_image(proposal_img, config.IMAGE_SIZE, config.IMAGE_SIZE)
        img_float = np.asarray(resized_proposal_img, dtype = "float32")
        images.append(img_float)
        
        vertices.append(r['rect'])
        
        candidates.add(r['rect'])
        
    return images, vertices

# Load training images
def generate_single_svm_train(train_file):
    save_path = train_file.rsplit('.', 1)[0].strip()
    if len(os.listdir(save_path)) == 0:
        print("reading %s's svm dataset" % train_file.split('\\')[-1])
        prep.load_train_RegionProposals(
          train_file, 2, save_path, threshold = 0.3, is_svm = True, save = True)
    print("restoring svm dataset")
    images, labels = prep.load_from_npy(save_path)
    
    return images, labels

# Use a already trained alexnet with the last layer redesigned
def create_AlexNet():
    # Building AlexNet
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
    AlexNet = regression(AlexNet, optimizer = 'momentum', 
                         loss = 'categorical_crossentropy', 
                         learning_rate = 0.001)
    return AlexNet

# Construct cascade svms
def train_svms(train_file_folder, model):
    files = os.listdir(train_file_folder)
    print(files)
    svms = []
    for train_file in files:
        if train_file.split('.')[-1] == 'txt':
            X, Y = generate_single_svm_train(os.path.join(
                    train_file_folder, train_file))
            train_features = []
            for ind, i in enumerate(X):
                # extract features
                feats = model.predict([i])
                # print(feats, feats.shape)
                train_features.append(feats[0])
                tools.view_bar(
                    "extract features of %s" % train_file, ind + 1, len(X))
            print(' ')
            print("feature dimension")
            print(np.shape(train_features))
            # SVM training
            clf = svm.LinearSVC()
            print("fit svm")
            clf.fit(train_features, Y)
            svms.append(clf)
            joblib.dump(clf, 
              os.path.join(train_file_folder, str(train_file.split('.')[0])
              + '_svm.pkl'))
    return svms

if __name__ == '__main__':
    train_file_folder = config.TRAIN_SVM
    # or './17flowers/jpg/16/****.jpg'
    img_path = './17flowers/jpg/16/image_1285.jpg'
    imgs, verts = image_proposal(img_path)
    tools.show_rect(img_path, verts)
    
    net = create_AlexNet()
    model = tflearn.DNN(net)
    model.load(config.FINE_TUNE_MODEL_PATH)
    svms = []
    for file in os.listdir(train_file_folder):
        if file.split('_')[-1] == 'svm.pkl':
            svms.append(joblib.load(os.path.join(train_file_folder, file)))
    if len(svms) == 0:
        svms = train_svms(train_file_folder, model)
    print("Fitting svms done!")
    features = model.predict(imgs)
    print("predict image:", np.shape(features))
    results = []
    results_label = []
    count = 0
    for f in features:
        for Svm in svms:
            pred = Svm.predict([f.tolist()])
            # print(pred)
            # not background
            if pred[0] != 0:
                results.append(verts[count])
                results_label.append(pred[0])
        count += 1
    print("result:", results)
    print("result_label:", results_label)
    tools.show_rect(img_path, results)


