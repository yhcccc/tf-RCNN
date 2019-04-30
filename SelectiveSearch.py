"""
Created on Wed Feb 27 16:09:37 2019

@author: yh
"""
'''
https://blog.csdn.net/guoyunfei20/article/details/78723646
'''

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np

# "Selective Search for Object Recognition" by J.R.R. Uijlings et al.
#
#  - Modified version with LBP extractor for texture vectorization

def _generate_segments(im_orig, scale, sigma, min_size):    
    """
    segment smallest regions by the algorithm of Felzenswalb and Huttenlocher
    """
    # segmentation
    im_mask = skimage.segmentation.felzenszwalb(
            skimage.util.img_as_float(im_orig), scale = scale, sigma = sigma,
            min_size = min_size)
    
    # merge mask channel to the image as a 4th channel
    im_orig = np.append(im_orig, 
                    np.zeros(im_orig.shape[:2])[:, :, np.newaxis], axis = 2)
    im_orig[:, :, 3] = im_mask
    
    return im_orig

def _sim_colour(r1, r2):
    """
    calculate the sum of histogram intersection of colour
    """
    return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])

def _sim_texture(r1, r2):
    """
    calculate the sum of histogram intersection of texture
    """
    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])

def _sim_size(r1, r2, imsize):
    """
    calculate the size similarity over the image
    """
    return 1.0 - (r1["size"] + r2["size"]) / imsize

def _sim_fill(r1, r2, imsize):
    """
    calculate the fill similarity over the image
    """
    bbsize = ((max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
            * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"])))
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize

def _calc_sim(r1, r2, imsize):
    return (_sim_colour(r1, r2) + _sim_texture(r1, r2)
            + _sim_size(r1, r2, imsize) + _sim_fill(r1, r2, imsize))

def _calc_colour_hist(img):
    """
    使用L1-norm归一化获取图像每个颜色通道的25 bins的直方图，
    这样每个区域都可以得到一个75维的向量
    calculate colour histogram for each region
    the size of output histogram will be BINS * COLOUR_CHANNELS(3)
    number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
    extract HSV
    args:
        img：ndarray类型， 形状为候选区域像素数 x 3(h,s,v)
        
    return：一维的ndarray类型，长度为75
    """
    BINS = 25
    hist = np.array([])
    
    for colour_channel in (0, 1, 2):
        # extracting one colour channel
        c = img[:, colour_channel]
        
        # calculate histogram for each colour and join to the result
        # 计算每一个颜色通道的25 bins的直方图 然后合并到一个一维数组中
        ########################################
        # 对于一维数组拼接，axis的值不影响最后的结果
        hist = np.concatenate([hist] + [np.histogram(c, BINS, (0.0, 255.0))[0]])
        
    # L1 normalize len(img):候选区域像素数
    hist = hist / len(img)
    
    return hist

def _calc_texture_gradient(img):
    """
    原文：对每个颜色通道的8个不同方向计算方差σ=1的高斯微分（Gaussian Derivative，
    这里使用LBP替代
    calculate texture gradient for entire image

    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we use LBP instead.

    output will be [height(*)][width(*)]
    args：
        img： ndarray类型，形状为height x width x 4，
        每一个像素的值为 [r,g,b,(region)]
        
    return：纹理特征，形状为height x width x 4
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    
    for colour_channel in (0, 1, 2):
        # ret[:, :, colour_channel] = skimage.feature.local_binary_pattern(
        #         img[:, :, colour_channel], 8, 1.0, method = uniform)
        ret[:, :, colour_channel] = skimage.feature.local_binary_pattern(
                img[:, :, colour_channel], 8, 1.0)
        
    return ret

def _calc_texture_hist(img):
    """
    使用L1-norm归一化获取图像每个颜色通道的每个方向的10 bins的直方图，
    这样就可以获取到一个240（10x8x3）维的向量
    calculate texture histogram for each region

    calculate the histogram of gradient for each colours
    the size of output histogram will be
        BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
        
    args:
        img：候选区域纹理特征   形状为候选区域像素数 x 4(r,g,b,(region))
        
    return：一维的ndarray类型，长度为240
    """
    BINS = 10
    hist = np.array([])
    
    for colour_channel in (0, 1, 2):
        # mask by the colour channel 
        fd = img[:, colour_channel]
        
        # calculate histogram for each orientation and concatenate them all
        # and join to the result
        hist = np.concatenate(
                [hist] + [np.histogram(fd, BINS, (0.0, 1.0))[0]])
    
    # L1 Normalize
    hist = hist / len(img)
    
    return hist

def _extract_regions(img):
    '''
    提取每一个候选区域的信息  
    比如类别(region)为5的区域表示的是一只猫的选区，
    这里就是提取这只猫的边界框，左上角后右下角坐标
    
    args:
        img: ndarray类型，形状为height x width x 4，
        每一个像素的值为 [r,g,b,(region)]
    
    return : 
        R:dict 每一个元素对应一个候选区域， 每个元素也是一个dict类型
                              {min_x:边界框的左上角x坐标,
                              min_y:边界框的左上角y坐标,
                              max_x:边界框的右下角x坐标,
                              max_y:边界框的右下角y坐标,
                              size:像素个数,
                              hist_c:颜色的直方图,
                              hist_t:纹理特征的直方图,} 
    '''
    #保存所有候选区域的bounding box  
    # 每一个元素都是一个dict {最小x坐标值，最小y坐标值，最大x坐标值，最大y坐标值,类别}
    #                            通过上面四个参数确定一个边界框
    
    R = {}
    
    # get hsv image
    hsv = skimage.color.rgb2hsv(img[:, :, :3])
    
    # step 1: count pixel positions
    for y, i in enumerate(img): # y = 0 -> height - 1
        for x, (r, g, b, l) in enumerate(i): #x = 0 -> width - 1
            # initialize a new region
            if l not in R:
                R[l] = {"min_x": 0xffff, "min_y": 0xffff, 
                        "max_x": 0, "max_y": 0, "labels": [l]}
                
            # bounding box
            if R[l]["min_x"] > x:
                R[l]["min_x"] = x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y
                
    # step 2: calculate texture gradient 纹理特征提取利用LBP算子height x width x 4
    text_grad = _calc_texture_gradient(img)
    
    # step 3: calculate colour histogram of each region
    for k, v in R.items():
        
        # colour histogram   height x width x 3 -> 候选区域k像素数 x 3? x 4?
        # （img[:, :, 3] == k返回的是一个二维坐标的集合）
        # 取出k域中的hsv像素点，得到的是：array([[...],[h,s,v,k],...,[...]])?
        masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
        # 像素点数衡量size
        # print(type(masked_pixels),masked_pixels.shape)
        # / 4 ?
        R[k]["size"] = len(masked_pixels / 4)
        # print(len(masked_pixels / 4))
        #在hsv色彩空间下，使用L1-norm归一化获取图像每个颜色通道的25 bins的直方图，
        # 这样每个区域都可以得到一个75维的向量
        R[k]["hist_c"] = _calc_colour_hist(masked_pixels)
        
        # texture histogram
        # tex_grad[:, :][img[:, :, 3] == k]形状为候选区域像素数 x 4
        # print(type(text_grad[:, :][img[:, :, 3] == k]),text_grad[:, :][img[:, :, 3] == k].shape)
        R[k]["hist_t"] = _calc_texture_hist(text_grad[:, :][img[:, :, 3] == k])
        
    return R
   
def _extract_neighbours(regions):
    '''
    提取 邻居候选区域对(ri,rj)(即两两相交)
    
    args:
        regions：dict 每一个元素都对应一个候选区域
    return：
        返回一个list，每一个元素都对应一个邻居候选区域对
        区域的表达也变了，由字典变为列表： [(标签，{属性字典}),...]
    '''
    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False
    
    R = list(regions.items())
    neighbours = []
    for cur, a in enumerate(R[:-1]):
        for b in R[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))
                
    return neighbours
    
def _merge_regions(r1, r2):
    '''
    合并两个候选区域
    
    args:
        r1：候选区域1
        r2：候选区域2
        
    return：
        返回合并后的候选区域rt
    '''
    new_size = r1["size"] + r2["size"]
    rt = {
          "min_x": min(r1["min_x"], r2["min_x"]),
          "min_y": min(r1["min_y"], r2["min_y"]),
          "max_x": max(r1["max_x"], r2["max_x"]),
          "max_y": max(r1["max_y"], r2["max_y"]),
          "size": new_size,
          "hist_c": (
             r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
          "hist_t": (
             r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
          "labels": r1["labels"] + r2["labels"]  
            }
    return rt

def selective_search(im_orig, scale = 1.0, sigma = 0.8, min_size = 50):
    '''
    Selective Search
    
    首先通过基于图的图像分割方法初始化原始区域，就是将图像分割成很多很多的小块
    然后我们使用贪心策略，计算每两个相邻的区域的相似度
    然后每次合并最相似的两块，直到最终只剩下一块完整的图片
    然后这其中每次产生的图像块包括合并的图像块我们都保存下来
    
    Parameters
    ----------
        im_orig : ndarray
            Input image
        scale : int
            Free parameter. Higher means larger clusters in felzenszwalb segmentation.
        sigma : float
            Width of Gaussian kernel for felzenszwalb segmentation.
        min_size : int
            Minimum component size for felzenszwalb segmentation.
    Returns
    -------
        img : ndarray
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions : array of dict
            [
                {
                    'rect': (left, top, right, bottom),
                    'labels': [...]
                },
                ...
            ]
    '''
    assert im_orig.shape[2] == 3, "3ch image is expected"
    
    # load image and get smallest regions
    # region label is stored in the 4th value of each pixel [r,g,b,(region)]
    img = _generate_segments(im_orig, scale, sigma, min_size)
    if img is None:
        return None, ()
    
    imsize = img.shape[0] * img.shape[1]
    R = _extract_regions(img) # R:dict
    
    # extract neighbouring information
    neighbours = _extract_neighbours(R) # list:[(index，{property_dict}),...]
    
    # calculate initial similarities
    S = {}
    for (ri, ri_property), (rj, rj_property) in neighbours:
        S[(ri, rj)] = _calc_sim(ri_property, rj_property, imsize)
        
    # hierarchal search
    while S:
        # get highest similarity
        # i, j = sorted(S.items(), cmp=lambda a, b: cmp(a[1], b[1]))[-1][0]
        i, j = sorted(list(S.items()), key = lambda a: a[1], reverse = True)[0][0]
        
        # merge corresponding regions
        t = max(R.keys()) + 1.0
        R[t] = _merge_regions(R[i], R[j]) # R = R U rt
        
        # mark similarities for regions to be removed
        key_to_delete = []
        for k, v in S.items():
            if (i in k) or (j in k):
                key_to_delete.append(k)
                
        # remove old similarities of related regions
        for k in key_to_delete:
            del S[k]
        
        # calculate similarity set St between rt and its neighbours
        for k in filter(lambda a: a != (i, j), key_to_delete):
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = _calc_sim(R[t], R[n], imsize) # S = S U st
            
    regions = []
    for k, r in R.items():
        regions.append({
                'rect': (
                        r['min_x'], r['min_y'], 
                        r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
                'size': r['size'], 
                'labels': r['labels']
                })
                
    return img, regions
        

