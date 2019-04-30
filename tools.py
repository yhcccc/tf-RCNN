"""
Created on Tue Feb 26 16:17:48 2019

@author: yh
"""

import math
import sys
import skimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s: [%s%s]%d%%\t%d/%d' % (
            message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total)
    sys.stdout.write(r)
    sys.stdout.flush()
    
def show_rect(img_path, regions):
    fig, ax = plt.subplots(1, 1, figsize = (6, 6))
    # (height, width, channel)
    img = skimage.io.imread(img_path)
    ax.imshow(img)
    for x, y, w, h in regions:
        rect = mpatches.Rectangle(
                (x, y), w, h, fill = False, edgecolor = 'red', linewidth = 1)
        ax.add_patch(rect)
    plt.show()
        