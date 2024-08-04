import cv2
import torch
import os
import collections
from collections import Counter
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os, cv2
import torchvision.transforms as transforms
from math import exp
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def conv_x(img):
    sobel = torch.Tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    filter = torch.reshape(sobel, [1, 1, 3, 3])
    filter = filter.cuda()
    gx = F.conv2d(img, filter, stride=1, padding=1)
    return gx

def entropy(ori_img, areas_sum):
    unsup_img = transforms.ToTensor()(ori_img)
    unsup_img = unsup_img.unsqueeze(0)
    unsup_img = unsup_img.cuda()
    new_img = conv_x(unsup_img)

    new_img = np.squeeze((new_img * 255 / 8).cpu().data.numpy())
    img = torch.from_numpy(ori_img)
    compare_list = []
    for m in range(1, img.size()[0] - 1):
        for n in range(1, img.size()[1] - 1):
            pix = int(img[m, n])
            mean_element = int(new_img[m, n])
            temp = (pix, mean_element)
            compare_list.append(temp)

    # print(compare_list)
    compare_dict = collections.Counter(compare_list)
    H = 0.0
    for freq in compare_dict.values():
        f_n2 = freq / (img.size()[0] * img.size()[1])
        log_f_n2 = math.log(f_n2)
        h = -(f_n2 * log_f_n2)
        H += h

    return H / math.sqrt(areas_sum)

def entropy_pick(im, th_pix=15):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(im, connectivity=8, ltype=None)
    areas_list = []
    areas_sum = 0
    img = torch.from_numpy(im)
    if num_labels == 1:
        return 0, 0
    else:
        for i in range(1, num_labels):
            areas = stats[i][4]
            areas_sum += areas
            areas_list.append(areas)
        max_area = max(areas_list)
        sal_area = max_area // 20
        j = 0
        for area_i in areas_list:
            if area_i >= sal_area and area_i <= (5 * sal_area):
                j = j + 1

        if j != 0 or max_area <= th_pix or (max_area <= 300 and (num_labels - 1) > 3):
            return 0, 0
        else:
            sh = entropy(im, areas_sum)
            ratio = areas_sum / (img.size()[0] * img.size()[1])
            return sh, ratio
        