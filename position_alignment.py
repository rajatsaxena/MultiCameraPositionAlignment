#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 13:49:06 2019

@author: rajat
"""

import cv2
import glob
import numpy as np
import scipy.io as scsio
import matplotlib.pyplot as plt

# load the position data
def load_pos_data(filename):
    data = scsio.loadmat(filename)
    posX = data['pos_x'][0]
    posY = data['pos_y'][0]
    redX = data['red_x'][0]
    redY = data['red_y'][0]
    return data, posX, posY, redX, redY

# find intersecting coordinates between 2 camera position
def intersect_coords(posX1, posY1, posX2, posY2):
    max_length = max(len(cam3_posX),len(cam4_posX))
    if len(posX1)==max_length:
        posX2 = np.append(posX2, [-1]*(max_length - len(posX2)))
        posY2 = np.append(posY2, [-1]*(max_length - len(posY2)))
    else:
        posX1 = np.append(posX1, [-1]*(max_length - len(posX1)))
        posY1 = np.append(posY1, [-1]*(max_length - len(posY1)))
    # return common indices
    indices = np.where((posX1!=-1) & (posY1!=-1) & (posX2!=-1) & (posY2!=-1))[0]
    # merged dictionary
    data = {}
    data['posX1'] = posX1
    data['posY1'] = posY1
    data['posX2'] = posX2
    data['posY2'] = posY2
    data['common_ind'] = indices
    data['posX1_c'] = posX1[indices]
    data['posY1_c'] = posY1[indices]
    data['posX2_c'] = posX2[indices]
    data['posY2_c'] = posY2[indices]
    # return the length adjusted position and indices
    return data

# funtions to generate the arrays to calculate homography
def create_src_dst_points(srcptX, srcptY, dstptX, dstptY):
    pts_src = []
    pts_dst = []
    for x1,y1,x2,y2 in zip(srcptX, srcptY, dstptX, dstptY):
        pts_src.append([x1,y1])
        pts_dst.append([x2,y2])
    return np.array(pts_src), np.array(pts_dst)

# load the position data
_, cam4_posX, cam4_posY, _, _ = load_pos_data('cam4_Pos.mat')
_, cam3_posX, cam3_posY, _, _ = load_pos_data('cam3_Pos.mat')

#quick HACK for reflection
cam4_posY[cam4_posY>=440] = cam4_posY[cam4_posY>=440] - 25

plt.figure(1)
plt.scatter(cam4_posX, cam4_posY, c='b', s=1)
plt.scatter(cam3_posX+440, cam3_posY, c='r', s=1)
plt.show()

# find preprocessed and intersecting coordinates bw 2 cameras
preprocessed_data = intersect_coords(cam3_posX, cam3_posY, cam4_posX, cam4_posY)
# common x,y coordinates
cam3_posX_c = preprocessed_data['posX1_c']
cam3_posY_c = preprocessed_data['posY1_c']
cam4_posX_c = preprocessed_data['posX2_c']
cam4_posY_c = preprocessed_data['posY2_c']

plt.figure(2)
plt.scatter(cam4_posX_c, cam4_posY_c, c='b', s=1)
plt.scatter(cam3_posX_c+440, cam3_posY_c, c='r', s=1)
plt.show()

# coordinates used to calculate the homography
cam3_coords, cam4_coords = create_src_dst_points(cam3_posX_c, cam3_posY_c, cam4_posX_c, cam4_posY_c)

# Calculate Homography
h, status = cv2.findHomography(cam3_coords, cam4_coords)


# TODO
# add method to transform second camera coordinates to first camera reference frame
 
## Warp source image to destination based on homography
#im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
