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


# load the position data from mat file
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

# function to create compatible array
def create_compatible_array(posX, posY):
    pts_ = []
    for x1,y1 in zip(posX, posY):
        pts_.append([x1,y1])
    # convert them to float32 datatype
    return np.array(pts_, dtype=np.float32)

# funtions to generate the arrays to calculate homography
def create_src_dst_points(srcptX, srcptY, dstptX, dstptY):
    pts_src = create_compatible_array(srcptX, srcptY)
    pts_dst = create_compatible_array(dstptX, dstptY)
    return pts_src, pts_dst

# get perspective transformed data
def transformed_coords(c1_coords_c, c2_coords_c, c1_coords):
    # Calculate Homography
    hg, status = cv2.findHomography(c1_coords_c, c2_coords_c)
    # perform perspective transformation
    c1_coords_t = cv2.perspectiveTransform(np.array([c1_coords]), hg)[0]
    # return the homography, trasnformed coordinates
    return hg, c1_coords_t

# load the position data
_, cam4_posX, cam4_posY, _, _ = load_pos_data('cam3_Pos.mat')
_, cam3_posX, cam3_posY, _, _ = load_pos_data('cam2_Pos.mat')

#quick HACK for reflection
#cam4_posY[cam4_posY>=440] = cam4_posY[cam4_posY>=440] - 25

plt.figure(1)
plt.scatter(cam4_posX, cam4_posY, s=1)
plt.scatter(cam3_posX+400, cam3_posY, s=1)
plt.show()

# find preprocessed and intersecting coordinates bw 2 cameras
preprocessed_data = intersect_coords(cam3_posX, cam3_posY, cam4_posX, cam4_posY)
# common x,y coordinates
cam3_posX_c = preprocessed_data['posX1_c']
cam3_posY_c = preprocessed_data['posY1_c']
cam4_posX_c = preprocessed_data['posX2_c']
cam4_posY_c = preprocessed_data['posY2_c']

# create source and destination use to calculate homography
cam3_coords_c, cam4_coords_c = create_src_dst_points(cam3_posX_c, cam3_posY_c, cam4_posX_c, cam4_posY_c)
# create the compatible pos coordinates for a single camera that needs to be transformed 
cam3_coords = create_compatible_array(cam3_posX, cam3_posY)
# find the transformed coordinates
hg_c3c4, cam3_coords_t = transformed_coords(cam3_coords_c, cam4_coords_c, cam3_coords)
    
# transformed points
cam3_posX_T = cam3_coords_t[:,0]
cam3_posY_t = cam3_coords_t[:,1]

plt.figure(2)
plt.scatter(cam4_posX, cam4_posY, s=1)
plt.scatter(cam3_posX_T, cam3_posY_t, s=1)
plt.show()