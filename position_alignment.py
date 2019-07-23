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

# function to get length matched
def get_length_adjusted(posX1, posY1, posX2, posY2):
    max_length = max(len(posX1),len(posX2))
    if len(posX1)==max_length:
        posX2 = np.append(posX2, [-1]*(max_length - len(posX2)))
        posY2 = np.append(posY2, [-1]*(max_length - len(posY2)))
    else:
        posX1 = np.append(posX1, [-1]*(max_length - len(posX1)))
        posY1 = np.append(posY1, [-1]*(max_length - len(posY1)))
    posX1 = np.array(posX1, dtype=np.float32)
    posY1 = np.array(posY1, dtype=np.float32)
    posX2 = np.array(posX2, dtype=np.float32)
    posY2 = np.array(posY2, dtype=np.float32)
    return posX1, posY1, posX2, posY2
        
# find intersecting coordinates between 2 camera position
def find_intersect_coords(posX1, posY1, posX2, posY2):
    # find lengt adjusted coords
    posX1, posY1, posX2, posY2 = get_length_adjusted(posX1, posY1, posX2, posY2)
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
def get_transformed_coords(c1_posX, c1_posY, c2_posX, c2_posY, unocc_c1, unocc_c2):
    # find preprocessed and intersecting coordinates bw 2 cameras
    intersect_coords = find_intersect_coords(c1_posX, c1_posY, c2_posX, c2_posY)
    c1_posX_c = intersect_coords['posX1_c']
    c1_posY_c = intersect_coords['posY1_c']
    c2_posX_c = intersect_coords['posX2_c']
    c2_posY_c = intersect_coords['posY2_c']
    # change -1 to nan
    c1_posX = intersect_coords['posX1']
    c1_posY = intersect_coords['posY1']
    c2_posX = intersect_coords['posX2']
    c2_posY = intersect_coords['posY2']
    c1_posX[c1_posX==-1] = np.nan
    c1_posY[c1_posY==-1] = np.nan
    c2_posX[c2_posX==-1] = np.nan
    c2_posY[c2_posY==-1] = np.nan
    # create the compatible pos coordinates data format for a single camera that needs to be transformed
    c1_coords = create_compatible_array(c1_posX, c1_posY)
    # create source and destination use to calculate homography
    c1_coords_c, c2_coords_c = create_src_dst_points(c1_posX_c, c1_posY_c, c2_posX_c, c2_posY_c)
    # find the transformed coordinates
    hg, status = cv2.findHomography(c1_coords_c, c2_coords_c)
    # perform perspective transformation
    c1_coords_t = cv2.perspectiveTransform(np.array([c1_coords]), hg)[0]
    # transformed points
    c1_posX_t = c1_coords_t[:,0]
    c1_posY_t = c1_coords_t[:,1]
    # change unoccupied pixels to nan
    c1_posX_t[unocc_c1] = np.nan
    c1_posY_t[unocc_c1] = np.nan
    c2_posX[unocc_c2] = np.nan
    c2_posY[unocc_c2] = np.nan
    # return the homography, trasnformed coordinates
    return intersect_coords, hg, c1_posX_t, c1_posY_t, c2_posX, c2_posY

# load the position data
_, cam4_posX, cam4_posY, _, _ = load_pos_data('cam3_Pos.mat')
_, cam3_posX, cam3_posY, _, _ = load_pos_data('cam2_Pos.mat')
# find unoccupied pixels
unocc_c1 = np.where(cam3_posX==-1)[0]
unocc_c2 = np.where(cam4_posX==-1)[0]

plt.figure(1)
plt.scatter(cam4_posX, cam4_posY, s=1)
plt.scatter(cam3_posX+400, cam3_posY, s=1)
plt.show()

# find the homography and transformed coordinates
common_coords_c3c4, hg_c3c4, cam3_posX_t, cam3_posY_t, cam4_posX, cam4_posY = get_transformed_coords(cam3_posX, cam3_posY, cam4_posX, cam4_posY, unocc_c1, unocc_c2)
merged_cam_posX = np.nanmean(np.transpose(np.vstack((cam3_posX_t, cam4_posX))), axis=1)
merged_cam_posY = np.nanmean(np.transpose(np.vstack((cam3_posY_t, cam4_posY))), axis=1)

plt.figure(2)
plt.scatter(cam4_posX, cam4_posY, s=1)
plt.scatter(cam3_posX_t, cam3_posY_t, s=1)
plt.scatter(merged_cam_posX, merged_cam_posY, s=1)
plt.show()

