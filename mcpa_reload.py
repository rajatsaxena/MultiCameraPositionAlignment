#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 13:49:06 2019
@author: rajat
"""

# import the modules used
import cv2
import numpy as np
import scipy.io as scsio
import matplotlib.pyplot as plt

# load the position data from mat file
def load_pos_data(filename, loadRedPos):
    data = scsio.loadmat(filename)
    posX = data['pos_x'][0]
    posY = data['pos_y'][0]
    if loadRedPos and 'red_x' in data.keys() and 'red_y' in data.keys():
        redX = data['red_x'][0]
        redY = data['red_y'][0]
        return data, redX, redY, posX, posY
    else:
        return data, np.array(posX, dtype=np.float32), np.array(posY, dtype=np.float32), np.nan, np.nan

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
        
# function to create compatible array
def create_compatible_array(posX, posY):
    pts_ = []
    for x1,y1 in zip(posX, posY):
        pts_.append([x1,y1])
    # convert them to float32 datatype
    return np.array(pts_, dtype=np.float32)

# get perspective transformed data
def get_transformed_coords_v2(c1_posX, c1_posY, c2_posX, c2_posY, unocc_c1, unocc_c2, hg_cAcB):
     # create the compatible pos coordinates data format for a single camera that needs to be transformed
    c1_coords = create_compatible_array(c1_posX, c1_posY)
    # perform perspective transformation
    c1_coords_t = cv2.perspectiveTransform(np.array([c1_coords]), hg_cAcB)[0]
    # transformed points
    c1_posX_t = c1_coords_t[:,0]
    c1_posY_t = c1_coords_t[:,1]
    # change unoccupied pixels to nan
    c1_posX_t[unocc_c1] = np.nan
    c1_posY_t[unocc_c1] = np.nan
    c2_posX[unocc_c2] = np.nan
    c2_posY[unocc_c2] = np.nan
    # return the homography, transformed coordinates
    return c1_posX_t, c1_posY_t, c2_posX, c2_posY

# function to get reloaded camera positions
def get_merged_cams(camA_posX, camA_posY, camB_posX, camB_posY, hg_cAcB):
    # find lengt adjusted coords
    camA_posX, camA_posY, camB_posX, camB_posY = get_length_adjusted(camA_posX, camA_posY, camB_posX, camB_posY)
    # find unoccupied pixels
    unocc_cA = np.where(camA_posX==-1)[0]
    unocc_cB = np.where(camB_posX==-1)[0]
    # find the transformed coordinates
    camA_posX_t, camA_posY_t, camB_posX, camB_posY \
                = get_transformed_coords_v2(camA_posX, camA_posY, camB_posX, camB_posY, unocc_cA, unocc_cB, hg_cAcB)
    merged_cam_posX = np.nanmean(np.transpose(np.vstack((camA_posX_t, camB_posX))), axis=1)
    merged_cam_posY = np.nanmean(np.transpose(np.vstack((camA_posY_t, camB_posY))), axis=1)   
    # set nan to -1 (not a good coding practice)
    merged_cam_posX[np.isnan(merged_cam_posX)] = -1.
    merged_cam_posY[np.isnan(merged_cam_posY)] = -1.
    # return the merged cam position
    return camA_posX_t, camA_posY_t, camB_posX, camB_posY, merged_cam_posX, merged_cam_posY

# all camera posiiton filename
cam1_pos_filename = 'cam1_Pos.mat'
cam2_pos_filename = 'cam2_Pos.mat'
cam3_pos_filename = 'cam3_Pos.mat'
cam4_pos_filename = 'cam4_Pos.mat'
cam5_pos_filename = 'cam5_Pos.mat'
cam6_pos_filename = 'cam6_Pos.mat'
cam7_pos_filename = 'cam7_Pos.mat'
cam8_pos_filename = 'cam8_Pos.mat'

# load the position data for each camera 
_, cam1_posX, cam1_posY, _, _ = load_pos_data(cam1_pos_filename, True)
_, cam2_posX, cam2_posY, _, _ = load_pos_data(cam2_pos_filename, False)
_, cam3_posX, cam3_posY, _, _ = load_pos_data(cam3_pos_filename, False)    
_, cam4_posX, cam4_posY, _, _ = load_pos_data(cam4_pos_filename, False)
_, cam5_posX, cam5_posY, _, _ = load_pos_data(cam5_pos_filename, True)
_, cam6_posX, cam6_posY, _, _ = load_pos_data(cam6_pos_filename, False)
_, cam7_posX, cam7_posY, _, _ = load_pos_data(cam7_pos_filename, False)    
_, cam8_posX, cam8_posY, _, _ = load_pos_data(cam8_pos_filename, False)

# load the combined position data
homography_d = np.load('homography.npy')
homography_d = homography_d.item()

# function to find merged cam2 and cam3 data
cam2_posX_t, cam2_posY_t, cam3_posX, cam3_posY, merged_cam23_posX,\
        merged_cam23_posY = get_merged_cams(cam2_posX, cam2_posY, cam3_posX, cam3_posY, homography_d['cam2cam3'])
cam1_posX_t, cam1_posY_t, merged_cam23_posX, merged_cam23_posY, merged_cam123_posX, \
        merged_cam123_posY = get_merged_cams(cam1_posX, cam1_posY, merged_cam23_posX, merged_cam23_posY, homography_d['cam1cam2cam3'])
cam4_posX_t, cam4_posY_t, merged_cam123_posX, merged_cam123_posY, merged_cam1234_posX, \
        merged_cam1234_posY = get_merged_cams(cam4_posX, cam4_posY, merged_cam123_posX, merged_cam123_posY, homography_d['cam1cam2cam3cam4'])

# dict to hold the transformed coordinates
transformed_coords = {}
transformed_coords['cam1_posX'] = cam1_posX_t
transformed_coords['cam1_posY'] = cam1_posY_t
transformed_coords['cam2_posX'] = cam2_posX_t
transformed_coords['cam2_posY'] = cam2_posY_t
transformed_coords['cam4_posX'] = cam4_posX_t
transformed_coords['cam4_posY'] = cam4_posY_t

# dictionary to hold the merged coordinates
merged_coords = {}
merged_coords['cam2_cam3_posX'] = merged_cam23_posX
merged_coords['cam2_cam3_posY'] = merged_cam23_posY
merged_coords['cam1_cam2_cam3_posX'] = merged_cam123_posX
merged_coords['cam1_cam2_cam3_posY'] = merged_cam123_posY
merged_coords['cam1_cam2_cam3_cam4_posX'] = merged_cam1234_posX
merged_coords['cam1_cam2_cam3_cam4_posY'] = merged_cam1234_posY

# plot the processed data
plt.figure(1)
plt.scatter(cam3_posX, cam3_posY, s=1)
plt.scatter(cam2_posX_t, cam2_posY_t, s=1)
plt.scatter(merged_cam23_posX, merged_cam23_posY, s=1)
plt.title('cam2 and cam3')
plt.show()

plt.figure(2)
plt.scatter(merged_cam23_posX, merged_cam23_posY, s=1)
plt.scatter(cam1_posX_t, cam1_posY_t, s=1)
plt.scatter(merged_cam123_posX, merged_cam123_posY, s=1)
plt.title('cam1, cam2 and cam3')
plt.show()
    
plt.figure(3)
plt.scatter(merged_cam123_posX, merged_cam123_posY, s=1)
plt.scatter(cam4_posX_t, cam4_posY_t, s=1)
plt.scatter(merged_cam1234_posX, merged_cam1234_posY, s=1)
plt.title('cam1, cam2, cam3 and cam4')
plt.show()

plt.figure(4)
plt.scatter(cam1_posX_t, cam1_posY_t, s=1)
plt.scatter(cam2_posX_t, cam2_posY_t, s=1)
plt.scatter(cam3_posX, cam3_posY, s=1)
plt.scatter(cam4_posX_t, cam4_posY_t, s=1)
plt.title('cam1, cam2, cam3 and cam4')
plt.show()

# get the trasnformed data, homography matrix and matching vertices
cam6_posX_t, cam6_posY_t, cam7_posX, cam7_posY, merged_cam67_posX, merged_cam67_posY\
         = get_merged_cams(cam6_posX, cam6_posY, cam7_posX, cam7_posY, homography_d['cam6cam7'])
cam5_posX_t, cam5_posY_t, merged_cam67_posX, merged_cam67_posY, merged_cam567_posX, \
        merged_cam567_posY = get_merged_cams(cam5_posX, cam5_posY, merged_cam67_posX, merged_cam67_posY, homography_d['cam5cam6cam7'])
cam8_posX_t, cam8_posY_t, merged_cam567_posX, merged_cam567_posY, merged_cam5678_posX, \
        merged_cam5678_posY = get_merged_cams(cam8_posX, cam8_posY, merged_cam567_posX, merged_cam567_posY, homography_d['cam5cam6cam7cam8'])

# dict to hold the transformed coordinates
transformed_coords['cam5_posX'] = cam5_posX_t
transformed_coords['cam5_posY'] = cam5_posY_t
transformed_coords['cam6_posX'] = cam6_posX_t
transformed_coords['cam6_posY'] = cam6_posY_t
transformed_coords['cam8_posX'] = cam8_posX_t
transformed_coords['cam8_posY'] = cam8_posY_t

# dictionary to hold the merged coordinates
merged_coords['cam6_cam7_posX'] = merged_cam67_posX
merged_coords['cam6_cam7_posY'] = merged_cam67_posY
merged_coords['cam5_cam6_cam7_posX'] = merged_cam567_posX
merged_coords['cam5_cam6_cam7_posY'] = merged_cam567_posY
merged_coords['cam5_cam6_cam7_cam8_posX'] = merged_cam5678_posX
merged_coords['cam5_cam6_cam7_cam8_posY'] = merged_cam5678_posY

# plot the processed data
plt.figure(5)
plt.scatter(cam7_posX, cam7_posY, s=1)
plt.scatter(cam6_posX_t, cam6_posY_t, s=1)
plt.scatter(merged_cam67_posX, merged_cam67_posY, s=1)
plt.title('cam6 and cam7')
plt.show()

plt.figure(6)
plt.scatter(merged_cam67_posX, merged_cam67_posY, s=1)
plt.scatter(cam5_posX_t, cam5_posY_t, s=1)
plt.scatter(merged_cam567_posX, merged_cam567_posY, s=1)
plt.title('cam5, cam6 and cam7')
plt.show()

plt.figure(7)
plt.scatter(merged_cam567_posX, merged_cam567_posY, s=1)
plt.scatter(cam8_posX_t, cam8_posY_t, s=1)
plt.scatter(merged_cam5678_posX, merged_cam5678_posY, s=1)
plt.title('cam5, cam6, cam7 and cam8')
plt.show()

plt.figure(8)
plt.scatter(cam5_posX_t, cam5_posY_t, s=1)
plt.scatter(cam6_posX_t, cam6_posY_t, s=1)
plt.scatter(cam7_posX, cam7_posY, s=1)
plt.scatter(cam8_posX_t, cam8_posY_t, s=1)
plt.title('cam5, cam6, cam7 and cam8 transformed')
plt.show()

# run stitching on merged cam1234 and merged cam5678 to merge all the cameras
merged_cam1234_posX_t, merged_cam1234_posY_t, merged_cam5678_posX, merged_cam5678_posY, \
    merged_allcams_posX, merged_allcams_posY = get_merged_cams(merged_cam1234_posX, merged_cam1234_posY, \
                                                               merged_cam5678_posX, merged_cam5678_posY, homography_d['cam14cam58'])

# dictionary to hold the merged coordinates
merged_coords['cam1234_posX_t'] = merged_cam1234_posX_t
merged_coords['cam1234_posY_t'] = merged_cam1234_posY_t
merged_coords['cam5678_posX'] = merged_cam5678_posX
merged_coords['cam5678_posY'] = merged_cam5678_posY
merged_coords['cam14cam58_posX'] = merged_allcams_posX
merged_coords['cam14cam58_posY'] = merged_allcams_posY

# plot all the merged camera
plt.figure(9)
plt.scatter(merged_cam1234_posX_t, merged_cam1234_posY_t, s=1)
plt.scatter(merged_cam5678_posX, merged_cam5678_posY, s=1)
plt.scatter(merged_allcams_posX, merged_allcams_posY, s=1)
plt.title('All cameras merged')
plt.show()

np.save('transformed_coords.npy', transformed_coords)
np.save('merged_coords.npy', merged_coords)