import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def calc_X(cam1_redX,cam2_redX,cam3_redX, cam4_redX, cam5_redX,cam6_redX,cam7_redX,cam8_redX):
    red_x = np.array([])
    for i in range(len(cam3_redX)):
        if cam1_redX[i]==cam2_redX[i]==cam3_redX[i]==cam4_redX[i]==cam5_redX[i]==cam6_redX[i]==cam7_redX[i]==cam8_redX[i]==-1:
            redX=-1
        elif cam2_redX[i]!=-1:
            redX=cam2_redX[i]+835
        elif cam2_redX[i]==-1 and cam3_redX[i]!=-1:
            redX=cam3_redX[i]+440
        elif cam2_redX[i]==cam3_redX[i]==-1 and cam4_redX[i]!=-1:
            redX=cam4_redX[i]
        elif cam2_redX[i]==cam3_redX[i]==cam4_redX[i] and cam5_redX[i]!=-1:
            redX=cam5_redX[i]+1090
        elif cam2_redX[i]==cam3_redX[i]==cam4_redX[i]==cam5_redX[i]==-1 and cam6_redX[i]!=-1:
            redX=cam6_redX[i]+755
        elif cam2_redX[i]==cam3_redX[i]==cam4_redX[i]==cam5_redX[i]==cam6_redX[i]==-1 and cam7_redX[i]!=-1:
            redX=cam7_redX[i]+445
        elif cam2_redX[i]==cam3_redX[i]==cam4_redX[i]==cam5_redX[i]==cam6_redX[i]==cam7_redX[i]==-1 and cam8_redX[i]!=-1:
            redX=cam8_redX[i]+68
        elif cam1_redX[i]!=-1:
            redX=cam1_redX[i]+1070
        else:
            redX=-1
        #print redX, cam1_redX[i]==cam2_redX[i]==cam3_redX[i]==cam4_redX[i]==cam5_redX[i]==cam6_redX[i],cam7_redX[i],cam8_redX[i]==-1
        red_x = np.append(red_x,round(float(redX),3))
    
    return red_x

def calc_Y(cam1_redY,cam2_redY,cam3_redY, cam4_redY, cam5_redY,cam6_redY,cam7_redY,cam8_redY):
    red_y = np.array([])
    for i in range(len(cam3_redY)):
        if cam1_redY[i]==cam2_redY[i]==cam3_redY[i]==cam4_redY[i]==cam5_redY[i]==cam6_redY[i]==cam7_redY[i]==cam8_redY[i]==-1:
            redY=-1
        elif cam2_redY[i]!=-1:
            redY=cam2_redY[i]+407
        elif cam2_redY[i]==-1 and cam3_redY[i]!=-1:
            redY=cam3_redY[i]+405
        elif cam2_redY[i]==cam3_redY[i]==-1 and cam4_redY[i]!=-1:
            redY=cam4_redY[i]+405
        elif cam2_redY[i]==cam3_redY[i]==cam4_redY[i] and cam5_redY[i]!=-1:
            redY=cam5_redY[i]+50
        elif cam2_redY[i]==cam3_redY[i]==cam4_redY[i]==cam5_redY[i]==-1 and cam6_redY[i]!=-1:
            redY=cam6_redY[i]+45
        elif cam2_redY[i]==cam3_redY[i]==cam4_redY[i]==cam5_redY[i]==cam6_redY[i]==-1 and cam7_redY[i]!=-1:
            redY=cam7_redY[i]+28
        elif cam2_redY[i]==cam3_redY[i]==cam4_redY[i]==cam5_redY[i]==cam6_redY[i]==cam7_redY[i]==-1 and cam8_redY[i]!=-1:
            redY=cam8_redY[i]
        elif cam1_redY[i]!=-1:
            redY=cam1_redY[i]+415
        else:
            redY=-1
        red_y = np.append(red_y,round(float(redY),3))
        
    return red_y

cam1PosData = sio.loadmat('cam1_Pos.mat')
cam1X = cam1PosData['pos_x'][0]
cam1Y = cam1PosData['pos_y'][0]
cam1redX = cam1PosData['red_x'][0]
cam1redY = cam1PosData['red_y'][0]
cam1greenX = cam1PosData['green_x'][0]
cam1greenY = cam1PosData['green_y'][0]

#plt.figure()
#plt.scatter(cam1X, cam1Y)

cam2PosData = sio.loadmat('cam2_Pos.mat')
cam2X = cam2PosData['pos_x'][0]
cam2Y = cam2PosData['pos_y'][0]
cam2redX = cam2PosData['red_x'][0]
cam2redY = cam2PosData['red_y'][0]
cam2greenX = cam2PosData['green_x'][0]
cam2greenY = cam2PosData['green_y'][0]

#plt.figure()
#plt.scatter(cam2X, cam2Y)

cam3PosData = sio.loadmat('cam3_Pos.mat')
cam3X = cam3PosData['pos_x'][0]
cam3Y = cam3PosData['pos_y'][0]
cam3redX = cam3PosData['red_x'][0]
cam3redY = cam3PosData['red_y'][0]
cam3greenX = cam3PosData['green_x'][0]
cam3greenY = cam3PosData['green_y'][0]

#plt.figure()
#plt.scatter(cam3X, cam3Y)

cam4PosData = sio.loadmat('cam4_Pos.mat')
cam4X = cam4PosData['pos_x'][0]
cam4Y = cam4PosData['pos_y'][0]
cam4redX = cam4PosData['red_x'][0]
cam4redY = cam4PosData['red_y'][0]
cam4greenX = cam4PosData['green_x'][0]
cam4greenY = cam4PosData['green_y'][0]

#plt.figure()
#plt.scatter(cam4X, cam4Y)

cam5PosData = sio.loadmat('cam5_Pos.mat')
cam5X = cam5PosData['pos_x'][0]
cam5Y = cam5PosData['pos_y'][0]
cam5redX = cam5PosData['red_x'][0]
cam5redY = cam5PosData['red_y'][0]
cam5greenX = cam5PosData['green_x'][0]
cam5greenY = cam5PosData['green_y'][0]

#plt.figure()
#plt.scatter(cam5X, cam5Y)

cam6PosData = sio.loadmat('cam6_Pos.mat')
cam6X = cam6PosData['pos_x'][0]
cam6Y = cam6PosData['pos_y'][0]
cam6redX = cam6PosData['red_x'][0]
cam6redY = cam6PosData['red_y'][0]
cam6greenX = cam6PosData['green_x'][0]
cam6greenY = cam6PosData['green_y'][0]

#plt.figure()
#plt.scatter(cam6X, cam6Y)

cam7PosData = sio.loadmat('cam7_Pos.mat')
cam7X = cam7PosData['pos_x'][0]
cam7Y = cam7PosData['pos_y'][0]
cam7redX = cam7PosData['red_x'][0]
cam7redY = cam7PosData['red_y'][0]
cam7greenX = cam7PosData['green_x'][0]
cam7greenY = cam7PosData['green_y'][0]

#plt.figure()
#plt.scatter(cam7X, cam7Y)

cam8PosData = sio.loadmat('cam8_Pos.mat')
cam8X = cam8PosData['pos_x'][0]
cam8Y = cam8PosData['pos_y'][0]
cam8redX = cam8PosData['red_x'][0]
cam8redY = cam8PosData['red_y'][0]
cam8greenX = cam8PosData['green_x'][0]
cam8greenY = cam8PosData['green_y'][0]

#plt.figure()
#plt.scatter(cam8X, cam8Y)

print "Working on position data"

minlength = min(np.shape(cam1X)[0],np.shape(cam2X)[0],np.shape(cam3X)[0],np.shape(cam4X)[0],np.shape(cam5X)[0],np.shape(cam6X)[0],np.shape(cam7X)[0],np.shape(cam8X)[0])

cam1X = cam1X[0:minlength]
cam1Y = cam1Y[0:minlength]
cam1redX = cam1redX[0:minlength]
cam1redY = cam1redY[0:minlength]
cam1greenX = cam1greenX[0:minlength]
cam1greenY = cam1greenY[0:minlength]
cam2X = cam2X[0:minlength]
cam2Y = cam2Y[0:minlength]
cam2redX = cam2redX[0:minlength]
cam2redY = cam2redY[0:minlength]
cam2greenX = cam2greenX[0:minlength]
cam2greenY = cam2greenY[0:minlength]
cam3X = cam3X[0:minlength]
cam3Y = cam3Y[0:minlength]
cam3redX = cam3redX[0:minlength]
cam3redY = cam3redY[0:minlength]
cam3greenX = cam3greenX[0:minlength]
cam3greenY = cam3greenY[0:minlength]
cam4X = cam4X[0:minlength]
cam4Y = cam4Y[0:minlength]
cam4redX = cam4redX[0:minlength]
cam4redY = cam4redY[0:minlength]
cam4greenX = cam4greenX[0:minlength]
cam4greenY = cam4greenY[0:minlength]
cam5X = cam5X[0:minlength]
cam5Y = cam5Y[0:minlength]
cam5redX = cam5redX[0:minlength]
cam5redY = cam5redY[0:minlength]
cam5greenX = cam5greenX[0:minlength]
cam5greenY = cam5greenY[0:minlength]
cam6X = cam6X[0:minlength]
cam6Y = cam6Y[0:minlength]
cam6redX = cam6redX[0:minlength]
cam6redY = cam6redY[0:minlength]
cam6greenX = cam6greenX[0:minlength]
cam6greenY = cam6greenY[0:minlength]
cam7X = cam7X[0:minlength]
cam7Y = cam7Y[0:minlength]
cam7redX = cam7redX[0:minlength]
cam7redY = cam7redY[0:minlength]
cam7greenX = cam7greenX[0:minlength]
cam7greenY = cam7greenY[0:minlength]
cam8X = cam8X[0:minlength]
cam8Y = cam8Y[0:minlength]
cam8redX = cam8redX[0:minlength]
cam8redY = cam8redY[0:minlength]
cam8greenX = cam8greenX[0:minlength]
cam8greenY = cam8greenY[0:minlength]

print 'Function calling'

redLedX = calc_X(cam1redX,cam2redX,cam3redX,cam4redX,cam5redX,cam6redX,cam7redX,cam8redX)
posLedX = calc_X(cam1X,cam2X,cam3X,cam4X,cam5X,cam6X,cam7X,cam8X)
greenLedX = calc_X(cam1greenX,cam2greenX,cam3greenX,cam4greenX,cam5greenX,cam6greenX,cam7greenX,cam8greenX)

redLedY = calc_Y(cam1redY,cam2redY,cam3redY,cam4redY,cam5redY,cam6redY,cam7redY,cam8redY)
posLedY = calc_Y(cam1Y,cam2Y,cam3Y,cam4Y,cam5Y,cam6Y,cam7Y,cam8Y)
greenLedY = calc_Y(cam1greenY,cam2greenY,cam3greenY,cam4greenY,cam5greenY,cam6greenY,cam7greenY,cam8greenY)


sio.savemat('RatC5_Day56_Pos.mat', mdict={'red_x':redLedX,'red_y':redLedY,'pos_x':posLedX,'pos_y':posLedY, 'green_x':greenLedX, 'green_y':greenLedY})