# Multi Camera Position Alignment

scripts to merge position tracking data from multiple cameras (n=8 in this case). Takes .mat file from 8 different cameras as input. The .mat files should have 'pos_x' and 'pos_y' variables stored. Use mcpa.py when you are running the script with a new data and mcpa_reload.py when you are using already saved homography matrices.

https://github.com/rajatsaxena/rajatsaxena.github.io/blob/master/_posts/2019-07-26-Position-Data-Alignment-Multiple-Cameras.md


**Requirements**
- OpenCV
- Numpy
- Scipy
- Matplotlib
- Scikit-image
