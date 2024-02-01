'''
This script provides function to compute the any metric between two videos (i.e. original and processed).

Authors: Saksham Bhutani, Mohamed Elgendi and Carlo Menon
License: MIT
'''

import cv2
import numpy as np

def compute_errors(path1, path2, metric):
    print(path2)
    cap1 = cv2.VideoCapture(path1)
    cap2 = cv2.VideoCapture(path2)
    
    mse_lst = []
    ssim_lst = []

    # read until end of video
    while(cap1.isOpened() or cap2.isOpened()):
        # capture each frame of the videos
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if ret1 and ret2:
            mse_val = metric(frame1, frame2)

            mse_lst.append(mse_val)
        # if no frame found
        else:
            break
    # release VideoCapture()
    cap1.release()
    cap2.release()
    
    # return np.mean(mse_lst), np.mean(ssim_lst)
    return np.mean(mse_lst)