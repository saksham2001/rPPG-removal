'''
This script provides function to compute the any metric between two videos (i.e. original and processed).

Authors: Saksham Bhutani, Mohamed Elgendi and Carlo Menon
License: MIT
'''

import cv2
import numpy as np

def compute_errors(path1, path2, metric, notebook_mode=False):
    """
    This function computes the error between two videos.

    Parameters:
        path1 (str): Path of the first video.
        path2 (str): Path of the second video.
        metric (function): Metric to be computed.
        notebook_mode (bool): Whether to run in notebook mode.

    Returns:
        error (float): Error between the two videos.
    """
    print(path2)
    cap1 = cv2.VideoCapture(path1)
    cap2 = cv2.VideoCapture(path2)
    
    metric_last = []
    
    vid1_length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    vid2_length = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

    # check if the videos are of same length
    if vid1_length != vid2_length:
        print('Videos are of different lengths')
        return None
    
    print('Computing metric between videos...')
    
    # add tqdm progress bar
    if notebook_mode:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    
    with tqdm(total=vid1_length) as pbar:
        # read until end of video
        while(cap1.isOpened() or cap2.isOpened()):
            # capture each frame of the videos
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if ret1 and ret2:
                metric_val = metric(frame1, frame2)

                metric_last.append(metric_val)
                pbar.update(1)
                pbar.set_description('Metric Current: {:.2f} | Avg: {:.2f}'.format(metric_val, np.mean(metric_last)))
            # if no frame found
            else:
                break
    # release VideoCapture()
    cap1.release()
    cap2.release()
    
    # return np.mean(metric_last), np.mean(ssim_lst)
    return np.mean(metric_last)