'''
This script provides the metrics to quantify the speed of the algorithms.

Authors: Saksham Bhutani, Mohamed Elgendi and Carlo Menon
License: MIT
'''

import numpy as np

def fps(prev_time, new_time):
    '''
    This function calculates the frame per second (fps) of the filter.

    Parameters:
        prev_time (float): Previous time.
        new_time (float): New time.

    Returns:
        fps (float): Frames per second.
    '''
    fps = 1/(new_time - prev_time)

    return fps


def boilerplate_metric(prev_time, new_time):
    '''
    This function is a boilerplate for the metrics for comparision of images to be added.

    Parameters:
        prev_time (float): Previous time.
        new_time (float): New time.
    
    Returns:
        metric (float): Metric value.
    '''
    
    # do the measurement here!

    # return the estimated (float) value
    pass