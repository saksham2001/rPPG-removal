'''
This script provides the metrics to quantify the loss of information in the image due to the removal of rPPG signal.

Authors: Saksham Bhutani, Mohamed Elgendi and Carlo Menon
License: MIT
'''

import numpy as np

def mse(imageA, imageB):
    '''
    This function calculates the Mean Square Error between two images.

    Parameters:
        imageA (numpy.ndarray): First image.
        imageB (numpy.ndarray): Second image.

    Returns:
        err (float): Mean Square Error.
    '''
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err


def boilerplate_metric(imageA, imageB):
    '''
    This function is a boilerplate for the metrics for comparision of images to be added.

    Parameters:
        imageA (numpy.ndarray): First image.
        imageB (numpy.ndarray): Second image.

    Returns:
        metric (float): Metric value.
    '''
    
    # do the measurement here!

    # return the estimated (float) value
    pass