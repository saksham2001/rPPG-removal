'''
This script provides the filters to apply on the incoming frame(s) to process the dataset.

Authors: Saksham Bhutani, Mohamed Elgendi and Carlo Menon
License: MIT
'''

import cv2
from skimage.util import random_noise
import numpy as np

def medianblur(frame, filter_params):
    '''
    This function applies median blur on the frame.

    Parameters:
        frame (numpy.ndarray): Frame to be processed.
        filter_params (dict): Dictionary containing the parameters for the filter. [Required]
            kernel_size (int): Kernel size for the filter.

    Returns:
        frame (numpy.ndarray): Processed frame.

    Reference:
        [1] https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=medianblur#medianblur
    '''
    kernel_size = filter_params['kernel_size']

    frame = cv2.medianBlur(frame, kernel_size)

    return frame

def gaussianblur(frame, filter_params):
    '''
    This function applies gaussian blur on the frame.

    Parameters:
        frame (numpy.ndarray): Frame to be processed.
        filter_params (dict): Dictionary containing the parameters for the filter. [Required]
            kernel_size (int): Kernel size for the filter.

    Returns:
        frame (numpy.ndarray): Processed frame.

    Reference:
        [1] https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=gaussianblur#gaussianblur
    '''
    kernel_size = filter_params['kernel_size']

    frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

    return frame

def bilateralblur(frame, filter_params):
    '''
    This function applies bilateral blur on the frame.

    Parameters:
        frame (numpy.ndarray): Frame to be processed.
        filter_params (dict): Dictionary containing the parameters for the filter. [Required]
            kernel_size (int): Kernel size for the filter.

    Returns:
        frame (numpy.ndarray): Processed frame.

    Referenece:
        [1] https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=bilateralfilter#bilateralfilter
    '''
    kernel_size = filter_params['kernel_size']

    frame = cv2.bilateralFilter(frame, kernel_size, 75, 75)

    return frame

def gaussiannoise(frame, filter_params=None):
    '''
    This function applies gaussian noise on the frame.

    Parameters:
        frame (numpy.ndarray): Frame to be processed.
        filter_params (dict): Dictionary containing the parameters for the filter. [Optional]
            mean (int): Mean of the distribution.
            sigma (int): Standard deviation of the distribution.

    Returns:
        frame (numpy.ndarray): Processed frame.

    Reference:
        [1] https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.random_noise
    '''
    if filter_params:
        mean = filter_params['mean']
        sigma = filter_params['sigma']

        frame = np.uint8(random_noise(frame, mode='gaussian', mean=mean, var=sigma**2)*255)
    else:
        frame = np.uint8(random_noise(frame, mode='gaussian')*255)

    return frame

def localvarnoise(frame, filter_params=None):
    '''
    This function applies gaussian noise with local variance noise on the frame.

    Parameters:
        frame (numpy.ndarray): Frame to be processed.
        filter_params (dict): Dictionary containing the parameters for the filter. [Optional]
            local_vars (numpy.ndarray): Array of the same shape as `frame` giving the local variance at each point.

    Returns:
        frame (numpy.ndarray): Processed frame.

    Reference:
        [1] https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.random_noise
    '''
    if filter_params:
        local_vars = filter_params['local_vars']

        frame = np.uint8(random_noise(frame, mode='localvar', local_vars=local_vars)*255)
    else:
        frame = np.uint8(random_noise(frame, mode='localvar')*255)

    return frame

def saltpeppernoise(frame, filter_params=None):
    '''
    This function applies salt and pepper noise on the frame.

    Parameters:
        frame (numpy.ndarray): Frame to be processed.
        filter_params (dict): Dictionary containing the parameters for the filter. [Optional]
            amount (float): Proportion of the image pixels to replace with noise.

    Returns:
        frame (numpy.ndarray): Processed frame.

    Reference:
        [1] https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.random_noise
    '''
    if filter_params:
        amount = filter_params['amount']

        frame = np.uint8(random_noise(frame, mode='s&p', amount=amount)*255)
    else:
        frame = np.uint8(random_noise(frame, mode='s&p')*255)

    return frame

def poissonnoise(frame, filter_params=None):
    '''
    This function applies poisson noise on the frame.

    Parameters:
        frame (numpy.ndarray): Frame to be processed.
        filter_params (dict): Dictionary containing the parameters for the filter.
            No parameters required

    Returns:
        frame (numpy.ndarray): Processed frame.

    Reference:
        [1] https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.random_noise
    '''
    frame = np.uint8(random_noise(frame, mode='poisson')*255)

    return frame

def peppernoise(frame, filter_params=None):
    '''
    This function applies pepper noise on the frame.

    Parameters:
        frame (numpy.ndarray): Frame to be processed.
        filter_params (dict): Dictionary containing the parameters for the filter. [Optional]
            amount (float): Proportion of the image pixels to replace with noise.

    Returns:
        frame (numpy.ndarray): Processed frame.

    Reference:
        [1] https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.random_noise
    '''
    if filter_params:
        amount = filter_params['amount']

        frame = np.uint8(random_noise(frame, mode='pepper', amount=amount)*255)
    else:
        frame = np.uint8(random_noise(frame, mode='pepper')*255)

    return frame

def specklenoise(frame, filter_params=None):
    '''
    This function applies speckle noise on the frame.

    Parameters:
        frame (numpy.ndarray): Frame to be processed.
        filter_params (dict): Dictionary containing the parameters for the filter.
            No parameters required

    Returns:
        frame (numpy.ndarray): Processed frame.

    Reference:
        [1] https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.random_noise
    '''
    frame = np.uint8(random_noise(frame, mode='speckle')*255)

    return frame

def timeblur(frame, filter_params):
    '''
    This function applies the time blur filter on the frames.

    Parameters:
        frame (numpy.ndarray): Frame to be processed.
        filter_params (dict): Dictionary containing the parameters for the filter.
            total (int): Total number of frames processed.
            rAvg (numpy.ndarray): Average of the red channel.
            gAvg (numpy.ndarray): Average of the green channel.
            bAvg (numpy.ndarray): Average of the blue channel.

    Returns:
        frame (numpy.ndarray): Processed frame.
    '''

    total_frames_proc = filter_params['total']
    rAvg_prev = filter_params['rAvg']
    gAvg_prev = filter_params['gAvg']
    bAvg_prev = filter_params['bAvg']

    # split each frame in rgb channels
    (B, G, R) = cv2.split(frame.astype("float"))

    if total_frames_proc == 1:
        rAvg = R
        gAvg = G
        bAvg = B
    else:
        # calculate the timeblur
        rAvg = ((total_frames_proc * rAvg_prev) + (1 * R)) / (total_frames_proc + 1.0)
        gAvg = ((total_frames_proc * gAvg_prev) + (1 * G)) / (total_frames_proc + 1.0)
        bAvg = ((total_frames_proc * bAvg_prev) + (1 * B)) / (total_frames_proc + 1.0)

        # merge the channels back together and convert back to image (uint8)
        frame = cv2.merge([bAvg, gAvg, rAvg]).astype("uint8")

    return frame, [rAvg, gAvg, bAvg]
    
def timeblur_sliding(frames_queue, filter_params):
    '''
    This function applies the time blur filter on the last f frames.

    Parameters:
        frames_queue (numpy.ndarray): List of all the frames.
        filter_params (dict): Dictionary containing the parameters for the filter. [Required]
            window_size (int): Size of the sliding window.

    Returns:
        frame (numpy.ndarray): Processed frame.
    '''

    window_size = filter_params['window_size']

    # split each frame in rgb channels
    r = []
    g = []
    b = []

    for frame in frames_queue:
        (B, G, R) = cv2.split(frame.astype("float"))

        r.append(R)
        g.append(G)
        b.append(B)

    # calculate the average of each channel
    rAvg = (sum(r)/len(r))
    gAvg = (sum(g)/len(g))
    bAvg = (sum(b)/len(b))

    # merge the channels back together and convert back to image (uint8)
    frame = cv2.merge([bAvg, gAvg, rAvg]).astype("uint8")

    return frame

def boilerplate_filter(frame, filter_params):
    '''
    This function is a boilerplate for the filters to be added that only need a single frame.

    Parameters:
        frame (numpy.ndarray): Frame to be processed.
        filter_params (dict): Dictionary containing the parameters for the filter.

    Returns:
        frame (numpy.ndarray): Processed frame.
    '''
    
    # do the filtering here!

    # return the processed frame
    pass

def boilerplate_filter_temporal(frames_array, filter_params):
    '''
    This function is a boilerplate for the temporal filters to be added that need a array of frames.

    Parameters:
        frames_array (numpy.ndarray): List of the frames.
        filter_params (dict): Dictionary containing the parameters for the filter.

    Returns:
        frame (numpy.ndarray): Processed frame.
    '''
    
    # do the filtering here!

    # return the processed frame
    pass