'''
This script provides code to bulk process videos for a subject in the LGI-PPGI dataset for all the activities applying all the filters.

Authors: Saksham Bhutani, Mohamed Elgendi and Carlo Menon
License: MIT
'''

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
import mediapipe as mp
import re
from pyRemoval.processing.converter import apply_filter
from pyRemoval.processing.filters import *
from pyRemoval.processing.extract import facial_roi

methods = {'medianblur': medianblur, 'gaussianblur': gaussianblur, 'bilateralblur': bilateralblur, 
           'gausiannoise': gaussiannoise, 'saltpeppernoise': saltpeppernoise, 'poissonnoise': poissonnoise, 
           'specklenoise': specklenoise, 'localvarnoise': localvarnoise, 'peppernoise': peppernoise,
           'timebluring': timeblur, 'timeblurwindow': timeblur_sliding}

parameters = {'medianblur': {'kernel_size': 5}, 'gaussianblur': {'kernel_size': 5}, 'bilateralblur': {'kernel_size': 5},
                'gausiannoise': {'mean': 0.1, 'sigma': 0.01}, 'saltpeppernoise': {'amount': 0.05}, 'poissonnoise': {}, 'specklenoise': {}, 'localvarnoise': {}, 
                'peppernoise': {'amount': 0.05}, 'timebluring': {}, 'timeblurwindow': {'window_size': 5}}

lgi_activity = ['resting', 'talking', 'rotation', 'gym']

if __name__ == '__main__':

    for act in lgi_activity:
        print("Processing: " + act, end='\n')
        for meth in list(methods.keys()):
            output_location = f'/Volumes/Seagate/College/ETH/Work/Data/meta/LGI-PPG/alex/alex_{act}/facial'

            # if dir does not exist create directory
            try:
                os.makedirs(output_location)
            except OSError as e:
                pass

            # find avi file in folder
            path = [pt for pt in os.listdir(f'/Volumes/Seagate/College/ETH/Work/Data/orignal/LGI-PPG Dataset/alex/alex_{act}') if pt.endswith('.avi')]

            input_path = os.path.join(f'/Volumes/Seagate/College/ETH/Work/Data/orignal/LGI-PPG Dataset/alex/alex_{act}', path[0])
            output_path = os.path.join(output_location, f'{meth}.avi')

            filter_func = methods[meth]
            print(f'Applying {meth} filter...')

            if meth == 'timebluring':
                filter_temporal = 'timeblur'
            elif meth == 'timeblurwindow':
                filter_temporal = 'timeblur_sliding'
            else:
                filter_temporal = False

            roi_func = facial_roi

            filter_params = parameters[meth]

            apply_filter(input_path, output_path, filter_func, filter_temporal, roi_func, filter_params)

            print(f'Filter {meth} applied successfully!', end='\n\n')

        print(f'All filters applied successfully for {act}!', end='\n\n')
