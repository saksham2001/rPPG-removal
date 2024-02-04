'''
This script provides code to bulk process videos for a subject in the LGI-PPGI dataset for all the activities applying all the filters.

Authors: Saksham Bhutani, Mohamed Elgendi and Carlo Menon
License: MIT
'''

import os
from pyRemoval.processing.converter import apply_filter
from pyRemoval.processing.filters import *
from pyRemoval.processing.extract import facial_roi

# path to the LGI-PPGI dataset
input_location = '/Volumes/Seagate/College/ETH/Work/Data/original/LGI-PPG/'

# path to save the processed videos
output_location = '/Volumes/Seagate/College/ETH/Work/Data/processed/LGI-PPG/'

# subject name
subject_name = 'alex'

# all the filters to process the video
methods = {'medianblur': medianblur, 'gaussianblur': gaussianblur, 'bilateralblur': bilateralblur, 
           'gausiannoise': gaussiannoise, 'saltpeppernoise': saltpeppernoise, 'poissonnoise': poissonnoise, 
           'specklenoise': specklenoise, 'localvarnoise': localvarnoise, 'peppernoise': peppernoise,
           'timebluring': timeblur, 'timeblurwindow': timeblur_sliding}

# parameters for the filters
parameters = {'medianblur': {'kernel_size': 5}, 'gaussianblur': {'kernel_size': 5}, 'bilateralblur': {'kernel_size': 5},
                'gausiannoise': {'mean': 0.1, 'sigma': 0.01}, 'saltpeppernoise': {'amount': 0.05}, 'poissonnoise': {}, 'specklenoise': {}, 'localvarnoise': {}, 
                'peppernoise': {'amount': 0.05}, 'timebluring': {}, 'timeblurwindow': {'window_size': 5}}

# activities in the LGI-PPGI dataset to be considered
lgi_activity = ['resting', 'talking', 'rotation', 'gym']

# region of interest
roi_func = facial_roi

if __name__ == '__main__':

    for act in lgi_activity:
        print("Processing: " + act, end='\n')
        for meth in list(methods.keys()):

            # if dir does not exist create directory
            try:
                os.makedirs(output_location)
            except OSError as e:
                pass

            # find avi file in folder
            path = [pt for pt in os.listdir(os.input_location.join(input_location, f'{subject_name}/{subject_name}_{act}')) if pt.endswith('.avi')]

            input_path = os.path.join(input_location, f'/{subject_name}/{subject_name}_{act}', path[0])

            # if dir does not exist create directory
            try:
                os.makedirs(os.output_location.join(output_location, f'{subject_name}/{subject_name}_{act}'))
            except OSError as e:
                pass
            
            output_path = os.path.join(output_location, f'{subject_name}/{subject_name}_{act}/{meth}.avi')

            filter_func = methods[meth]
            print(f'Applying {meth} filter...')

            if meth == 'timebluring':
                filter_temporal = 'timeblur'
            elif meth == 'timeblurwindow':
                filter_temporal = 'timeblur_sliding'
            else:
                filter_temporal = False

            filter_params = parameters[meth]

            apply_filter(input_path, output_path, filter_func, filter_temporal, roi_func, filter_params)

            print(f'Filter {meth} applied successfully!', end='\n\n')

        print(f'All filters applied successfully for {act}!', end='\n\n')
