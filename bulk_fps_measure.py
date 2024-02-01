'''
This script provides code to measure the speed of the filters. This script will use the webcam to measure the speed of the filters.
Authors: Saksham Bhutani, Mohamed Elgendi and Carlo Menon
License: MIT
'''

from pyRemoval.processing.filters import *
from pyRemoval.processing.extract import facial_roi
from pyRemoval.processing.converter import apply_filter_live
from pyRemoval.metrics.speed import fps

activities = ['resting', 'talking', 'rotation', 'gym']

methods = {'medianblur': medianblur, 'gaussianblur': gaussianblur, 'bilateralblur': bilateralblur, 
           'gausiannoise': gaussiannoise, 'saltpeppernoise': saltpeppernoise, 'poissonnoise': poissonnoise, 
           'specklenoise': specklenoise, 'localvarnoise': localvarnoise, 'peppernoise': peppernoise,
           'timebluring': timeblur, 'timeblurwindow': timeblur_sliding}

parameters = {'medianblur': {'kernel_size': 5}, 'gaussianblur': {'kernel_size': 5}, 'bilateralblur': {'kernel_size': 5},
                'gausiannoise': {'mean': 0.1, 'sigma': 0.01}, 'saltpeppernoise': {'amount': 0.05}, 'poissonnoise': {}, 'specklenoise': {}, 'localvarnoise': {}, 
                'peppernoise': {'amount': 0.05}, 'timebluring': {}, 'timeblurwindow': {'window_size': 5}}

frames_to_process = 100
metric = fps
roi_func = facial_roi
display = False

if __name__ == '__main__':
    for meth in list(methods.keys()):
        print(f'Applying {meth} filter...')
        filter_func = methods[meth]
        filter_params = parameters[meth]
        
        if meth == 'timebluring':
            filter_temporal = 'timeblur'
        elif meth == 'timeblurwindow':
            filter_temporal = 'timeblur_sliding'
        else:
            filter_temporal = False

        fps_list = apply_filter_live(filter_func, filter_temporal, roi_func, filter_params, fps, display, frames_to_process)

        print('\nAverage FPS for {}: {:.2f}'.format(meth, sum(fps_list)/len(fps_list)))
