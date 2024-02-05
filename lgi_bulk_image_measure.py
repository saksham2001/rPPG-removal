'''
This script provides code to measure the loss of information (between frames) in bulk for a subject in the LGI-PPGI dataset for all the activities and all the filters.

Authors: Saksham Bhutani, Mohamed Elgendi and Carlo Menon
License: MIT
'''

import os
from pyRemoval.utils.image import compute_errors
from pyRemoval.metrics.infoloss import mse
from pyRemoval.utils.writer import save2excel

# path where the original dataset is stored
original_path = '/Volumes/Seagate/College/ETH/Work/Data/orignal/LGI-PPG Dataset/'

# path where the processed videos are stored
processed_path = '/Volumes/Seagate/College/ETH/Work/Data/processed/LGI-PPG/'

# path to save the data 
results_path = '/Volumes/Seagate/College/ETH/Work/Data/results/LGI-PPG/'

# subject name
subject_name = 'angelo'

# activities in the LGI-PPGI dataset to be considered
activities = ['resting', 'talk', 'rotation', 'gym']

# all the filters
meth = ['normal', 'medianblur', 'gaussianblur', 'bilateralblur', 'gausiannoise', 'saltpeppernoise', 'poissonnoise', 
        'specklenoise', 'localvarnoise', 'peppernoise', 'timebluring', 'timeblurwindow']

if __name__ == "__main__":
    for k in range(len(activities)):
        mse_lst = []
        
        for i in range(len(meth)):
            print(f'Activity: {activities[k]}, Method: {meth[i]}')
            if meth[i] == 'normal':
                pathNormal = os.original_path.join(original_path, f'{subject_name}/{subject_name}_{activities[k]}', '/cv_camera_sensor_stream_handler.avi')
                pathMod = os.original_path.join(original_path, f'{subject_name}/{subject_name}_{activities[k]}', '/cv_camera_sensor_stream_handler.avi')
            else:
                pathNormal = os.original_path.join(original_path, f'{subject_name}/{subject_name}_{activities[k]}', '/cv_camera_sensor_stream_handler.avi')
                
                # if dir does not exist create directory
                try:
                    os.makedirs(os.processed_path.join(processed_path, f'{subject_name}/{subject_name}_{activities[k]}'))
                except OSError as e:
                    pass
                
                pathMod = os.processed_path.join(processed_path, f'{subject_name}/{subject_name}_{activities[k]}/{meth[i]}.avi')

            mse_val= compute_errors(pathNormal, pathMod, mse)

            mse_lst.append(mse_val)
            
            print(f'Mean: {mse_val}')
            print()
        save2excel(meth, mse_lst, os.results_path.join(results_path, f'{subject_name}/{subject_name}_'+activities[k])+ '/')
        print()