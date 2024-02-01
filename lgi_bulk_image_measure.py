'''
This script provides code to measure the loss of information (between frames) in bulk for a subject in the LGI-PPGI dataset for all the activities and all the filters.

Authors: Saksham Bhutani, Mohamed Elgendi and Carlo Menon
License: MIT
'''

from pyRemoval.utils.image import compute_errors
from pyRemoval.metrics.infoloss import mse
from pyRemoval.utils.writer import save2excel

activities = ['resting', 'talking', 'rotation', 'gym']

meth = ['normal', 'medianblur', 'gaussianblur', 'bilateralblur', 'gausiannoise', 'saltpeppernoise', 'poissonnoise', 
        'specklenoise', 'localvarnoise', 'peppernoise', 'timebluring', 'timeblurwindow']

if __name__ == "__main__":
    for k in range(len(activities)):
        mse_lst = []
        
        for i in range(len(meth)):
            print(f'Activity: {activities[k]}, Method: {meth[i]}')
            if meth[i] == 'normal':
                pathNormal = '/Volumes/Seagate/College/ETH/Work/Data/orignal/LGI-PPG Dataset/angelo/angelo_'+activities[k]+ '/cv_camera_sensor_stream_handler.avi'
                pathMod = '/Volumes/Seagate/College/ETH/Work/Data/orignal/LGI-PPG Dataset/angelo/angelo_'+activities[k]+ '/cv_camera_sensor_stream_handler.avi'
            else:
                pathNormal = '/Volumes/Seagate/College/ETH/Work/Data/orignal/LGI-PPG Dataset/angelo/angelo_'+activities[k]+ '/cv_camera_sensor_stream_handler.avi'
                pathMod = '/Volumes/Seagate/College/ETH/Work/Data/meta/LGI-PPG/angelo/angelo_'+activities[k]+ '/' + meth[i] + '.avi'

            mse_val= compute_errors(pathNormal, pathMod, mse)

            mse_lst.append(mse_val)
            
            print(f'Mean: {mse_val}')
            print()
        save2excel(meth, mse_lst, '/Users/sakshambhutani/Desktop/College/ETH/Work/Experimental/AntiPPG/Results/Datasets/LGI-PPG/angelo/angelo_'+activities[k]+ '/')
        print()