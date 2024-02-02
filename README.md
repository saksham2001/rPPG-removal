# Preserving Privacy and Video Quality through Remote Physiological Signal Removal

## Abstract
The revolutionary remote photoplethysmography (rPPG) technique has enabled intelligent devices to estimate physiological parameters with remarkable accuracy. However, the continuous and surreptitious recording of individuals by these devices and the collecting of sensitive health data without users’ knowledge or consent raise serious privacy concerns. In response, this study explores frugal methods for modifying facial videos to conceal physiological signals while maintaining image quality. Eleven lightweight modification methods, including blurring operations, additive noises, and time-averaging techniques, were evaluated using five different rPPG techniques across four activities: rest, talking, head rotation, and gym. These rPPG methods require minimal computational resources, enabling real-time implementation on low-compute devices. Our results indicate that the time-averaging sliding frame method achieved the greatest balance between preserving the information within the frame and inducing a heart rate error, with an average error of 22 beats per minute (bpm). Further, the facial region of interest was found to be the most effective and to offer the best trade-off between bpm errors and information loss.

![pipeline](figures/modification_pipeline.png)

## Code Description
This repository contains code for the paper 'Preserving Privacy and Video Quality through Remote Physiological Signal Removal'. This repo contains two packages pyVHR and pyRemoval along with other usage demonstration code. The pyVHR package is the code from https://github.com/phuselab/pyVHR with minor changes to bulk process the videos. pyRemoval is the package developed for this paper that provides modules to process the videos, measure the loss of information and generate the results.

### pyRemoval package structure
The pyRemoval package is structured as follows:
```
pyRemoval
│
├─ processing
│   ├─ __init__.py
│   ├─ converter.py
│   │   ├─ apply_filter
│   │   └─ apply_filter_live
│   ├─ extract.py
│   │   ├─ full_roi
│   │   ├─ facial_roi
│   │   ├─ selected_facial_roi
│   │   └─ boilerplate_roi
│   └─ filters.py
│       ├─ medianblur
│       ├─ gaussianblur
│       ├─ bilateralblur
│       ├─ gaussiannoise
│       ├─ localvarnoise
│       ├─ saltpeppernoise
│       ├─ poissonnoise
│       ├─ peppernoise
│       ├─ specklenoise
│       ├─ timeblur
│       ├─ timeblur_sliding
│       └─ boilerplate_filter
├─ metrics
│   ├─ __init__.py
│   ├─ infoloss.py
│   │   ├─ mse
│   │   └─ boilerplate_metric
│   └─ speed.py
│       ├─ fps
│       └─ boilerplate_metric
└─ utils
    ├─ __init__.py
    ├─ image.py
    │   └─ compute_errors
    └─ writer.py
        └─ save2excel

```

## Setup
The code is tested in python 3.8+ on MacOS and Linux. Use the following commands to clone the repository and install the dependencies.
```Shell
git clone https://github.com/saksham2001/rPPG-removal
cd rPPG-removal/
pip install -r requirements.txt
```
**Note** Please ensure that you do not have the pyVHR python package installed. If you do, please uninstall it using `pip uninstall pyVHR` and use the modified version provided in this repository.

## Basic usage
Run the following code to process a video using one of the filters.
```python
from pyRemoval.processing.converter import apply_filter
from pyRemoval.processing.filters import peppernoise
from pyRemoval.processing.extract import facial_roi

# Define parameters
input_path = '/path/to/input/video.avi'
output_path = '/path/to/output/video.avi'
filter_func = peppernoise
filter_temporal = False
roi_func = facial_roi
filter_params = {'amount': 0.05}

# Apply filter
apply_filter(input_path, output_path, filter_func, filter_temporal, roi_func, filter_params)
```
For more detailed demostrations please refer to the [`pyRemoval Demo.ipynb`](https://github.com/saksham2001/rPPG-removal/blob/main/pyRemoval_Demo.ipynb) notebook

## Files
Files that can be used to process the videos from the LGI-PPGI dataset are:
* [`lgi_bulk_process.py`](https://github.com/saksham2001/rPPG-removal/blob/main/lgi_bulk_process.py): This file allows processing the LGI-PPGI dataset, automating the process of applying each filter for all the activities.
* [`lgi_bulk_image_measure.py`](https://github.com/saksham2001/rPPG-removal/blob/main/lgi_bulk_image_measure.py): This file allows measuring the loss of information in the images obtained after applying the filters for all the activities.
* [`lgi_bulk_rppg_estimate.py`](https://github.com/saksham2001/rPPG-removal/blob/main/lgi_bulk_rppg_estimate.py): This file allows estimating the rPPG signal from the images obtained after applying the filters for all the activities.
* [`bulk_fps_measure.py`](https://github.com/saksham2001/rPPG-removal/blob/main/bulk_fps_measure.py): This file allows measuring the FPS for applying all the filters.

**Note:** We have tried to make the code agnostic to the dataset. Hence, the code can be used to process any data which is available in the form of videos.

## Dataset
In this paper we use the LGI-PPGI dataset. The dataset can be downloaded from [https://github.com/partofthestars/LGI-PPGI-DB](https://github.com/partofthestars/LGI-PPGI-DB).

## Building upon the code
The code is designed to be modular and easy to build upon. The code can be extended to add new filters, test with different regions of interest, and test with new metrics to measure the loss of information. The code can also be extended to process other datasets. We have provided boilerplate code to add new filters, ROIs and metrics. 

* **Filters:** Create a function in [`pyRemoval/processing/filters.py`](https://github.com/saksham2001/rPPG-removal/blob/main/pyRemoval/processing/filters.py). The function should take in the image and the parameters as input and return the filtered image. If the filter has a temporal component (relies on multiple frames), then modifications to the [`apply_filter`](https://github.com/saksham2001/rPPG-removal/blob/0a089341738981ceaa76a0e8074f39a049d411fd/pyRemoval/processing/converter.py#L11) function in [`pyRemoval/processing/converter.py`](https://github.com/saksham2001/rPPG-removal/blob/main/pyRemoval/processing/converter.py#L11) might be required. For boilerplate code, refer to the [`boilerplate_filter`](https://github.com/saksham2001/rPPG-removal/blob/e7e4e0fff578a3d6072bda3198f6c1049d18ad89/pyRemoval/processing/filters.py#L287) function. After creating your own filter simply change the import statment in your file to test your filter.
* **ROI:** Create a function in [`pyRemoval/processing/extract.py`](https://github.com/saksham2001/rPPG-removal/blob/main/pyRemoval/processing/extract.py). The function should take in the frame and frame dimensions as input and return the region of interest as a mask. For boilerplate code, refer to the [`boilerplate_roi`](https://github.com/saksham2001/rPPG-removal/blob/e7e4e0fff578a3d6072bda3198f6c1049d18ad89/pyRemoval/processing/extract.py#L130). After creating your own ROI function simply change the import statment in your file to test your ROI.
* **Information Loss Metrics:** Create a function in [`pyRemoval/metrics/infoloss.py`](https://github.com/saksham2001/rPPG-removal/blob/main/pyRemoval/metrics/infoloss.py). The function should take in the original image and the filtered image as input and return the metric value. For boilerplate code, refer to the [`boilerplate_metric`](https://github.com/saksham2001/rPPG-removal/blob/e7e4e0fff578a3d6072bda3198f6c1049d18ad89/pyRemoval/metrics/infoloss.py#L27). After creating your own metric function simply change the import statment in your file to test your metric.
* **Speed Metrics:** Create a function in [`pyRemoval/metrics/speed.py`](https://github.com/saksham2001/rPPG-removal/blob/main/pyRemoval/metrics/speed.py). The function requires the previous time and the current time as inputs and return the metric value. For boilerplate code, refer to the [`boilerplate_metric`](https://github.com/saksham2001/rPPG-removal/blob/e7e4e0fff578a3d6072bda3198f6c1049d18ad89/pyRemoval/metrics/speed.py#L26). After creating your own metric function simply change the import statment in your file to test your metric.

<!-- ## Citation
If you use any of the data or resources provided on this page in any of your publications we ask you to cite the following work.
```add citation here``` -->
## Contact
If you have any questions, please feel free to contact us though email: Saksham Bhutani (sakshambhutani2001@gmail.com) or Mohamed Elgendi (moe.elgendi@hest.ethz.ch)

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/saksham2001/rPPG-removal/blob/main/LICENSE) file for details.

