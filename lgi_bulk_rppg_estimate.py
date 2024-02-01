'''
This script provides the code to estimate the effectivness of the filters (mean bpm error, etc) for a subject in the LGI-PPGI dataset for all the activities and all the filters.

Authors: Saksham Bhutani, Mohamed Elgendi and Carlo Menon
License: MIT
'''

import pyVHR as vhr
import numpy as np
from pyVHR.BVP import *
from pyVHR.utils.errors import bpm_diff
from pyVHR.plot.visualize import VisualizeParams
from scipy.signal import welch
from pyVHR.utils.errors import printErrors, RMSEerror, MAEerror, MAXError, PearsonCorr, LinCorr
from scipy.interpolate import interp1d
from pyVHR.utils.errors import printErrors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import pandas as pd
import os

# Plotting: set 'colab' for Google Colaboratory, 'notebook' otherwise
vhr.plot.VisualizeParams.renderer = 'colab'  # or 'notebook'

meth = ["medianblur", "gaussianblur", "bilateralblur", "gausiannoise", "saltpeppernoise", "poissonnoise", "specklenoise",
        "localvarnoise", "peppernoise", "timebluring", "timeblurwindow"]

techniques = ["CHROM", "POS", "LGI", "GREEN", "ICA"]

activities = ['resting', 'talking', 'rotation', 'gym'] 

# -- LOAD A DATASET

dataset_name = 'lgi_ppgi'          # the name of the python class handling it 
video_DIR = '/Volumes/Seagate/College/ETH/Work/Data/orignal/LGI-PPG Dataset/alex/'  # dir containing videos
BVP_DIR = '/Volumes/Seagate/College/ETH/Work/Data/orignal/LGI-PPG Dataset/alex/'    # dir containing BVPs GT

dataset = vhr.datasets.datasetFactory(dataset_name, videodataDIR=video_DIR, BVPdataDIR=BVP_DIR)
allvideo = dataset.videoFilenames

# print the list of video names with the progressive index (idx)
for v in range(len(allvideo)):
  print(v, allvideo[v])

def split_array(array, wsize, stride, fps):
    ''' Splits an array in n windows'''
    N = len(array)
    wsize_fr = wsize*fps
    stride_fr = stride*fps
    idx = []
    timesES = []
    num_win = int((N-wsize_fr)/stride_fr)+1
    s = 0
    for i in range(num_win):
        idx.append(array[s:s+wsize_fr]) # check
        s += stride_fr
        timesES.append(wsize/2+stride*i)
    return np.array(idx), np.array(timesES, dtype=np.float32)

def reduce_sampling(data, new_points=200):
    # Calculate the factor of reduction
    reduction_factor = len(data) // new_points
    # Downsample the data by taking only one sample for every 'reduction_factor' samples
    downsampled_data = data[::reduction_factor]
    return downsampled_data

def reduce_sampling_interp(data, new_points=200):
    # Generate the original time samples
    t = np.linspace(0, 1, len(data))
    # Generate the new time samples
    new_t = np.linspace(0, 1, new_points)
    # Interpolate the data at the new time samples
    f = interp1d(t, data, kind='linear')
    interpolated_data = f(new_t)
    return interpolated_data

def plotWindow(timeseries):
    plt.plot(timeseries)
    plt.show()

def pipe(videoFileName, name, tech, activity_num):
    wsize = 8 # seconds of video processed (with overlapping) for each estimate 
    video_idx = activity_num      # index of the video to be processed (0-Gym, 1-Resting, 2-Rotation, 3-Talk)
    fname = dataset.getSigFilename(video_idx)
    sigGT = dataset.readSigfile(fname)
    bvpGT = sigGT.data
    bpmGT, timesGT = sigGT.getBPM(wsize)

    print('Video processed name: ', videoFileName)
    fps = vhr.extraction.get_fps(videoFileName)
    print('Video frame rate:     ',fps)
    
    print(name.capitalize())
    
    sig_extractor = vhr.extraction.SignalProcessing()
    
    sig_extractor.set_skin_extractor(vhr.extraction.SkinExtractionConvexHull())
    
    seconds = 0
    sig_extractor.set_total_frames(seconds*fps)
    
    vhr.extraction.SkinProcessingParams.RGB_LOW_TH = 2
    vhr.extraction.SkinProcessingParams.RGB_HIGH_TH = 254

    vhr.extraction.SignalProcessingParams.RGB_LOW_TH = 2
    vhr.extraction.SignalProcessingParams.RGB_HIGH_TH = 254
    
    landmarks = vhr.extraction.MagicLandmarks.cheek_left_top +\
                   vhr.extraction.MagicLandmarks.forehead_center +\
                   vhr.extraction.MagicLandmarks.forehoead_right +\
                   vhr.extraction.MagicLandmarks.cheek_right_top +\
                   vhr.extraction.MagicLandmarks.forehead_left +\
                   vhr.extraction.MagicLandmarks.nose 

    # ... or sample the face by 100 equispaced landmarks
    landmarks = [2, 3, 4, 5, 6, 8, 9, 10, 18, 21, 32, 35, 36, 43, 46, 47, 48, 50, 54, \
                 58, 67, 68, 69, 71, 92, 93, 101, 103, 104, 108, 109, 116, 117, \
                 118, 123, 132, 134, 135, 138, 139, 142, 148, 149, 150, 151, 152, 182, 187, 188, 193, 197, 201, 205, 206, 207, \
                 210, 211, 212, 216, 234, 248, 251, 262, 265, 266, 273, 277, 278, 280, \
                 284, 288, 297, 299, 322, 323, 330, 332, 333, 337, 338, 345, \
                 346, 361, 363, 364, 367, 368, 371, 377, 379, 411, 412, 417, 421, 425, 426, 427, 430, 432, 436]
    
    print('Num landmarks: ', len(landmarks))
    
    sig_extractor.set_landmarks(landmarks)
    
    hol_sig = sig_extractor.extract_holistic(videoFileName)
    
    # visualize_skin_coll = sig_extractor.get_visualize_skin()
    
    windowed_hol_sig, timesES = vhr.extraction.sig_windowing(hol_sig, wsize, 1, fps)
    
    try:
        print('Num channels and window length: ', windowed_hol_sig[0].shape)
    except:
        pass
    
    filtered_windowed_hol_sig = vhr.BVP.apply_filter(windowed_hol_sig, vhr.BVP.rgb_filter_th, params={'RGB_LOW_TH': 0, 'RGB_HIGH_TH': 255})
    
    filtered_windowed_hol_sig = vhr.BVP.apply_filter(filtered_windowed_hol_sig, vhr.BVP.BPfilter, params={'order':6,'minHz':0.65,'maxHz':4.0,'fps':fps})

    print('Win size: (#signals, #channels, #frames) = ', filtered_windowed_hol_sig[0].shape)

    if tech == "CHROM":
        print("Technique: CHROM")
        hol_bvps_CHROM = RGB_sig_to_BVP(windowed_hol_sig, fps, device_type='cpu', method=cpu_CHROM)
        print('Number of windows: ', len(hol_bvps_CHROM))
        print('Number of estimators and number of number of frames in a windows: ', hol_bvps_CHROM[0].shape)
        hol_bvps = vhr.BVP.apply_filter(hol_bvps_CHROM, BPfilter, params={'order':6,'minHz':0.65,'maxHz':4.0,'fps':fps})
    elif tech == "POS":
        print("Technique: POS")
        hol_bvps_POS = RGB_sig_to_BVP(windowed_hol_sig, fps, device_type='cpu', method=cpu_POS, params={'fps':fps})
        print('Number of windows: ', len(hol_bvps_POS))
        print('Number of estimators and number of number of frames in a windows: ', hol_bvps_POS[0].shape)
        hol_bvps = vhr.BVP.apply_filter(hol_bvps_POS, BPfilter, params={'order':6,'minHz':0.65,'maxHz':4.0,'fps':fps})
    elif tech == "LGI":
        print("Technique: LGI")
        hol_bvps_LGI = RGB_sig_to_BVP(windowed_hol_sig, fps, device_type='cpu', method=cpu_LGI)
        print('Number of windows: ', len(hol_bvps_LGI))
        print('Number of estimators and number of number of frames in a windows: ', hol_bvps_LGI[0].shape)
        hol_bvps = vhr.BVP.apply_filter(hol_bvps_LGI, BPfilter, params={'order':6,'minHz':0.65,'maxHz':4.0,'fps':fps})
    elif tech == "GREEN":
        print("Technique: GREEN")
        hol_bvps_GREEN = RGB_sig_to_BVP(windowed_hol_sig, fps, device_type='cpu', method=cpu_GREEN)
        print('Number of windows: ', len(hol_bvps_GREEN))
        print('Number of estimators and number of number of frames in a windows: ', hol_bvps_GREEN[0].shape)
        hol_bvps = vhr.BVP.apply_filter(hol_bvps_GREEN, BPfilter, params={'order':6,'minHz':0.65,'maxHz':4.0,'fps':fps})
    elif tech == "ICA":
        print("Technique: ICA")
        try:
            hol_bvps_ICA = RGB_sig_to_BVP(windowed_hol_sig, fps, device_type='cpu', method=cpu_ICA, params={'component':'second_comp'})
        except:
            return {'RMSE': None}
        print('Number of windows: ', len(hol_bvps_ICA))
        print('Number of estimators and number of number of frames in a windows: ', hol_bvps_ICA[0].shape)
        hol_bvps = vhr.BVP.apply_filter(hol_bvps_ICA, BPfilter, params={'order':6,'minHz':0.65,'maxHz':4.0,'fps':fps})
    
    print()
    
    hol_bpmES = vhr.BPM.BVP_to_BPM(hol_bvps, fps)       # CPU version
    
    hol_bpmES = np.expand_dims(hol_bpmES, axis=0)
    # hol_bpmES_ICA = np.mean(hol_bpmES_ICA, axis=2)
    
    # print(hol_bpmES.shape)

    # RMSE, MAE, MAX, PCC, CCC, SNR = getErrors(hol_bvps, fps, hol_bpmES_ICA, bpmGT, timesES, timesGT)
    
    RMSE = RMSEerror(hol_bpmES, bpmGT, timesES, timesGT)
    MAE = MAEerror(hol_bpmES, bpmGT, timesES, timesGT)
    MAX = MAXError(hol_bpmES, bpmGT, timesES, timesGT)
    PCC = PearsonCorr(hol_bpmES, bpmGT, timesES, timesGT)
    CCC = LinCorr(hol_bpmES, bpmGT, timesES, timesGT)
    SNR = 0
    
    print("\n")
    
    w = np.random.randint(0, len(windowed_hol_sig))
    
    res = {'video': videoFileName, 'fps': fps, 'window_holl': windowed_hol_sig[w], 'filtered_window_holl': filtered_windowed_hol_sig[w],
           'hol_bpmES': hol_bpmES, 'hol_bvps': hol_bvps, 'timesES': timesES, 'RMSE': RMSE, 'MAE': MAE, 'MAX': MAX, 'PCC': PCC, 'CCC': CCC, 'SNR': SNR, 'w': w}
    
    del hol_bvps, hol_bpmES, timesES, hol_sig, sig_extractor, windowed_hol_sig, filtered_windowed_hol_sig
    
    return res

def vis_vid(results):
    for i in range(len(meth)):
        print(meth[i].capitalize())
        vhr.plot.display_video(results[i]['video'])
    
def vis_skin_holl(results):
    for i in range(len(meth)):
        print(meth[i].capitalize())
        print('Number of frames processed: ',len(results[i]['skin_coll']))
        vhr.plot.interactive_image_plot(results[i]['skin_coll'],1.0)

def visualize_windowed_sig(results):
    for i in range(len(meth)):
        w = results[i]['w']
        print(meth[i].capitalize())
        vhr.plot.visualize_windowed_sig(results[i]['window_holl'], w)

def visualize_filtered_window(results):
    for i in range(len(meth)):
        w = results[i]['w']
        print(meth[i].capitalize())
        vhr.plot.visualize_windowed_sig(results[i]['filtered_window_holl'], w)

def visualize_window_BVP(results):
    fig = go.Figure()

    for i in range(len(meth)):
        bvp = results[i]['hol_bvps']
        w = results[i]['w']

        for e in bvp:
            name = "BVP_" + str(i)
            fig.add_trace(go.Scatter(x=np.arange(bvp.shape[1]), y=e[:],
                                     mode='lines', name=meth[i].capitalize()))

    fig.update_layout(title="BVP #" + str(w))
    fig.show(renderer=VisualizeParams.renderer)


def visualize_PSD(results):
    fig = go.Figure()
    for i in range(len(meth)):

        fps = results[i]['fps']
        data = results[i]['hol_bvps']
        w = results[i]['w']

        minHz=0.65
        maxHz=4.0

        _, n = data.shape
        if data.shape[0] == 0:
            return np.float32(0.0)
        if n < 256:
            seglength = n
            overlap = int(0.8*n)  # fixed overlapping
        else:
            seglength = 256
            overlap = 200
        # -- periodogram by Welch
        F, P = welch(data, nperseg=seglength,
                     noverlap=overlap, fs=fps, nfft=2048)
        F = F.astype(np.float32)
        P = P.astype(np.float32)

        # -- freq subband (0.65 Hz - 4.0 Hz)
        band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
        Pfreqs = 60*F[band]
        Power = P[:, band]

        # -- BPM estimate by PSD
        Pmax = np.argmax(Power, axis=1)  # power max

        # plot
        for idx in range(P.shape[0]):
            fig.add_trace(go.Scatter(
                x=F*60, y=P[idx], name=f"{meth[i].capitalize()} PSD_"+str(idx)+" no band"))
            fig.add_trace(go.Scatter(
                x=Pfreqs, y=Power[idx], name=f"{meth[i].capitalize()} PSD_"+str(idx)+" band"))

    fig.update_layout(title="PSD #" + str(w), xaxis_title='Beats per minute [BPM]')
    fig.show(renderer=VisualizeParams.renderer)

def plot_errors(results, file_name, plot=False):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    rmse = []
    mae = []
    maxx = []
    pcc = []
    ccc = []
    snr = []

    for j in range(len(techniques)):
        rmse_meth = []
        mae_meth = []
        maxx_meth = []
        pcc_meth = []
        ccc_meth = []
        snr_meth = []
        for i in range(len(meth)):
        
            print(techniques[j], meth[i])
            if results[j][i]['RMSE'] == None:
                rmse_meth.append(0)
                mae_meth.append(0)
                maxx_meth.append(0)
                pcc_meth.append(0)
                ccc_meth.append(0)
                snr_meth.append(0)
            else:
                printErrors(results[j][i]['RMSE'], results[j][i]['MAE'], results[j][i]['MAX'], results[j][i]['PCC'], results[j][i]['CCC'], results[j][i]['SNR'])
                print()
                
                rmse_meth.append(results[j][i]['RMSE'].tolist()[0])
                mae_meth.append(results[j][i]['MAE'].tolist()[0])
                maxx_meth.append(results[j][i]['MAX'].tolist()[0])
                pcc_meth.append(results[j][i]['PCC'].tolist()[0])
                ccc_meth.append(results[j][i]['CCC'].tolist()[0])
                snr_meth.append(results[j][i]['SNR'])
                
        rmse.append(rmse_meth)
        mae.append(mae_meth)
        maxx.append(maxx_meth)
        pcc.append(pcc_meth)
        ccc.append(ccc_meth)
        snr.append(snr_meth)

    if plot:
        fig = go.Figure(go.Bar(
                x=rmse,
                y=meth,
                orientation='h'))
        fig.update_layout(title="RMSE")
        fig.show()

        fig = go.Figure(go.Bar(
                x=mae,
                y=meth,
                orientation='h'))
        fig.update_layout(title="MAE")
        fig.show()

        fig = go.Figure(go.Bar(
                x=maxx,
                y=meth,
                orientation='h'))
        fig.update_layout(title="MAX Error")
        fig.show()

        fig = go.Figure(go.Bar(
                x=pcc,
                y=meth,
                orientation='h'))
        fig.update_layout(title="PCC")
        fig.show()

        fig = go.Figure(go.Bar(
                x=ccc,
                y=meth,
                orientation='h'))
        fig.update_layout(title="CCC")
        fig.show()

        fig = go.Figure(go.Bar(
                x=snr,
                y=meth,
                orientation='h'))
        fig.update_layout(title="SNR")
        fig.show()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for i in range(len(meth)):
            bpmES = results[j][i]['hol_bpmES']
            timesES = results[j][i]['timesES']

            if type(bpmES) == list:
                bpmES = np.expand_dims(bpmES, axis=0)
            if type(bpmES) == np.ndarray:
                if len(bpmES.shape) == 1:
                    bpmES = np.expand_dims(bpmES, axis=0)

            diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
            n, m = diff.shape  # n = num channels, m = bpm length
            df = np.abs(diff)
            dfMean = np.around(np.mean(df, axis=1), 1)

            # -- plot errors

            name = f'{techniques[j]} {meth[i]}: Ch {i} (µ = ' + str(dfMean[0]) + ' )'
            fig.add_trace(go.Scatter(
                x=timesES, y=df[0, :], name=name, mode='lines+markers'))

        fig.update_layout(xaxis_title='Times (sec)',
                          yaxis_title='MAE', showlegend=True)

        fig.show(renderer=VisualizeParams.renderer)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        GTmean = np.around(np.mean(bpmGT), 1)
        name = 'GT (µ = ' + str(GTmean) + ' )'
        fig.add_trace(go.Scatter(x=timesGT, y=bpmGT,
                                     name=name, mode='lines+markers'))
        for i in range(len(meth)):
            bpmES = results[j][i]['hol_bpmES']
            # print(type(bpmES[0]))
            timesES = results[j][i]['timesES']

            ESmean = np.around(np.mean(bpmES[0, :]), 1)
            name = f'{meth[i].capitalize()} (µ = ' + str(ESmean) + ' )'

            fig.add_trace(go.Scatter(
                x=timesES, y=bpmES[0, :], name=name, mode='lines+markers'))

        fig.update_layout(xaxis_title='Times (sec)',
                              yaxis_title='BPM', showlegend=True)

        fig.show(renderer=VisualizeParams.renderer)
            
    df_rmse = pd.DataFrame(rmse, index=techniques, columns=meth)
    df_mae = pd.DataFrame(mae, index=techniques, columns=meth)
    df_maxx = pd.DataFrame(maxx, index=techniques, columns=meth)
    df_pcc = pd.DataFrame(pcc, index=techniques, columns=meth)
    df_ccc = pd.DataFrame(ccc, index=techniques, columns=meth)
    df_snr = pd.DataFrame(snr, index=techniques, columns=meth)
    
    file_name = os.path.join('/Users/sakshambhutani/Desktop/College/ETH/Work/Experimental/AntiPPG/Results/Datasets/', file_name)
    
    with pd.ExcelWriter(file_name) as writer: 
        df_rmse.to_excel(writer, sheet_name='RMSE', float_format="%.2f")
        df_mae.to_excel(writer, sheet_name='MAE', float_format="%.2f")
        df_maxx.to_excel(writer, sheet_name='MAX', float_format="%.2f")
        df_pcc.to_excel(writer, sheet_name='PCC', float_format="%.2f")
        df_ccc.to_excel(writer, sheet_name='CCC', float_format="%.2f")


if __name__ == '__main__':
    for k in range(len(activities)):
        print('Activity: ', activities[k])
        results = []
        for j in techniques:
            res_meth = []
            for i in range(len(meth)):
                if meth[i] == 'normal':
                    path = '/Volumes/Seagate/College/ETH/Work/Data/orignal/LGI-PPG Dataset/alex/alex_'+activities[k]+ '/cv_camera_sensor_stream_handler.avi'
                else:
                    path = '/Volumes/Seagate/College/ETH/Work/Data/meta/LGI-PPG/alex/alex_'+activities[k]+ '/' + meth[i] + '.avi'

                res = pipe(path, meth[i], j, k)
                res_meth.append(res)

            results.append(res_meth)

        plot_errors(results, '/Users/sakshambhutani/Desktop/College/ETH/Work/Experimental/AntiPPG/Results/Datasets/LGI-PPG/alex/alex_'+activities[k]+ '/'+ 'results.xlsx')

