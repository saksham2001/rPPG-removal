'''
This script provides functions to apply the filter(s) to videos.

Authors: Saksham Bhutani, Mohamed Elgendi and Carlo Menon
License: MIT
'''
import cv2
import time

def apply_filter(input_path, output_path, filter_func, filter_temporal, roi_func, filter_params, notebook_mode=False):
    '''
    This function processes a video with a single filter.

    Parameters:
        input_path (str): Path of the input video.
        output_path (str): Path of the output video.
        filter_func (function): Filter function to be applied.
        filter_temporal (string): Name of the temporal filter.
        roi_func (function): ROI function to be applied.
        filter_params (dict): Dictionary containing the parameters for the filter.
        notebook_mode (bool): Whether to run in notebook mode.

    Returns:
        None
    '''
    print('Starting video conversion...', end='\n\n')

    # read video
    cap = cv2.VideoCapture(input_path)

    # get the frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # define codec and create VideoWriter object
    out = cv2.VideoWriter(output_path, 0, input_fps, (frame_width, frame_height))

    total_frames_proc = 0

    if filter_temporal=='timeblur':
        rAvg = 0
        gAvg = 0
        bAvg = 0
    elif filter_temporal=='timeblur_sliding':
        frame_queue = []

    if notebook_mode:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    # create tqdm progress bar
    with tqdm(total=total_frames) as pbar:
        # read until end of video
        while cap.isOpened():
            # capture each frame of the video
            ret, frame = cap.read()

            if ret:
                total_frames_proc += 1

                # convert to rgb
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # generate roi mask
                roi_mask = roi_func(frame_rgb, frame_height, frame_width)

                # if temporal function add to array
                if filter_temporal=='timeblur':
                    filter_params = {'total': total_frames_proc,
                                        'rAvg': rAvg,
                                        'gAvg': gAvg,
                                        'bAvg': bAvg}

                    frame_copy, [rAvg, gAvg, bAvg] = filter_func(frame.copy(), filter_params)
                elif filter_temporal=='timeblur_sliding':
                    frame_queue.append(frame.copy())

                    if len(frame_queue) > filter_params['window_size']:
                        frame_queue.pop(0)

                    frame_copy = filter_func(frame_queue, filter_params)
                else:
                    # apply filter
                    frame_copy = filter_func(frame.copy(), filter_params)

                # extract the mask area from the edited frame
                face_edited = cv2.bitwise_and(frame_copy, frame_copy, mask=roi_mask)

                # extract the are not in roi from unedited frame
                mask_not = cv2.bitwise_not(roi_mask)
                frame_full = cv2.bitwise_and(frame, frame, mask=mask_not)

                # paste the filtered face region back on the original frame
                final_frame = cv2.add(face_edited, frame_full)

                # write the frame to the output file
                out.write(final_frame)
                
                # update progress bar
                pbar.update(1)
            else:
                break
    # release Video Capture
    cap.release()

    # release Video 
    out.release()
    # close all frames and video windows
    cv2.destroyAllWindows()

    print('Conversion completed successfully!')
    print('Video saved at: {}'.format(output_path))


def apply_filter_live(filter_func, filter_temporal, roi_func, filter_params, metric, display, frames_to_process=None, notebook_mode=False):
    '''
    This function processes a video with a single filter.

    Parameters:
        filter_func (function): Filter function to be applied.
        filter_temporal (string): Name of the temporal filter.
        roi_func (function): ROI function to be applied.
        filter_params (dict): Dictionary containing the parameters for the filter.
        metric (function): Metric function to be applied. [If None, metric is not calculated.]
        display (bool): Whether to display the video.
        frames_to_process (int): Number of frames to process. [If None, frames are processed until program is stopped.]
        notebook_mode (bool): Whether to run in notebook mode.

    Returns:
        metric_lst (list): List of metric values. [Only returned if metric is not None]
    '''
    print('Starting live conversion...', end='\n\n')

    # read video
    cap = cv2.VideoCapture(0)

    # get the frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    total_frames_proc = 0

    if filter_temporal=='timeblur':
        rAvg = 0
        gAvg = 0
        bAvg = 0
    elif filter_temporal=='timeblur_sliding':
        frame_queue = []

    metric_lst = []

    prev_time = time.time()
    new_time = time.time()

    if frames_to_process is not None:
        if notebook_mode:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        pbar = tqdm(total=frames_to_process)


    # read until end of video
    while cap.isOpened():
        # capture each frame of the video
        ret, frame = cap.read()

        if ret:
            total_frames_proc += 1

            # convert to rgb
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # generate roi mask
            roi_mask = roi_func(frame_rgb, frame_height, frame_width)

            # if temporal function add to array
            if filter_temporal=='timeblur':
                filter_params = {'total': total_frames_proc,
                                    'rAvg': rAvg,
                                    'gAvg': gAvg,
                                    'bAvg': bAvg}

                frame_copy, [rAvg, gAvg, bAvg] = filter_func(frame, filter_params)
            elif filter_temporal=='timeblur_sliding':
                frame_queue.append(frame)

                if len(frame_queue) > filter_params['window_size']:
                    frame_queue.pop(0)

                frame_copy = filter_func(frame_queue, filter_params)
            else:
                # apply filter
                frame_copy = filter_func(frame, filter_params)

            # extract the mask area from the edited frame
            face_edited = cv2.bitwise_and(frame_copy, frame_copy, mask=roi_mask)

            # extract the are not in roi from unedited frame
            mask_not = cv2.bitwise_not(roi_mask)
            frame_full = cv2.bitwise_and(frame, frame, mask=mask_not)

            # paste the filtered face region back on the original frame
            final_frame = cv2.add(face_edited, frame_full)

            if metric is not None:
                new_time = time.time()
                metric_val = metric(prev_time, new_time)
                prev_time = new_time

                metric_lst.append(metric_val)

                try:
                    avg = sum(metric_lst)/len(metric_lst)
                except:
                    avg = 0

            if display:
                if metric is not None:
                    # show fps on the frame and frame count
                    cv2.putText(final_frame, "Metric Current: {:.2f} | Avg: {:.2f}".format(metric_val, sum(metric_lst)/len(metric_lst)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # display the frame
                cv2.imshow('frame', final_frame)
        else:
            break

        if frames_to_process is not None:
            pbar.update(1)
            pbar.set_description('Metric Current: {:.2f} | Avg: {:.2f}'.format(metric_val, avg), refresh=True)
            if total_frames_proc >= frames_to_process:
                break

        # if 'q' is pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release Video Capture
    cap.release()

    # close all frames and video windows
    cv2.destroyAllWindows()

    if metric is not None:
        return metric_lst