'''
This script provides the code to extract different ROIs from the face.

Authors: Saksham Bhutani, Mohamed Elgendi and Carlo Menon
License: MIT
'''

import cv2
import mediapipe as mp
import numpy as np

forehead_lmk = [54, 68, 103, 104, 67, 69, 109, 108, 10, 151, 338, 337, 297, 299, 332, 333, 298, 63, 105, 66, 107, 9, 336, 296, 334, 293]
left_cheek_lmk = [116, 117, 118, 119, 120, 234, 100, 101, 142, 123, 137, 36, 93, 205, 147, 177, 187, 132, 207, 213, 215, 192, 58]
right_cheek_lmk = [349, 348, 347, 346, 345, 329, 447, 454, 330, 371, 266, 352, 323, 366, 425, 376, 401, 427, 361, 433, 435, 416, 288]


mouth_lmk = [61, 185, 40, 39, 37, 0, 267, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
left_eye_lmk = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173, 133]
right_eye_lmk = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398, 362]

mp_face_mesh = mp.solutions.face_mesh

def full_roi(frame, frame_height, frame_width):
    '''
    This function converts the entire frame into a mask to use the full frame as ROI.

    Parameters:
        frame (numpy.ndarray): Frame to be processed.
    
    Returns:
        mask (numpy.ndarray): Mask of the frame.
    '''

    # Create a mask of the same size as the frame
    return np.ones((frame_height, frame_width), np.uint8)

def facial_roi(frame, frame_height, frame_width):
    '''
    This function creates a mask for the facial region [face-(eyes+mouth)].

    Parameters:
        frame (numpy.ndarray): Frame to be processed.
        frame_height (int): Height of the frame.
        frame_width (int): Width of the frame.

    Returns:
        mask (numpy.ndarray): Mask of the frame.
    '''

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        
        # Process the frame to get the facial landmarks
        results = face_mesh.process(frame)

        # If face is detected
        if results.multi_face_landmarks:
            # Iterate over the detected faces
            for face_landmarks in results.multi_face_landmarks:
                
                # Get the list of facial landmarks
                lmk_lst = np.array([np.array([int(pt.x*frame_width), int(pt.y*frame_height)]) for pt in face_landmarks.landmark])

                # Create a Convex Hull with the facial landmarks
                hull = cv2.convexHull(lmk_lst)

                # Create the mask using the Convex Hull
                mask = np.zeros((frame_height, frame_width), np.uint8)
                mask = cv2.fillConvexPoly(mask, hull, 255)

                # Create a Convex hull for the eyes and mouth
                mouth_hull = cv2.convexHull(lmk_lst[mouth_lmk])
                left_eye_hull = cv2.convexHull(lmk_lst[left_eye_lmk])
                right_eye_hull = cv2.convexHull(lmk_lst[right_eye_lmk])

                # make the area in the mask for the eyes and mouth as 0
                mask = cv2.fillConvexPoly(mask, mouth_hull, 0)
                mask = cv2.fillConvexPoly(mask, left_eye_hull, 0)
                mask = cv2.fillConvexPoly(mask, right_eye_hull, 0)

                return mask
        else:
            # no face detected, return empty mask
            return np.zeros((frame_height, frame_width), np.uint8)
            
def selected_facial_roi(frame, frame_height, frame_width):
    '''
    This function creates a mask for particular facial regions [forehead+cheeks].

    Parameters:
        frame (numpy.ndarray): Frame to be processed.
        frame_height (int): Height of the frame.
        frame_width (int): Width of the frame.
    
    Returns:
        mask (numpy.ndarray): Mask of the frame.
    '''

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        
        # Process the frame to get the facial landmarks
        results = face_mesh.process(frame)

        # If face is detected
        if results.multi_face_landmarks:
            # Iterate over the detected faces
            for face_landmarks in results.multi_face_landmarks:

                # Get the list of facial landmarks
                landmarks = face_landmarks.landmark

                # Get points for forehead, left cheek, and right cheek
                forehead_points = np.array([(landmarks[lm_idx].x * frame_width, landmarks[lm_idx].y * frame_height) for lm_idx in forehead_lmk], np.int32)
                left_cheek_points = np.array([(landmarks[lm_idx].x * frame_width, landmarks[lm_idx].y * frame_height) for lm_idx in left_cheek_lmk], np.int32)
                right_cheek_points = np.array([(landmarks[lm_idx].x * frame_width, landmarks[lm_idx].y * frame_height) for lm_idx in right_cheek_lmk], np.int32)

                # Get the facial regions using Convex Hull
                hull_forehead = cv2.convexHull(forehead_points)
                hull_cheek_left = cv2.convexHull(left_cheek_points)
                hull_cheek_right = cv2.convexHull(right_cheek_points)

                # Create masks for each facial region
                mask_forehead = np.zeros((frame_height, frame_width), np.uint8)
                mask_cheek_left = np.zeros((frame_height, frame_width), np.uint8)
                mask_cheek_right = np.zeros((frame_height, frame_width), np.uint8)

                # Fill the masks with the facial regions
                mask_forehead = cv2.fillConvexPoly(mask_forehead, hull_forehead, 255)
                mask_cheek_left = cv2.fillConvexPoly(mask_cheek_left, hull_cheek_left, 255)
                mask_cheek_right = cv2.fillConvexPoly(mask_cheek_right, hull_cheek_right, 255)

                # Combine the masks
                mask = mask_forehead + mask_cheek_left + mask_cheek_right

                return mask
        else:
            # no face detected, return empty mask
            return np.zeros((frame_height, frame_width), np.uint8)

def boilerplate_roi(frame, frame_height, frame_width):
    '''
    This function is a boilerplate for creating new ROI functions.

    Parameters:
        frame (numpy.ndarray): Frame to be processed.
        frame_height (int): Height of the frame.
        frame_width (int): Width of the frame.
    
    Returns:
        mask (numpy.ndarray): Mask of the frame.
    '''

    # find new roi here and return mask. 
    
    # The mask should have 255 pixel values for the pixels that need to be edited by the filter and 0 pixel values 
    # for the pixels that need to be left unedited.

    pass