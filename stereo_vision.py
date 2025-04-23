import cv2
import numpy as np

def detect_stereo_obstacles(left_frame, right_frame):
    # Konversi ke grayscale
    gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    # Buat stereo matcher (StereoBM untuk efisiensi)
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

    # Hitung disparity map
    disparity_left = stereo.compute(gray_left, gray_right)  # Depth untuk kiri
    disparity_right = stereo.compute(gray_right, gray_left)  # Depth untuk kanan

    # Normalisasi untuk ditampilkan
    disparity_left_normalized = cv2.normalize(disparity_left, None, alpha=0, beta=255, 
                                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disparity_right_normalized = cv2.normalize(disparity_right, None, alpha=0, beta=255, 
                                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return disparity_left_normalized, disparity_right_normalized