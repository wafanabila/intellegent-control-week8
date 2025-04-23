import cv2

def open_cameras():
    cam_bawah = cv2.VideoCapture(0)
    cam_kiri = cv2.VideoCapture(1)
    cam_kanan = cv2.VideoCapture(2)

    # Bisa disesuaikan untuk resolusi dan pengaturan lainnya
    return cam_bawah, cam_kiri, cam_kanan
