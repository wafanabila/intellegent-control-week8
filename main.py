import cv2
from yolo_detector import detect_objects
from camera_utils import open_cameras

def main():
    cam_bawah, cam_kiri, cam_kanan = open_cameras()

    while True:
        ret1, frame_bawah = cam_bawah.read()
        ret2, frame_kiri = cam_kiri.read()
        ret3, frame_kanan = cam_kanan.read()

        if not (ret1 and ret2 and ret3):
            print("Gagal mengambil frame dari kamera.")
            break

        # Deteksi kerusakan bantalan dengan YOLO
        hasil_bantalan, _ = detect_objects(frame_bawah, conf=0.4)

        # Gabungkan stereo view
        stereo_frame = cv2.hconcat([frame_kiri, frame_kanan])
        hasil_stereo, _ = detect_objects(stereo_frame, conf=0.4)

        # Tampilkan
        cv2.imshow("Deteksi Bantalan (YOLO)", hasil_bantalan)
        cv2.imshow("Deteksi Jalur (Stereo + YOLO)", hasil_stereo)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam_bawah.release()
    cam_kiri.release()
    cam_kanan.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
