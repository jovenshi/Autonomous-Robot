import time
import cv2
from display import RobotDisplay

def try_usb_camera(disp):
    print("Checking for USB Cameras using V4L2 backend...")
    # Explicitly force cv2.CAP_V4L2 to bypass GStreamer backend on Pi 5
    for idx in range(10):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"--> USB Camera found at index {idx}!")
                disp.set_status(f"USB Camera (Index {idx}) Active", position="top", bg_color=(0, 100, 0), text_color=(255, 255, 255))
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        disp.set_frame(frame)
                        disp.poll_events()
                        time.sleep(0.01)
                except KeyboardInterrupt:
                    pass
                finally:
                    cap.release()
                return True
            cap.release()
    return False

def try_pi_ribbon_camera(disp):
    print("Checking for Raspberry Pi Ribbon Camera (Picamera2)...")
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        picam2.start()
        disp.set_status("Pi Ribbon Camera Active", position="top", bg_color=(0, 100, 0), text_color=(255, 255, 255))
        print("--> Pi Ribbon Camera connected! Press Ctrl+C in terminal to exit.")

        try:
            while True:
                # Picamera2 returns RGB numpy array; convert to OpenCV BGR
                frame_rgb = picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                disp.set_frame(frame_bgr)
                disp.poll_events()
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            picam2.stop()
        return True
    except Exception as e:
        print(f"Picamera2 check failed: {e}")
        return False

def main():
    disp = RobotDisplay(fullscreen=False, size=(800, 480))

    # 1. Try USB camera via V4L2
    if try_usb_camera(disp):
        disp.close()
        return

    # 2. Try Pi Ribbon camera via Picamera2
    if try_pi_ribbon_camera(disp):
        disp.close()
        return

    print("\n[ERROR] No working USB or Pi Ribbon camera detected.")
    input("Press Enter to close...")
    disp.close()

if __name__ == "__main__":
    main()
