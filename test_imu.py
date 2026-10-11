import time
from imu_controller import IMUDevice

imu = IMUDevice()
try:
    for _ in range(20):
        print(f"heading: {imu.heading()}   delta: {imu.delta()}")
        time.sleep(0.5)
finally:
    imu.close()