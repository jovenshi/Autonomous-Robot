import time
from sabertooth import Sabertooth

try:
    print("Initializing Sabertooth...")
    saber = Sabertooth()  # Default: address 128, 9600 baud
    saber.set_ramping(8)

    print("Driving Forward for 3 seconds...")
    saber.drive(50, 0)  # 50% speed, 0 turn
    time.sleep(3)

    print("Stopping motors...")
    saber.stop()

except Exception as e:
    print(f"Error: {e}")

finally:
    saber.close()
    print("Done.")
