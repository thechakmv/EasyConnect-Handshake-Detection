import serial
import csv
import time
from datetime import datetime
import os

# Set up serial connection
ser = serial.Serial('COM6', 115200, timeout=1)
time.sleep(2)  # Allow Arduino to reset

def record_one_second_session(sample_rate_hz=100, folder='Python/data/non_handshake', prefix='handshake'):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(folder, f"{prefix}_{timestamp}.csv")

    print(f"Recording: {filename}")
    ser.write(b'read\n')

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'])

        start_time = time.time()
        sample_count = 0
        try:
            while time.time() - start_time < 1.0:  # 1 second duration
                line = ser.readline().decode().strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 6:
                        try:
                            row = [float(p) for p in parts]
                            writer.writerow(row)
                            sample_count += 1
                        except ValueError:
                            print("Invalid numeric data, skipping.")
        except Exception as e:
            print(f"Error during recording: {e}")
        finally:
            ser.write(b'sleep\n')
            print(f"Recorded {sample_count} samples.\n")

# Create output directory if it doesn't exist
os.makedirs("Python/data", exist_ok=True)

# Continuous loop
try:
    print("Starting continuous handshake recording. Press Ctrl+C to stop.")
    while True:
        record_one_second_session()
        time.sleep(2)  # Wait 2 more seconds (3 seconds total including the 1s recording)
except KeyboardInterrupt:
    print("\nData collection stopped by user.")
finally:
    ser.close()
    print("Serial connection closed.")