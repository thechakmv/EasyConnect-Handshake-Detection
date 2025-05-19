import serial
import csv
import time
from datetime import datetime

# Set up serial connection
ser = serial.Serial('COM6', 115200, timeout=1)
time.sleep(2)  # Allow Arduino to reset

# Timestamped filename
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"Python/data/sensor_data_{timestamp}.csv"

# Countdown before logging
print("Starting in:")
for i in range(1, 0, -1):
    print(i)
    time.sleep(1)
print("Recording now...\n")

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write CSV headers (no timestamp column)
    writer.writerow([
        'Accel_X', 'Accel_Y', 'Accel_Z',
        'Gyro_X', 'Gyro_Y', 'Gyro_Z'
    ])

    sample_count = 0
    try:
        while sample_count < 100:
            line = ser.readline().decode().strip()
            if line:
                parts = line.split(',')
                if len(parts) == 6:
                    try:
                        row = [float(p) for p in parts]
                        writer.writerow(row)
                        print(row)
                        sample_count += 1
                    except ValueError:
                        print("Invalid numeric data, skipping.")
    except Exception as e:
        print(f"Error: {e}")

    finally:
        ser.close()
        print(f"\nLogging complete. Saved to {filename}. Serial port closed.")