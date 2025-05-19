import serial
import csv
import time
from datetime import datetime

# Set up serial connection
ser = serial.Serial('COM6', 115200, timeout=1)
time.sleep(2)  # Allow Arduino to reset

# Create timestamped filename
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"Python/data/sensor_data_{timestamp}.csv"  # Save in current directory

# Countdown before logging
print("Starting in:")
for i in range(1, 0, -1):
    print(i)
    time.sleep(1)
print("Recording now...\n")

# Record start time
start_time = time.time()

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write CSV headers
    writer.writerow([
        'Timestamp',
        'Orientation_X', 'Orientation_Y', 'Orientation_Z',
        'Accel_X', 'Accel_Y', 'Accel_Z',
        'Gyro_X', 'Gyro_Y', 'Gyro_Z'
    ])

    try:
        while True:
            current_time = time.time()
            if current_time - start_time > 5:
                break  # Stop after 5 seconds

            line = ser.readline().decode().strip()
            if line:
                parts = line.split(',')
                if len(parts) == 9:
                    row = [current_time] + [float(p) for p in parts]
                    writer.writerow(row)
                    print(row)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        ser.close()
        print("\nLogging complete. Serial port closed.")
