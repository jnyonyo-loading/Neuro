# === Python: send a stream of test voltages to the Arduino ===
import serial
import time

arduino = serial.Serial('/dev/cu.usbmodem14401', 9600, timeout=1)  # match your actual port
time.sleep(2)  # wait for Arduino to reset after connection

# Dummy test values first -- NOT real pipeline output yet
test_voltages = [0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5, 0.0]

while True:
    for v in test_voltages:
        arduino.write(f"{v}\n".encode())
        time.sleep(0.5)  # controls how fast values stream -- start slow
        response = arduino.readline().decode().strip()
        print(response)
