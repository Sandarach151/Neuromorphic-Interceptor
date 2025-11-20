import serial, time

PORT = "/dev/ttyUSB0"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

def send(x):
    ser.write((x + "\n").encode())
    return ser.readline().decode().strip()

def trigger(v):
    return send(f"trigger {v}")

def yaw(v):
    return send(f"yaw {v}")

def pitch(v):
    return send(f"pitch {v}")

time.sleep(1)
yaw(180)
time.sleep(1)
yaw(0)
time.sleep(1)
