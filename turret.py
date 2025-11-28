import serial

class Turret:
    def __init__(self, port="/dev/ttyUSB0", baud=115200):
        self.ser = serial.Serial(port, baud, timeout=1)

    def send(self, cmd: str):
        self.ser.write((cmd + "\n").encode())

    def trigger(self, v: int):
        self.send(f"trigger {v}")

    def yaw(self, v: int):
        self.send(f"yaw {v}")

    def pitch(self, v: int):
        self.send(f"pitch {v}")
