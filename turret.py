import serial
import time

class Turret:
    def __init__(self, port="/dev/ttyUSB0", baud=115200):
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2.0)                 # allow Arduino reboot/bootloader to finish
        self.ser.reset_input_buffer()

    def send(self, cmd: str):
        self.ser.write((cmd + "\n").encode())

    def trigger(self, v: int):
        self.send(f"trigger {v}")

    def yaw(self, v: int):
        self.send(f"yaw {v}")

    def pitch(self, v: int):
        self.send(f"pitch {v}")

    def sync(self, cam_t: int, delay: int):
        return self.send(f"SYNC {cam_t} {delay}")

    def fire(self, pitch_angle: int, yaw_angle: int, t_fire_cam: int):
        return self.send(f"FIRE {pitch_angle} {yaw_angle} {t_fire_cam}")

t = Turret()
t.pitch(120)
t.yaw(120)
time.sleep(0.15)
t.trigger(60)
time.sleep(1.0)
t.trigger(110)
