import cupy as np
from metavision_sdk_stream import Camera, CameraStreamSlicer, SliceCondition
import os
from datetime import datetime
import time
import serial

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

PIXEL_X = 100
PIXEL_Y = 100
EVENT_THRESHOLD = 10
BATCH_US = 10_000
MAX_SENSOR_X = 639
MAX_SENSOR_Y = 479

recording = False
buffered_events = []   # list of (N,3) arrays

cam = Camera.from_first_available()
slicer = CameraStreamSlicer(cam.move(), SliceCondition.make_n_us(BATCH_US))

batch = 0

def downsample_to_grid(xs, ys):
    """Map sensor coordinates to 100x100 grid."""
    x_bins = (xs.astype(np.float32) / MAX_SENSOR_X * PIXEL_X).astype(np.int32)
    y_bins = (ys.astype(np.float32) / MAX_SENSOR_Y * PIXEL_Y).astype(np.int32)
    return (
        np.clip(x_bins, 0, PIXEL_X - 1),
        np.clip(y_bins, 0, PIXEL_Y - 1)
    )

green_mask = np.zeros((PIXEL_X, PIXEL_Y), dtype=bool)
red_mask   = np.zeros((PIXEL_X, PIXEL_Y), dtype=bool)

green_mask[0:2, :] = True
green_mask[:, 99][0:50] = True
green_mask[:, 0][0:50]  = True

red_mask[98:100, :] = True
red_mask[:, 99][50:100] = True
red_mask[:, 0][50:100]  = True

def fit_line(times, values):
    """Fit linear: values ≈ a * t + b."""
    t0 = times[0]
    t = times - t0

    A = np.vstack([t, np.ones_like(t)]).T
    a, b = np.linalg.lstsq(A, values, rcond=None)[0]
    return a, b - a * t0

print("Get Ready...")
time.sleep(5)
print("Reading Events...")

for sl in slicer:
    evs = sl.events
    if evs is None or evs.size == 0:
        continue

    # Normalize dtype
    if evs.dtype.names != ('t','x','y','p'):
        evs = np.rec.fromarrays(
            [evs['t'], evs['x'], evs['y'], evs['p']],
            dtype=[('t','<u8'),('x','<u2'),('y','<u2'),('p','u1')]
        )

    batch += 1
    batch_start_t = int(evs['t'][0])

    x_bins, y_bins = downsample_to_grid(evs['x'], evs['y'])

    grid = np.zeros((PIXEL_X, PIXEL_Y), dtype=np.uint16)
    np.add.at(grid, (x_bins, y_bins), 1)

    green_triggered = np.any(grid[green_mask] >= EVENT_THRESHOLD)
    red_triggered   = np.any(grid[red_mask]   >= EVENT_THRESHOLD)

    # --- START ---
    if green_triggered and not recording:
        print(">>> START RECORDING")
        recording = True
        buffered_events = []

    # --- STOP ---
    if red_triggered and recording:
        print(">>> STOP RECORDING")
        recording = False
        # print(f"Recording stopped at: {datetime.now().isoformat(timespec='microseconds')}")
        if len(buffered_events) > 0:
            # FIXED: merge the list of (N_i, 3) arrays
            arr = np.vstack(buffered_events)  # <-- correct way

            xs = arr[:,0]
            ys = arr[:,1]
            ts = arr[:,2]

            ax, bx = fit_line(ts, xs)
            ay, by = fit_line(ts, ys)
            trigger(50)
            time.sleep(1)
            trigger(110)

        buffered_events = []
        continue

    if not recording:
        continue

    # Store events for this batch
    xs, ys = np.where(grid >= EVENT_THRESHOLD)
    if xs.size > 0:
        ts = np.full(xs.shape, batch_start_t, dtype=np.int64)
        batch_arr = np.column_stack((xs, ys, ts))  # shape (N,3)
        buffered_events.append(batch_arr)
