import numpy as np
from metavision_sdk_stream import Camera, CameraStreamSlicer, SliceCondition
import os
from datetime import datetime
import time
import serial
from turret import Turret

# Measured Values
X_AC = 624
Y_AC = 37
Z_AC = 158.0
V_AC = 9.4e-4
FOV_X = 192
TIME_DELAY = 0

# Values to Calibrate
SYNC_DELAY = 0
TRIGGER_DELAY = 180000

PIXEL_X = 160
PIXEL_Y = 120
EVENT_THRESHOLD = 20
BATCH_US = 10_000
MAX_SENSOR_X = 640
MAX_SENSOR_Y = 480
G = 1.78e-10 #cm/us^2

recording = False
synced = False
buffered_events = []
#t=Turret()

cam = Camera.from_first_available()
slicer = CameraStreamSlicer(cam.move(), SliceCondition.make_n_us(BATCH_US))

green_mask = np.zeros((PIXEL_X, PIXEL_Y), dtype=bool)
red_mask   = np.zeros((PIXEL_X, PIXEL_Y), dtype=bool)

green_mask[0:2, :] = True
green_mask[:, PIXEL_Y - 1][0:PIXEL_X//2] = True
green_mask[:, 0][0:PIXEL_X//2] = True

red_mask[PIXEL_X - 2:PIXEL_X, :] = True
red_mask[:, PIXEL_Y - 1][PIXEL_X//2:PIXEL_X] = True
red_mask[:, 0][PIXEL_X//2:PIXEL_X] = True

def downsample_to_grid(xs, ys):
    x_bins = (xs.astype(np.float32) / MAX_SENSOR_X * PIXEL_X).astype(np.int32)
    y_bins = (ys.astype(np.float32) / MAX_SENSOR_Y * PIXEL_Y).astype(np.int32)
    return (x_bins, y_bins)

def fit_projectile(xs, ys, ts):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)

    cm_per_px = FOV_X / PIXEL_X
    x_cm = xs * cm_per_px
    y_cm = ys * cm_per_px

    t_min = ts.min()
    t = ts - t_min
    A = np.vstack([t, np.ones_like(t)]).T
    v_x, x0 = np.linalg.lstsq(A, x_cm, rcond=None)[0]

    y_corr = y_cm + 0.5 * (G * t) * t
    v_y, y0 = np.linalg.lstsq(A, y_corr, rcond=None)[0]

    # x = x0 + v_x(t-t_min)
    # y = y0 + v_y(t-t_min) - 0.5G(t-t_min)^2
    return x0, v_x, y0, v_y, t_min

print("Get Ready...")
time.sleep(TIME_DELAY)
print("Reading Events...")

for sl in slicer:
    evs = sl.events
    if evs is None or evs.size == 0:
        continue

    if evs.dtype.names != ('t','x','y','p'):
        evs = np.rec.fromarrays(
            [evs['t'], evs['x'], evs['y'], evs['p']],
            dtype=[('t','<u8'),('x','<u2'),('y','<u2'),('p','u1')]
        )


    if not synced:
        batch_end_t = int(evs['t'][-1])
        #t.sync(batch_end_t, SYNC_DELAY)
        synced = True
    
    x_bins, y_bins = downsample_to_grid(evs['x'], evs['y'])
    grid = np.zeros((PIXEL_X, PIXEL_Y), dtype=np.uint16)
    np.add.at(grid, (x_bins, y_bins), 1)

    green_triggered = np.any(grid[green_mask] >= EVENT_THRESHOLD)
    red_triggered   = np.any(grid[red_mask]   >= EVENT_THRESHOLD)

    if green_triggered and not recording:
        print(">>> START RECORDING")
        recording = True
        buffered_events = []

    if red_triggered and recording:
        print(">>> STOP RECORDING")
        recording = False
        # print(f"Recording stopped at: {datetime.now().isoformat(timespec='microseconds')}")
        if len(buffered_events) > 0:
            arr = np.vstack(buffered_events)

            xs = arr[:,0]
            ys = arr[:,1]
            ts = arr[:,2]

            x_0, v_x, y_0, v_y, t_min = fit_projectile(xs, ys, ts)
            print(x_0, v_x, y_0, v_y, sep=',')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"events/events_{timestamp}.npy"
            np.save(filename, arr)
            print(f"Saved {len(arr)} events to {filename}")

        buffered_events = []
        continue

    if not recording:
        continue


    active_cells = grid >= EVENT_THRESHOLD          # shape (PIXEL_X, PIXEL_Y)
    event_mask   = active_cells[x_bins, y_bins]     # boolean mask over events

    if np.any(event_mask):
        if np.any(event_mask):
            xs_evt = evs['x'][event_mask]
            ys_evt = evs['y'][event_mask]
            ts_evt = evs['t'][event_mask].astype(np.int64)

            batch_arr = np.column_stack((xs_evt, ys_evt, ts_evt))  # (N, 3)
            buffered_events.append(batch_arr)

