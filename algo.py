import numpy as np
from metavision_sdk_stream import Camera, CameraStreamSlicer, SliceCondition
import os
from datetime import datetime
import time
import serial
from turret import Turret

PIXEL_X = 160
PIXEL_Y = 120
EVENT_THRESHOLD = 20
BATCH_US = 10_000
MAX_SENSOR_X = 640
MAX_SENSOR_Y = 480

recording = False
synced = False
buffered_events = []
t=Turret()

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

def fit_line(times, values):
    t0 = times[0]
    t = times - t0

    A = np.vstack([t, np.ones_like(t)]).T
    a, b = np.linalg.lstsq(A, values, rcond=None)[0]
    
    # Calculate R^2
    y_pred = a * t + b
    ss_res = np.sum((values - y_pred) ** 2)
    ss_tot = np.sum((values - np.mean(values)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return a, b - a * t0, r2

print("Get Ready...")
time.sleep(0)
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
        t.sync(batch_end_t, 0)
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

            mx, cx, r2_x = fit_line(ts, xs)
            my, cy, r2_y = fit_line(ts, ys)

            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # filename = f"events/events_{timestamp}.npy"
            # np.save(filename, arr)
            # print(f"Saved {len(arr)} events to {filename}")
            
            break

        buffered_events = []
        continue

    if not recording:
        continue


    active_cells = grid >= EVENT_THRESHOLD          # shape (PIXEL_X, PIXEL_Y)
    event_mask   = active_cells[x_bins, y_bins]     # boolean mask over events

    if np.any(event_mask):
        xs_evt = x_bins[event_mask]
        ys_evt = y_bins[event_mask]
        ts_evt = evs['t'][event_mask].astype(np.int64)

        batch_arr = np.column_stack((xs_evt, ys_evt, ts_evt))  # (N, 3)
        buffered_events.append(batch_arr)
