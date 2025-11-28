import numpy as np
from metavision_sdk_stream import Camera, CameraStreamSlicer, SliceCondition
import os
from datetime import datetime
import time

PIXEL_X = 100
PIXEL_Y = 100
EVENT_THRESHOLD = 10
BATCH_US = 10_000
MAX_SENSOR_X = 639
MAX_SENSOR_Y = 479

recording = False

cam = Camera.from_first_available()
slicer = CameraStreamSlicer(cam.move(), SliceCondition.make_n_us(BATCH_US))

batch = 0
events_file = f"events/filtered_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

with open(events_file, "w") as f:
    f.write("x_bin,y_bin,batch_start_t\n")

def downsample_to_grid(xs, ys):
    """Map sensor coordinates to 100x100 grid."""
    x_bins = (xs.astype(np.float32) / MAX_SENSOR_X * PIXEL_X).astype(np.int32)
    y_bins = (ys.astype(np.float32) / MAX_SENSOR_Y * PIXEL_Y).astype(np.int32)
    return (
        np.clip(x_bins, 0, PIXEL_X - 1),
        np.clip(y_bins, 0, PIXEL_Y - 1)
    )

# Precompute masks so this is O(1) at runtime
green_mask = np.zeros((PIXEL_X, PIXEL_Y), dtype=bool)
red_mask   = np.zeros((PIXEL_X, PIXEL_Y), dtype=bool)

# --- GREEN REGIONS ---
green_mask[0:2, :] = True                     # first 2 columns
green_mask[:, 99][0:50] = True                # top row, left half
green_mask[:, 0][0:50] = True                 # bottom row, left half

# --- RED REGIONS ---
red_mask[98:100, :] = True                    # last 2 columns
red_mask[:, 99][50:100] = True                # top row, right half
red_mask[:, 0][50:100] = True                 # bottom row, right half

print("Get Ready...")
time.sleep(1)
print("Reading Events...")

for sl in slicer:
    evs = sl.events
    if evs is None or evs.size == 0:
        continue

    # Normalise dtype if needed
    if evs.dtype.names != ('t','x','y','p'):
        evs = np.rec.fromarrays(
            [evs['t'], evs['x'], evs['y'], evs['p']],
            dtype=[('t','<u8'),('x','<u2'),('y','<u2'),('p','u1')]
        )

    batch += 1
    batch_start_t = int(evs['t'][0])

    # Downsample
    x_bins, y_bins = downsample_to_grid(evs['x'], evs['y'])

    # Build grid counts
    grid = np.zeros((PIXEL_X, PIXEL_Y), dtype=np.uint16)
    np.add.at(grid, (x_bins, y_bins), 1)

    # Check triggers
    green_triggered = np.any(grid[green_mask] >= EVENT_THRESHOLD)
    red_triggered   = np.any(grid[red_mask]   >= EVENT_THRESHOLD)

    # Recording state machine
    if green_triggered and not recording:
        print(">>> START RECORDING")
        recording = True

    if red_triggered and recording:
        print(">>> STOP RECORDING")
        recording = False

    # If we are not recording, continue
    if not recording:
        continue

    # Save active pixels
    xs, ys = np.where(grid >= EVENT_THRESHOLD)
    if xs.size > 0:
        results = np.column_stack((xs, ys, np.full(xs.shape, batch_start_t)))
        with open(events_file, "a") as f:
            np.savetxt(f, results, fmt='%d', delimiter=',')
