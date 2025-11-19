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
buffered_events = []   # store (x, y, t)

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
    """
    Linear regression: values ≈ a * t + b
    Returns (a, b).
    """
    # center for numerical stability
    t0 = times[0]
    t = times - t0

    A = np.vstack([t, np.ones_like(t)]).T
    a, b = np.linalg.lstsq(A, values, rcond=None)[0]
    return a, b - a * t0   # uncenter b

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

    green_triggered = np.any(grid[green_mask] >= EVENT_THRESHOLD)
    red_triggered   = np.any(grid[red_mask]   >= EVENT_THRESHOLD)

    if green_triggered and not recording:
        print(">>> START RECORDING")
        recording = True
        buffered_events = []  # reset buffer

    if red_triggered and recording:
        print(">>> STOP RECORDING")
        recording = False

        if len(buffered_events) > 0:
            arr = np.array(buffered_events)  # (N, 3) → x, y, t

            xs = arr[:,0]
            ys = arr[:,1]
            ts = arr[:,2]

            # Fit x(t) and y(t)
            ax, bx = fit_line(ts, xs)
            ay, by = fit_line(ts, ys)

            print(f"\n=== Best-fit trajectories ===")
            print(f"x(t) = {ax:.6f} * t + {bx:.6f}")
            print(f"y(t) = {ay:.6f} * t + {by:.6f}")
            print("=============================\n")

            # Save to CSV
            out_file = f"events/recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            np.savetxt(out_file, arr, fmt='%d', delimiter=',', header="x,y,t", comments="")
            print(f"Saved recording to {out_file}")

        # Clear for next run
        buffered_events = []
        continue

    # If not recording, skip
    if not recording:
        continue

    xs, ys = np.where(grid >= EVENT_THRESHOLD)
    if xs.size > 0:
        ts = np.full(xs.shape, batch_start_t, dtype=np.int64)
        buffered_events.append(np.column_stack((xs, ys, ts)))
        # Flatten nested array
        if isinstance(buffered_events[-1], np.ndarray) and buffered_events[-1].ndim == 2:
            buffered_events[-1] = buffered_events[-1].reshape(-1, 3)
        # Convert list of arrays to single array lazily
        buffered_events = list(buffered_events)
        buffered_events[-1] = np.array(buffered_events[-1]).reshape(-1,3)