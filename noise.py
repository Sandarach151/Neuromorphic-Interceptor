import numpy as np
from metavision_sdk_stream import Camera, CameraStreamSlicer, SliceCondition

PIXEL_X = 160
PIXEL_Y = 120
MAX_SENSOR_X = 640
MAX_SENSOR_Y = 480
BATCH_US = 10_000
WINDOW_US = 1000_000

serial_no = "00001033"
# 00001033 or 00000768

cam = Camera.from_serial(serial_no)
slicer = CameraStreamSlicer(cam.move(), SliceCondition.make_n_us(BATCH_US))
it = iter(slicer)

def downsample_to_grid(xs, ys):
    x_bins = (xs.astype(np.float32) / MAX_SENSOR_X * PIXEL_X).astype(np.int32)
    y_bins = (ys.astype(np.float32) / MAX_SENSOR_Y * PIXEL_Y).astype(np.int32)
    return (x_bins, y_bins)

def has_data(thr):
    t0 = None
    while True:
        sl = next(it)
        evs = sl.events
        if evs is None or evs.size == 0:
            continue

        if t0 is None:
            t0 = int(evs["t"][0])

        xb, yb = downsample_to_grid(evs['x'], evs['y'])

        grid = np.zeros((PIXEL_X, PIXEL_Y), dtype=np.uint16)
        np.add.at(grid, (xb, yb), 1)

        if grid.max() >= thr:
            return True  # "got any data" => threshold too low

        if int(evs["t"][-1]) - t0 >= WINDOW_US:
            return False # no data in 100ms => threshold high enough


hi = 1024
lo = 0

while lo + 1 < hi:
    mid = (lo + hi) // 2
    if has_data(mid):
        print("too low:", mid)
        lo = mid
    else:
        print("ok:", mid)
        hi = mid

print("\nEVENT_THRESHOLD =", hi)
