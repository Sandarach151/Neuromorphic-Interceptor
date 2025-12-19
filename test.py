import os
import time
import numpy as np
import multiprocessing as mp
from datetime import datetime
from metavision_sdk_stream import Camera, CameraStreamSlicer, SliceCondition

TRIGGER_DELAY = 180000

EVENT_THRESHOLD = 20
PIXEL_X = 160
PIXEL_Y = 120
BATCH_US = 10_000
MAX_SENSOR_X = 640
MAX_SENSOR_Y = 480
G = 1.78e-10  # cm/us^2

SERIAL_XY = "00001033"
SERIAL_XZ = "00000768"

def downsample_to_grid(xs, ys):
    x_bins = (xs.astype(np.float32) / MAX_SENSOR_X * PIXEL_X).astype(np.int32)
    y_bins = (ys.astype(np.float32) / MAX_SENSOR_Y * PIXEL_Y).astype(np.int32)
    return (x_bins, y_bins)

def fit_projectile(xs, ys, ts):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)

    t_min = ts.min()
    t = ts - t_min

    # Quadratic regression: y = a*t^2 + b*t + c
    A = np.vstack([t**2, t, np.ones_like(t)]).T
    
    # Fit x(t) = a_x*t^2 + b_x*t + c_x
    a_x, b_x, c_x = np.linalg.lstsq(A, xs, rcond=None)[0]
    
    # Fit y(t) = a_y*t^2 + b_y*t + c_y
    a_y, b_y, c_y = np.linalg.lstsq(A, ys, rcond=None)[0]

    return (a_x, b_x, c_x), (a_y, b_y, c_y), t_min

def build_masks():
    green_mask = np.zeros((PIXEL_X, PIXEL_Y), dtype=bool)
    red_mask   = np.zeros((PIXEL_X, PIXEL_Y), dtype=bool)

    green_mask[0:2, :] = True
    green_mask[:, PIXEL_Y - 1][0:PIXEL_X // 2] = True
    green_mask[:, 0][0:PIXEL_X // 2] = True

    red_mask[PIXEL_X - 2:PIXEL_X, :] = True
    red_mask[:, PIXEL_Y - 1][PIXEL_X // 2:PIXEL_X] = True
    red_mask[:, 0][PIXEL_X // 2:PIXEL_X] = True

    return green_mask, red_mask

def camera_worker(serial, result_q):
    green_mask, red_mask = build_masks()

    recording = False
    buffered_events = []
    cam = Camera.from_serial(serial)
    slicer = CameraStreamSlicer(cam.move(), SliceCondition.make_n_us(BATCH_US))

    for sl in slicer:
        evs = sl.events
        if evs is None or evs.size == 0:
            continue

        x_bins, y_bins = downsample_to_grid(evs['x'], evs['y'])
        grid = np.zeros((PIXEL_X, PIXEL_Y), dtype=np.uint16)
        np.add.at(grid, (x_bins, y_bins), 1)

        green_triggered = np.any(grid[green_mask] >= EVENT_THRESHOLD)
        red_triggered   = np.any(grid[red_mask]   >= EVENT_THRESHOLD)

        if green_triggered and not recording:
            recording = True
            buffered_events = []

        if red_triggered and recording:
            recording = False

            if len(buffered_events) > 0:
                arr = np.vstack(buffered_events)

                xs = arr[:, 0]
                ys = arr[:, 1]
                ts = arr[:, 2]

                (a_x, b_x, c_x), (a_y, b_y, c_y), t_min = fit_projectile(xs, ys, ts)
                t_max = ts.max()

                result_q.put({
                    "type": "trajectory",
                    "serial": serial,
                    "c_x": float(c_x),
                    "b_x": float(b_x),
                    "c_y": float(c_y),
                    "b_y": float(b_y),
                    "a_x": float(a_x),
                    "a_y": float(a_y),
                    "t_min": float(t_min),
                    "t_max": float(t_max),
                    "n_events": int(arr.shape[0]),
                    "file": filename
                })

                timestamp = datetime.now().strftime("%d_%H%M%S")
                filename = os.path.join(f"events/{serial}_events_{timestamp}.npy")
                np.save(filename, arr)

            buffered_events = []
            continue

        if not recording:
            continue

        # Gate events by active cells, but STORE ORIGINAL events (full-res)
        active_cells = grid >= EVENT_THRESHOLD
        event_mask = active_cells[x_bins, y_bins]

        if np.any(event_mask):
            xs_evt = evs['x'][event_mask].astype(np.int32)
            ys_evt = evs['y'][event_mask].astype(np.int32)
            ts_evt = evs['t'][event_mask].astype(np.int64)

            batch_arr = np.column_stack((xs_evt, ys_evt, ts_evt))
            buffered_events.append(batch_arr)

if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    result_q = mp.Queue()

    pxy = mp.Process(target=camera_worker, args=(SERIAL_XY, result_q))
    pxz = mp.Process(target=camera_worker, args=(SERIAL_XZ, result_q))

    print("Starting dual-camera capture in separate processes...")
    pxy.start()
    pxz.start()

    try:
        resultXY = None
        resultXZ = None
        while not (resultXY and resultXZ):
            msg = result_q.get()
            if msg['type'] == "trajectory":
                if msg['serial']==SERIAL_XY:
                    resultXY = msg
                if msg['serial']==SERIAL_XZ:
                    resultXZ = msg
        
        t_start = max(resultXY.t_min, resultXZ.t_min)
        t_end = min(resultXY.t_max, resultXZ.t_max)
        

            
    except KeyboardInterrupt:
        time.sleep(0.2)

        while True:
            try:
                msg = result_q.get_nowait()
                print_msg(msg)
            except Exception:
                break

        p1.terminate()
        p2.terminate()
        p1.join()
        p2.join()

        print("Stopped.")
