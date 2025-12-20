import os
import time
import numpy as np
import multiprocessing as mp
from datetime import datetime
import signal
from metavision_sdk_stream import Camera, CameraStreamSlicer, SliceCondition

TRIGGER_DELAY = 180000
Y_CAM_XY = 115
Z_CAM_XZ = 88
TAN_THETA_X_2 = 0.5726
TAN_THETA_Y_2 = TAN_THETA_X_2 * 0.75

EVENT_THRESHOLD = 10
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

    # Quadratic regression: coord(t) = a*t^2 + b*t + c
    A = np.vstack([t**2, t, np.ones_like(t)]).T

    a_x, b_x, c_x = np.linalg.lstsq(A, xs, rcond=None)[0]
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

def camera_worker(serial, result_q, start_evt, stop_evt):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    green_mask, red_mask = build_masks()
    recording = False
    buffered_events = []
    cam = Camera.from_serial(serial)
    slicer = CameraStreamSlicer(cam.move(), SliceCondition.make_n_us(BATCH_US))

    start_evt.wait()

    for sl in slicer:
        if stop_evt.is_set():
            break
        evs = sl.events
        if evs is None or evs.size == 0:
            continue

        if serial == SERIAL_XZ:
            evs['x'] = MAX_SENSOR_X - evs['x']-1

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

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join("events", f"{serial}_events_{timestamp}.npy")

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

                np.save(filename, arr)

            buffered_events = []
            continue

        if not recording:
            continue

        # Gate events by active cells, but store original (full-res) events
        active_cells = grid >= EVENT_THRESHOLD
        event_mask   = active_cells[x_bins, y_bins]

        if np.any(event_mask):
            xs_evt = evs['x'][event_mask].astype(np.int32)
            ys_evt = evs['y'][event_mask].astype(np.int32)
            ts_evt = evs['t'][event_mask].astype(np.int64)

            batch_arr = np.column_stack((xs_evt, ys_evt, ts_evt))
            buffered_events.append(batch_arr)


def pixel_from_fit(result, t_abs):
    """Evaluate x(t), y(t) in *pixel* coordinates at absolute camera time t_abs."""
    t_rel = t_abs - result["t_min"]
    x = result["a_x"] * t_rel * t_rel + result["b_x"] * t_rel + result["c_x"]
    y = result["a_y"] * t_rel * t_rel + result["b_y"] * t_rel + result["c_y"]
    return x, y

def pixel_to_ray_xy(x_pix, y_pix):
    origin = np.array([0, Y_CAM_XY, 0])
    dir = np.array([(2*x_pix/MAX_SENSOR_X-1)*TAN_THETA_X_2, (2*y_pix/MAX_SENSOR_Y-1)*TAN_THETA_Y_2, 1])
    return origin, dir

def pixel_to_ray_xz(x_pix, y_pix):
    origin = np.array([0, 0, Z_CAM_XZ])
    dir = np.array([(2*x_pix/MAX_SENSOR_X-1)*TAN_THETA_X_2, 1, (2*y_pix/MAX_SENSOR_Y-1)*TAN_THETA_Y_2])
    return origin, dir

def closest_midpoint_between_lines(P1, d1, P2, d2, eps=1e-9):
    P1 = np.asarray(P1, dtype=np.float64)
    P2 = np.asarray(P2, dtype=np.float64)
    d1 = np.asarray(d1, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)

    n1 = np.linalg.norm(d1)
    n2 = np.linalg.norm(d2)
    d1 = d1 / n1
    d2 = d2 / n2

    w0 = P1 - P2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)
    denom = a * c - b * b

    if abs(denom) < eps:
        # Nearly parallel: project P1 onto line2 and midpoint between them
        if c < eps:
            Q1 = P1
            Q2 = P2
        else:
            t = e / c
            Q1 = P1
            Q2 = P2 + t * d2
    else:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
        Q1 = P1 + s * d1
        Q2 = P2 + t * d2

    return 0.5 * (Q1 + Q2)

if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    result_q = mp.Queue()
    stop_evt = mp.Event()
    start_evt = mp.Event()

    pxy = mp.Process(target=camera_worker, args=(SERIAL_XY, result_q, start_evt, stop_evt))
    pxz = mp.Process(target=camera_worker, args=(SERIAL_XZ, result_q, start_evt, stop_evt))

    print("Starting dual-camera capture in staggered processes...")
    pxy.start()
    time.sleep(1)
    pxz.start()
    time.sleep(1)
    print("Lifted capture barrier!")
    start_evt.set()

    try:
        resultXY = None
        resultXZ = None
        while not (resultXY and resultXZ):
            msg = result_q.get()
            if msg.get("type") == "trajectory":
                if msg["serial"] == SERIAL_XY and resultXY is None:
                    resultXY = msg
                elif msg["serial"] == SERIAL_XZ and resultXZ is None:
                    resultXZ = msg

        stop_evt.set()
        pxy.join()
        pxz.join()

        t_start = max(resultXY["t_min"], resultXZ["t_min"])
        t_end   = min(resultXY["t_max"], resultXZ["t_max"])

        N = 100
        times = np.linspace(t_start, t_end, N)

        points = []

        for t_abs in times:
            x_xy, y_xy = pixel_from_fit(resultXY, t_abs)
            x_xz, y_xz = pixel_from_fit(resultXZ, t_abs)
            P1, d1 = pixel_to_ray_xy(x_xy, y_xy)
            P2, d2 = pixel_to_ray_xz(x_xz, y_xz)
            mid = closest_midpoint_between_lines(P1, d1, P2, d2)
            points.append([t_abs, mid[0], mid[1], mid[2]])

        points = np.asarray(points, dtype=np.float64)
        out_name = "triangulated_midpoints.npy"
        np.save(out_name, points)
        print(f"Saved {points.shape[0]} 3D midpoints to {out_name}")

    except KeyboardInterrupt:
        stop_evt.set()
        pxy.join()
        pxz.join()
        print("Stopped by user")
