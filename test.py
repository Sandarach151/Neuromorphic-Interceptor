import os
import time
import numpy as np
import multiprocessing as mp
from datetime import datetime
from metavision_sdk_stream import Camera, CameraStreamSlicer, SliceCondition

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
G = 1.78e-10  # cm/us^2

SERIAL_1 = "00001033"
SERIAL_2 = "00000768"

def downsample_to_grid(xs, ys):
    x_bins = (xs.astype(np.float32) / MAX_SENSOR_X * PIXEL_X).astype(np.int32)
    y_bins = (ys.astype(np.float32) / MAX_SENSOR_Y * PIXEL_Y).astype(np.int32)
    return (x_bins, y_bins)

def fit_projectile(xs, ys, ts):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)

    # IMPORTANT:
    # We are storing full-res sensor pixels (0..MAX_SENSOR_X-1).
    # So the physically consistent scale is:
    cm_per_px = FOV_X / MAX_SENSOR_X
    # If you want to match your old grid-based scaling exactly, use:
    # cm_per_px = FOV_X / PIXEL_X

    x_cm = xs * cm_per_px
    y_cm = ys * cm_per_px

    t_min = ts.min()
    t = ts - t_min

    A = np.vstack([t, np.ones_like(t)]).T
    v_x, x0 = np.linalg.lstsq(A, x_cm, rcond=None)[0]

    y_corr = y_cm + 0.5 * (G * t) * t
    v_y, y0 = np.linalg.lstsq(A, y_corr, rcond=None)[0]

    return x0, v_x, y0, v_y, t_min

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

                x0, vx, y0, vy, t_min = fit_projectile(xs, ys, ts)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(f"events/{serial}_events_{timestamp}.npy")
                np.save(filename, arr)

                result_q.put({
                    "type": "trajectory",
                    "cam": serial,
                    "x0": float(x0),
                    "vx": float(vx),
                    "y0": float(y0),
                    "vy": float(vy),
                    "t_min": float(t_min),
                    "n_events": int(arr.shape[0]),
                    "file": filename
                })
            else:
                result_q.put({
                    "type": "trajectory",
                    "cam": serial,
                    "error": "No buffered events"
                })

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


def print_msg(msg):
    if msg.get("type") == "status":
        print(f"[{msg['cam']}] {msg['msg']} @ t={msg['t']}")

    elif msg.get("type") == "trajectory":
        if "error" in msg:
            print(f"[{msg['cam']}] Trajectory ERROR: {msg['error']}")
        else:
            print(
                f"[{msg['cam']}] Trajectory: "
                f"x0={msg['x0']:.6f}, vx={msg['vx']:.6f}, "
                f"y0={msg['y0']:.6f}, vy={msg['vy']:.6f}, "
                f"t_min={msg['t_min']:.0f} us, "
                f"events={msg['n_events']}, file={msg['file']}"
            )

    elif msg.get("type") == "error":
        print(f"[{msg['cam']}] WORKER ERROR: {msg['error']}")

    else:
        print("MSG:", msg)

if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    result_q = mp.Queue()

    p1 = mp.Process(target=camera_worker, args=(SERIAL_1, result_q))
    p2 = mp.Process(target=camera_worker, args=(SERIAL_2, result_q))

    print("Starting dual-camera capture in separate processes...")
    p1.start()
    p2.start()

    try:
        while True:
            msg = result_q.get()
            print_msg(msg)
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
