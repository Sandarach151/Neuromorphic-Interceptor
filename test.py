# test.py — live (t,x,y,p) batches from the first camera
import numpy as np
from metavision_sdk_stream import Camera, CameraStreamSlicer, SliceCondition
import os

cam = Camera.from_first_available()  # open live camera
# pass RValueCamera via .move()
slicer = CameraStreamSlicer(cam.move(), SliceCondition.make_n_us(10_000))

batch = 0
# File to append events to (CSV with header t,x,y,p)
events_file = "events.csv"
# If the file doesn't exist yet we'll write a header once before appending
need_header = not os.path.exists(events_file)
if need_header:
    # create empty file with header
    with open(events_file, 'w') as _f:
        _f.write('t,x,y,p\n')
for sl in slicer:  # no .begin()
    evs = sl.events
    if evs is None or evs.size == 0:
        continue

    # normalize dtype to (t,x,y,p)
    if evs.dtype.names != ('t','x','y','p'):
        evs = np.rec.fromarrays([evs['t'], evs['x'], evs['y'], evs['p']],
                                dtype=[('t','<u8'),('x','<u2'),('y','<u2'),('p','u1')])

    batch += 1
    print(f"[{batch}] {evs.size}")
    try:
        # stack columns into a 2D array for savetxt
        data = np.column_stack((evs['t'].astype(np.uint64),
                                evs['x'].astype(np.uint16),
                                evs['y'].astype(np.uint16),
                                evs['p'].astype(np.uint8)))
        # open in append mode and write rows as integers
        with open(events_file, 'a') as f:
            # np.savetxt accepts a file handle
            np.savetxt(f, data, fmt='%d', delimiter=',')
    except Exception as e:
        # Print error but continue processing further slices
        print(f"Error writing events to {events_file}: {e}")
