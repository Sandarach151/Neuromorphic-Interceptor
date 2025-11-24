# Event-Based Recording & Trajectory Estimation

**Neuromorphic Camera (Metavision SDK) – Jetson Optimized**

This repository contains a high-performance event-based detection pipeline built for neuromorphic cameras using the Prophesee Metavision SDK.
It performs **real-time downsampled event aggregation**, **region-based start/stop triggers**, **in-memory recording**, and **best-fit trajectory estimation**.

The system is optimized for **NVIDIA Jetson**, but runs on any Linux environment with Metavision installed.

---

## Pipeline

### 1. Event downsampling (100 × 100 grid)

Raw sensor coordinates (up to 640×480) are scaled into a 100×100 grid for fast real-time processing.

### 2. Noise-filtered activity map

Events are aggregated per 10,000 μs slice (configurable).
Any pixel with `count ≥ EVENT_THRESHOLD` is considered active.

### 3. Region-based start/stop triggers

The camera view is divided into two sets of regions:

#### Start Recording (green regions)

Recording begins when any green pixel exceeds the event threshold:

* Columns **0–1**
* Top row (Y=99): **X = 0–49**
* Bottom row (Y=0): **X = 0–49**

#### Stop Recording (red regions)

Recording ends when any red pixel exceeds the event threshold:

* Columns **98–99**
* Top row (Y=99): **X = 50–99**
* Bottom row (Y=0): **X = 50–99**

This makes it ideal for detecting fast-moving objects that enter one side of the frame and exit the other.

### 4. Zero file I/O during recording

All filtered events are stored **in RAM** until recording stops — minimizing latency and disk wear.

### 5. Best-fit trajectory (x(t), y(t))

When recording stops:

* All events are merged into a single `N × 3` array
* Linear least squares is used to estimate the best-fit motion:

  ```
  x(t) = a_x * t + b_x
  y(t) = a_y * t + b_y
  ```
* The trajectory parameters are printed to console

### 6. Automatic CSV export

Each recording is saved as:

```
events/recording_YYYYMMDD_HHMMSS.csv
```

Containing lines of:

```
x_bin, y_bin, timestamp
```

### 7. Fully automatic cycle

After saving, the system resets and waits for the next green trigger.

---

## Installation

### Install Metavision SDK (Prophesee)

Follow the official installation instructions:
[https://docs.prophesee.ai](https://docs.prophesee.ai)

### Install Python dependencies

```
pip install numpy
```

Jetson users should also ensure:

```
sudo apt install python3-numpy
sudo apt install python3-pip
```

---

## Usage

Run the algorithm:

```
python3 algo.py
```

You'll see:

```
Get Ready...
Reading Events...
>>> START RECORDING
...
>>> STOP RECORDING
=== Best-fit trajectories ===
x(t) = ax * t + bx
y(t) = ay * t + by
=============================
Saved recording to events/recording_2025xxxx_xxxxxx.csv
```

The system automatically resets and waits for the next run.

---
