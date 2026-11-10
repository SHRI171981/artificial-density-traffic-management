# Artificial Density Traffic Management

## Problem statement

Fixed-timer traffic signals give every road the same green time whether it is empty or jammed. The obvious fix, always serving the busiest lane, creates a new problem: a busy road can hog the signal while vehicles on quieter roads wait indefinitely.

## Solution

An adaptive controller for a four-way intersection. A YOLOv8 model counts the vehicles waiting in each lane from camera images, and a scheduler picks the lane that gets the green light. To prevent starvation, every lane that is passed over builds up an "artificial density", a waiting weight that is added to its real vehicle count. The lane with the highest total goes next.

Each cycle:

1. Count the vehicles in every lane (A, B, C, D). This is the **physical density**.
2. Add each lane's **artificial density** to get its **effective total**.
3. Give the green light to the lane with the highest effective total.
4. Reset that lane's artificial density to zero. Every other lane's grows by `alpha` times the number of vehicles that just cleared.

Busy lanes still win more often, but a lane that keeps losing accumulates weight until it gets its turn.

## Tech stack

Python, YOLOv8 (Ultralytics), PyTorch, OpenCV, Streamlit, Pandas, NumPy.

## Features

- Vehicle detection with a YOLOv8 model trained on a single-class vehicle dataset (50 epochs, 640 px images), with training, evaluation and prediction helpers
- Starvation-aware scheduling with a tunable `alpha`
- Optional random variation in the counts, so simulations do not repeat
- Streamlit dashboard that simulates 25 signal cycles, showing a traffic light for each lane, its physical, artificial and total density, and its camera image

## Future steps

- Report detection accuracy (mAP) on the vehicle dataset
- Compare average waiting time against a fixed-timer and a busiest-lane-first baseline
- Run on a live camera feed instead of stored images
- Variable green-light durations based on queue length

## Getting started

```bash
pip install -r requirements.txt
```

Create a `.env` file with `YOLO_MODEL_NAME`, place your vehicle dataset at `dataset/VehicleCount/` (with a `data.yaml`), train with `train_yolov8()` from `scripts/yolov8_training.py`, then:

```bash
streamlit run app.py
```

The dashboard loads trained weights from `MODEL_PATH` in `config.py`. Weights and dataset are not included in the repository.
