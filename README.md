# YOLOv10 Object Tracking

Object detection and tracking using YOLOv10 and Deep SORT.

## Installation and configuration

### 1. Project files

Place all the project files and folders in the directory yolov10-object-tracking/.

The directory should contain:

```text
yolov10-object-tracking/
├── START-yolov10-virtual-env.sh
├── main.py
├── tracker.py
├── yolov10n.pt
├── deep_sort/
└── model_data/
```

### 2. Create the virtual environment

Inside the project directory, create the RealSense virtual environment:

```bash
python3 -m venv realsense-env
```

Activate it:

```bash
source realsense-env/bin/activate
```

Install the required Python packages and dependencies.

### 3. Configure `main.py`

Before running the program, configure the required parameters at the beginning of `main.py`.

### 4. Run the program

Run the startup script:

```bash
./START-yolov10-virtual-env.sh
```

The script activates the `realsense-env` virtual environment and starts `main.py`.
