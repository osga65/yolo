#!/bin/bash
#To run the virtual environment in your current shell, use:  source activate-yolo.sh
#To leave virtual environment, use:  deactivate

#Delete the Entire Virtual Environment (Cleanest Method). This removes all installed packages and settings:
#rm -rf /home/oscar/Desktop/yolov10-object-tracking/yolo10-env

#Create a Python 3.10 virtual environment
#python3.10 -m venv realsense-env
#source realsense-env/bin/activate

# sudo minicom -D /dev/ttyUSB0 -b 115200
# cat /proc/sys/net/ipv4/tcp_fin_timeout  --> TIME_WAIT=60 s
# ls -l /tmp/my_socket
# scp root@192.168.1.1:/root/client_bidir.c /local/path/
# cc client_bidir.c -o client_bidir -lpthread


SAVE_LOG=0 #0 to save output log; 0 for not

# Set the log file path with timestamp
LOGFILE="SAVED_LOG.txt"
#LOGFILE="$HOME/yolo_run_$(date '+%Y-%m-%d_%H-%M-%S').log"

echo "SAVE_LOG=$SAVE_LOG"

# Activate virtual environment if not already active   
if [[ "$VIRTUAL_ENV" != "$HOME/realsense-env" ]]; then
    echo "[INFO] Activating realsense-env..." | tee -a "$LOGFILE"
    source "$HOME/Desktop/yolov10-object-tracking/realsense-env/bin/activate"
else
    echo "[INFO] Virtual environment already active." | tee -a "$LOGFILE"
fi

# Run the Python script and log output
echo "[INFO] Running YOLO tracking script..." | tee -a "$LOGFILE"
cd "$HOME/Desktop/yolov10-object-tracking  #/YOLO_videos"

if [ $SAVE_LOG -eq 1 ]; then
  echo "[INFO] Running main.py" | tee -a "$LOGFILE"
  python main.py >"$LOGFILE" 2>&1
else
  echo "[INFO] Running main.py" | tee -a "$LOGFILE"
  python main.py 2>&1 #| tee -a "$LOGFILE" #don’t pipe to tee if you want 'q' or OpenCV GUI interaction.
fi

# Deactivate virtual environment
echo "[INFO] Deactivating environment..." >> "$LOGFILE" 2>&1

if [[ "$VIRTUAL_ENV" != "" ]]; then
    deactivate
fi

#The reason it still fails even after sudo apt install python3-shapely is that apt installs Shapely system-wide, but your virtual environment has its own isolated site-packages directory — so the package installed via apt isn’t visible inside your venv.
#para instalar o package xyz no virtual env
#source "$HOME/Desktop/yolov10-object-tracking/realsense-env/bin/activate"
#pip install xyz


