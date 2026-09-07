#!/bin/env python

# print("YOLO - Nome:",name, "Modo:", mode)
# /home/oscar/.pyenv/versions/3.8.12/lib/python3.8/site-packages/torch/serialization.py

# https://github.com/osga65/neXt-Sense

# carro 20 km/h percorre c. 0.56 m em 100 ms.

# carro 50 km/h percorre c. 1.4 m / 100 ms

#to extract only the JSON part from each line of a file, i.e., everything starting at { and ending at }
# grep -o '{.*}' SAVED_LOG_125ms_err2.txt > videoLog_only_CPMs.txt 

# grep -vE '🚗|DST|DELTAFRAME|False|True|^$' SAVED_LOG_125ms_err2.txt |  sed -E '/CPM/ s/^[^{]*//' > videoLog_only_CPMs.txt

# Ethernet MTU = 1500 B. Após subtrair os cabeçalhos IP (20) + UDP (8), o payload máximo prático é 1472 B.

# https://gist.github.com/rcland12/dc48e1963268ff98c8b2c4543e7a9be8  # List of all 80 YOLO classes and its index in JSON format.

# ls /dev/tty*;  sudo minicom -D /dev/ttyUSB0

# ffmpeg -ss 00:02:10 -to 00:02:25 -i input.mp4 -qscale:v 2 frames_%04d.jpg  #Extract frames between two times


#marcaçoes na mesa do lab: pontos espaçados 20 cm na horizontal e 20 cm na vertical

import os
import random
import sys
import cv2
from ultralytics import YOLO
from tracker import Tracker
import math
import numpy as np
import json
import time
from collections import deque
import threading
import socket
import select
from multiprocessing import Process, Manager, Event
import time
import csv
import pyproj
from pyproj import Transformer
from pyproj import Proj
import pyrealsense2 as rs
import logging
import signal
from shapely.geometry import Point, Polygon
import itertools
import pytesseract
import re
from sklearn.cluster import DBSCAN
import subprocess

# 🔵 Blue  🟢 Green  🔴 Red  🟣 Magenta  🟡 Yellow  ⚪ White  ⚫ Black
colors = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255), "yellow": (0, 255, 255), "cyan": (255, 255, 0),
    "magenta": (255, 0, 255), "orange": (0, 128, 255), "white": (255, 255, 255), "black": (0, 0, 0), "gray": (128, 128, 128),
    "light_blue": (255, 200, 100), "pink": (203, 192, 255), "purple": (128, 0, 128), "brown": (42, 42, 165), "light_green": (144, 238, 144), "dark_red": (0, 0, 139)}

#-----------------------------------

USE_SAVED_VIDEO = 0
USE_REALSENSE = 1
USE_UVC = 0    #ligar ao usb dir.

USE_CASE = 1  # 1 -- ped collision risk na urbe; 2 -- # ped detection no cruzamento; 3 -- ped collision risk na mesa
# indiferente para USE_SAVED_VIDEO

USE_ONLY_FOR_OBJECT_DETECTION = 1 # 1 para detetar apenas pessoas e carros; 0 para usar os use cases 

CAMERA_POINT = (550079.0, 4601272.0) # UTM do local da camara (poste de sinal de trânsito)
#CAMERA_POINT = (550077.6, 4601271.4) # UTM do local da camara (poste de luz perto do sinal de trânsito)

OUTPUT_AVI_FILE = "SAVED_VIDEO.avi"
LOG_FILE = "SAVED_LOG.txt"

SAVED_VIDEO_FPS = 30

PRINT_OBJ_DISTANCE_TO_CAMERA = 1

VIDEO_FILE_1 = '/home/oscar/Desktop/yolov10-object-tracking/eu.mp4'
VIDEO_FILE_2 = '/home/oscar/Desktop/yolov10-object-tracking/SAVED_VIDEO_125ms_err2.avi'
#VIDEO_FILE_3 = '/home/oscar/Desktop/yolov10-object-tracking/SAVED_VIDEO_X1.avi'
VIDEO_FILE_3 = '/home/oscar/Desktop/yolov10-object-tracking/SAVED_VIDEO_cluster2.avi' #usado para escrever os titulos
VIDEO_FILE = VIDEO_FILE_3  ##################################

VIDEO_TYPE = "2peds2carsVideo" # "2peds1carVideo"

START_FRAME = 0 #2840 # 1480 #400 #1480
NR_FRAMES_TO_DISPLAY = 15 #600 #133 #333 #130




APPLY_LABEL_OFFSET = 0 # 1 para deslocar as labels das boxes, 0 para usar o valor default


PRINT_MY_JSON  = 0
PRINT_PRETTY_MY_JSON = 0
PRINT_PRETTY_PROTO = 0
PRINT_SENT_PROTO = 1
PRINT_UTM_COORD_IN_FRAME = 0
PRINT_AZIMUTH_IN_FRAME = 0
PRINT_LABEL_IN_FRAME = 1

UDP_IP = "192.168.1.124" #"127.0.0.1"
UDP_PORT = 9000

SAMPLING_AND_FLUSH_INTERVAL = 175 #miliseconds  era 125

BUFFER_TTL = 1000 #miliseconds; tempo máx de permanencia da informacao no buffer

HISTORY_LEN = 5

#USE_VIDEO_TIMESTAMP = 1 #0 para tempo real dado por time.time(); 1 para usar o timestamp do video

CALCULATE_STATS = 1
#CALCULATE_ONLY_STATS=0

#CHANGE_COLOR_AT_FRAME_NR = 85

PRINT_PED_ICON = 0

SEND_CPM_EVENTS = 0

SEND_ONLY_CAR_EVENTS = 1 # 1 para incluir na mensagem apenas eventos dos carros, 0 para incluir eventos dos carros e peões

FREEZE_FRAME_NUMBER = 5


NR_MAX_SUCESSIVE_CollisionRisk_EVENTS = 4
NR_MAX_SUCESSIVE_CollisionRiskEnd_EVENTS = 3

SAVE_ALL_FRAMES = 1 #1 to save all frames; 0 to save only tracked frames

FRAME_NR_COLOR_1 = "yellow" #text color at frames with detected tracks
FRAME_NR_COLOR_2 = "cyan" #text color at frames without detected tracks

MSG_ENCODING = "utf-8"

DRAW_PATH = 0 # Draw the line connecting successive center points

USE_NVIDIA_ORIN = 0 # 0 para usar pc

USE_UNIX_SOCKETS=0
UNIX_SOCKET_PATH = "/tmp/my_socket3"

# Configurações do servidor remoto
REMOTE_SERVER_IP = '192.168.1.1' 
REMOTE_SERVER_PORT = 65431

GET_CPM_FROM_FILE = 0

GET_FRAME_NUMBER_FROM_IMAGE = 0

SEE_VIDEO_WITH_MATCHED_FPS = 0

MAX_DISTANCE_TO_CAMERA = 30 #5 # distancia maxima (metros) entre objeto detetado e camara

DETECTION_THRESHOLD = 0.5 #0.2 #0.60 #0.25

MAX_DISTANCE_TO_BELONG_TO_CLUSTER = 0.5 #0.1

DISTANCE_MARGIN_ADD_UP = 0#-0.05 #2

NTP_SERVER_IP = "192.168.1.111"

HIDE_CAR_PLATE = 0

######################################################################################################

# estes são os pontos realmente usados

if VIDEO_FILE == VIDEO_FILE_1 and USE_SAVED_VIDEO: # eu.mp4
  START_FRAME = 1480
  NR_FRAMES_TO_DISPLAY = 130
  IMAGE_X_Y_POINTS = np.array(
    [[387, 435], [999, 294], [870, 347], [223, 186], [208, 144], [413, 131], [402, 121], [32, 110]]
    , dtype=np.float32)
       
elif VIDEO_FILE == VIDEO_FILE_2 and USE_SAVED_VIDEO: # SAVED_VIDEO_125ms_err2.avi 
  START_FRAME = 400
  NR_FRAMES_TO_DISPLAY = 333
  IMAGE_X_Y_POINTS = np.array(
    [[539, 655], [1212, 568], [1064, 616], [442, 381], [437, 358], [623, 359], [529, 343], [328, 352]]
    , dtype=np.float32)

elif VIDEO_FILE == VIDEO_FILE_3 and USE_SAVED_VIDEO: # SAVED_VIDEO_125ms_err2.avi
  if HIDE_CAR_PLATE and VIDEO_TYPE == "2peds2carsVideo":
    START_FRAME = 6110
    NR_FRAMES_TO_DISPLAY = 105
  if HIDE_CAR_PLATE and VIDEO_TYPE == "2peds1carVideo": 
    START_FRAME = 641
    NR_FRAMES_TO_DISPLAY = 59 
  
  IMAGE_X_Y_POINTS = np.array(
    [[1056, 586], [1203, 538], [544, 630], [442, 365], [436, 344], [635, 344], [526, 328], [321, 326]]
    , dtype=np.float32)


else: # real video, copy and paste values here

  IMAGE_X_Y_POINTS = np.array(  

#[[1211, 545], [843, 468], [701, 436], [733, 642], [545, 491], [491, 448]]

[[1056, 586], [1203, 538], [544, 630], [442, 365], [436, 344], [635, 344], [526, 328], [321, 326]]
            
    , dtype=np.float32)



#######################################################################################

BOX_LINE_THICKNESS = 2

DRAW_OBJECT_BOXES = 1 # 1 para desenhar boxes nos objectos (peões, carros), 0 para não

DRAW_CLUSTER_BOX = 0 # 1 para desenhar boxes nos clusters, 0 para não
    
ONE_MEMBER_CLUSTER_ID = -1 # clusterID for pedestrians that do not belong to a cluster

CLUSTER_FONT_COLOR = colors["yellow"]

NO_RISK_CLUSTER_BOX_COLOR = colors["yellow"]

RISK_CLUSTER_BOX_COLOR = colors["red"]

NO_RISK_BOX_COLOR = colors["green"]

RISK_BOX_COLOR = colors["red"]

CIRCLE_COLOR = colors["green"]

BOX_BLUR_COLOR = colors["gray"]

STRING1 = "@DELTAFRAME"

SENT_WGS84_IN_UDP_JSON = 1 # 0 para enviar coords UTM no json txed por udp, 1 para enviar WGS84 
# todos os calculos são feitos em utm. Só se faz a conversão para wgs84 no momento antes de se enviar a mensagem udp.

USE_PREDEFINED_COORDS = 0 # 1 para usar lat e lon predefinidos, 0 para usar os reais.

FLUSH_INTERVAL = SAMPLING_AND_FLUSH_INTERVAL

SAMPLING_INTERVAL = SAMPLING_AND_FLUSH_INTERVAL

GET_POINTS_FROM_FRAME = 1 # 1 para obter pontos (x,y) duma frame com o cursor

RESTART_AFTER_END = 0



AZIMUTH_UNAVAILABLE = 3601  #ver p.38 https://www.etsi.org/deliver/etsi_ts/102800_102899/10289402/01.03.01_60/ts_10289402v010301p.pdf
SPEED_UNAVAILABLE = 6789  # 16383 cf. p.57
USE_ZERO_SPEED = 0

        
        

if USE_SAVED_VIDEO: 
   print("\n########## USING SAVED VIDEO ##########\n")
   USE_CASE = 1
   USE_VIDEO_TIMESTAMP = 1
   GET_POINTS_FROM_FRAME = 0
   USE_PREDEFINED_COORDS = 0
if USE_REALSENSE: 
   print("\n########## USING REALSENSE ##########\n")
   USE_VIDEO_TIMESTAMP = 0
   GET_CPM_FROM_FILE = 0
if USE_UVC: 
   print("\n########## USING UVC ##########\n")
   USE_VIDEO_TIMESTAMP = 0
   GET_CPM_FROM_FILE = 0



if USE_SAVED_VIDEO + USE_REALSENSE + USE_UVC != 1: 
    print("\nERROR: Choose only one: USE_SAVED_VIDEO or USE_REALSENSE or USE_UVC !!!!!")
    sys.exit(0)



# Homografia baseada em pontos (x,y) conhecidos na imagem e correspondentes coordenadas UTM 
# Usar pelo menos 4 pontos

if USE_CASE == 1: #"ped collision risk"

  image_points = np.array([

  #### coordenadas dos pontos da imagem Oscar ####

    [387, 435],   # ponto 1 (grelha de drenagem esq., canto mais afastado)
    [999, 294],   # ponto 2 (banco no passeio, canto esq.)
    #[852, 458],   # ponto 3 (tampa de águas pluviais)
    [870, 347],   # ponto 3a (grelha de drenagem dir., canto mais afastado)
    [223, 186],   # ponto 4 (poste de sinal de trânsito)
    [208, 144],   # ponto 5 (poste de luz)
    [413, 131],   # ponto 6 (estrutura da tampa no fim da berma dir.)
    [402, 121],   # ponto 7 (marco castanho em frente da escola de direiro)
    [32, 110]     # ponto 8 (poste do ponto de encontro)
    

  ], dtype=np.float32)
  utm_points = np.array([

  #### coordenadas UTM (Este, Norte) ####   dst to camera
    [550083.2, 4601272.9],  # ponto 1 --- 4.0 m
    [550086.3, 4601265.8],  # ponto 2 --- 9.6 m
    #[550082.7, 4601269.5],  # ponto 3 --- 4.5 m
    [550085.0, 4601268.0],  # ponto 3a --- 4.5 m
    [550096.6, 4601279.2],  # ponto 4 --- 18.9 m
    [550101.0, 4601280.8],  # ponto 5 --- 23.5 m
    [550113.6, 4601275.4],  # ponto 6 --- 34.7 m 
    [550131.4, 4601285.8],  # ponto 7 --- 67.3 m
    [550103.5, 4601286.6]   # ponto 8 --- 28.6 m

  ], dtype=np.float32)


elif USE_CASE == 2: #"peds at cross area"

  image_points = np.array([

  #### coordenadas dos pontos da imagem SAMUEL ####

    [374, 458],   # ponto 11 (tampo das águas pluviais no passeio do cruzamento)
    [999, 294],   # ponto 12 (primeiro poste de luz)
    [852, 458],   # ponto 13 (grelha de drenagem no passeio esq. do cruzamento, perto do mapa)
    [223, 186],   # ponto 14 (banco mais próximo no passeio esq. (vértice mais afastado)
    [208, 144],   # ponto 15 (tampo das águas pluviais no meio da estrada, o mais afastado)
    [413, 131],   # ponto 16 (grelha de drenagem no passeio dir.)
    [402, 121],   # ponto 17 (grelha de drenagem no passeio esq, oposta à do passeio dir.)
    [32, 110],     # ponto 18 (banco mais distante no passeio esq., vértice mais afastado))
    [99, 99]      # ponto 19 (balde do lixo no fundo do passeio esq.)

  ], dtype=np.float32)
  utm_points = np.array([

  #### coordenadas UTM (Este, Norte) ####   dst to camera
    [550058.8, 4601269.8],  # ponto 11 --- 0.0 m
    [550054.3, 4601262.2],  # ponto 12 --- 0.0 m
    [550066.0, 4601260.3],  # ponto 13 --- 0.0 m
    [550058.5, 4601254.7],  # ponto 14 --- 0.0 m
    [550044.9, 4601255.3],  # ponto 15 --- 0.0 m
    [550048.2, 4601258.9],  # ponto 16 --- 0.0 m 
    [550050.3, 4601254.0],  # ponto 17 --- 0.0 m
    [550032.3, 4601244.2],   # ponto 18 --- 0.0 m
    [550023.8, 4601240.7]   # ponto 19 --- 0.0 m

  ], dtype=np.float32)


elif USE_CASE == 3: # lab table

  CAMERA_POINT = (0, 20) # UTM do local da camara (local da camara na mesa)
  MAX_DISTANCE_TO_CAMERA = 120
  MAX_DISTANCE_TO_BELONG_TO_CLUSTER = 5

  image_points = np.array([

  #### coordenadas dos pontos da imagem MESA ####


    [920, 464],   # ponto 1 dir.
    [782, 366],   # ponto 2 dir.
    [722, 329],   # ponto 3 dir.
    [293, 429],   # ponto 1 esq.
    [470, 343],   # ponto 2 esq.
    [530, 317]    # ponto 3 esq.

  ], dtype=np.float32)
  utm_points = np.array([

  #### coordenadas UTM (Este, Norte) ####   dst to camera
    [0, 40],   # ponto 1 dir.
    [0, 80],   # ponto 2 dir.
    [0, 120],   # ponto 3 dir.
    [40, 40],   # ponto 1 esq.
    [40, 80],   # ponto 2 esq.
    [40, 120]    # ponto 3 esq.

  ], dtype=np.float32)
  
  
else: 
 print("\nERROR: Unkown use case !!!")
 exit(0)




# Define area as polygon in UTM coordinates
AREA_POLYGON_UTM = [
    (550058.8, 4601269.8), #ponto 11
    (550054.3, 4601262.2), #ponto 12
    (550058.5, 4601254.7), #ponto 14
    (550066.0, 4601260.3)  #ponto 13
]



if GET_CPM_FROM_FILE:
     PRINT_SENT_PROTO = 0
     PRINT_OBJ_DISTANCE_TO_CAMERA = 0
     VIDEO_FILE='/home/oscar/Desktop/yolov10-object-tracking/SAVED_VIDEO_125ms_err2.avi'
     INPUT_LOG_FILE='/home/oscar/Desktop/yolov10-object-tracking/videoLog_only_CPMs.txt'

     image_points = np.array([
       [ 539,  659],
       [1214,  568],
       [1060,  616],
       [ 447,  377],
       [ 437,  362],
       [ 623,  361],
       [ 528,  345],
       [ 326,  341]
       ], dtype=np.float32)









NR_TOT_POINTS = len(utm_points)
print("NR_TOT_POINTS",NR_TOT_POINTS)



# List of all 80 YOLO classes: https://gist.github.com/rcland12/dc48e1963268ff98c8b2c4543e7a9be8
pedClass=0; biciClass=1; carClass=2; motoClass=3

#print() goes both to screen and file
logging.basicConfig(
    level=logging.INFO,
    #format="%(asctime)s [%(levelname)s] %(message)s", #tells logger how to display each log line. e.g. 2025-10-02 14:55:12 [INFO] YOLO model loaded
    format="%(message)s",   # only prints the log message itself
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("SAVED_LOG.txt", "w")
    ]
)

# to override print, wrap it safely so flush doesn’t break it
def safe_print(*args, **kwargs):
    # strip flush=True if passed
    kwargs.pop("flush", None)
    logging.info(" ".join(map(str, args)))

print = safe_print

udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Create UDP socket



# Calcular matriz de homografia e a sua inversa
H, _ = cv2.findHomography(image_points, utm_points)
H_inv = np.linalg.inv(H)  #it’s faster to precompute the inverse once

#function that maps from image coordinates → UTM using a homography matrix H.
# Função para converter centro da pessoa para coordenadas UTM usando homografia
def convert_to_utm_from_homography(image_point, H):
    #print("Point", image_point)
    #print("Matriz", H)
    point = np.array([image_point[0], image_point[1], 1.0]).reshape((3, 1))
    utm_point = H @ point
    utm_point /= utm_point[2]
    return (utm_point[0][0], utm_point[1][0])


def get_ntp_info():
    try:  
        result = subprocess.check_output(["ntpdate", "-q", NTP_SERVER_IP], text=True)
        print("[ NTP Info ]:", " ".join(result.split()[:6]))
        print("if time offset not ok, run: sudo systemctl restart systemd-timesyncd")
    except Exception:
        print("[ERROR] Failed to get NTP data.")
    print()   


def ask_yes_no(prompt):
    """
    Ask the user a yes/no question from the keyboard and return True/False.

    Args:
        prompt (str): The question to display to the user.

    Returns:
        bool: True if the user answered 'y' or 'yes', False otherwise.
    """

              
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("y", "yes"):
            return True
        elif answer in ("n", "no"):
            return False
        else:
            print("Please type 'y' or 'n'.")

def start_unix_socket_server(socket_path):
    # Remove existing socket
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass

    server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_socket.bind(socket_path)
    server_socket.listen(1)
    print(f"[UNIX SOCKET] Listening on {socket_path}...")



def cpm_time_stats(filepath):

    
    print("\nStatistics of the time interval (ms) between consecutive CPMs:", flush=True)

    times = []
    modulo = 65536
    pattern = re.compile(r'\bCPM tx\b')  # detect CPM tx lines

    with open(filepath, 'r') as f:
        for line in f:
            if pattern.search(line):
                parts = line.strip().split()
                if len(parts) > 2:
                    try:
                        t = int(parts[1])  # the 'time' field (2nd column)
                        times.append(t)
                    except ValueError:
                        continue

    # Need at least 2 timestamps to compute deltas
    if len(times) < 2:
        print("Not enough CPM messages found.")
        return None

    # Skip the first timestamp explicitly
    deltas = []
    for i in range(1, len(times)):
        dt = (times[i] - times[i - 1]) % modulo
        if dt < 5000: deltas.append(dt)

    if not deltas:
        print("No valid intervals found after skipping first CPM message.")
        return None

    # Compute statistics
    avg_dt = np.mean(deltas)
    max_dt = np.max(deltas)
    min_dt = np.min(deltas)
    size = len(deltas)

    print(f"Number of CPM messages: {len(times)}")
    print(f"Intervals computed: {size}")
    print(f"min: {round(min_dt,2)}, AVG: {round(avg_dt,2)}, max: {round(max_dt,2)}") #Δt


    return {
        "count": size,
        "avg": avg_dt,
        "min": min_dt,
        "max": max_dt
    }



def process_log_file(filepath):
    """
    Read a log file, extract values from lines starting with a string, and compute min, max, and average.
    Returns the dict: statistics (min, max, avg, count)
    """
    print("\nStatistics of the time interval (ms) between consecutive frames:", flush=True)

    values = []
    first_F_skipped = False
    
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith(STRING1):
                if not first_F_skipped:
                    # skip the first line with the special string
                    first_F_skipped = True
                    continue
                try:
                    value = float(line.split()[1])
                    if value < 5: values.append(value)
                except (IndexError, ValueError):
                    continue

    if not values:
        print({"min": None, "max": None, "avg": None, "count": 0})
        return {"min": None, "max": None, "avg": None, "count": 0}

        
    print(f"Deltaframes computed: {len(values)}")
    print (f"min: {round(min(values)*1000,2)} AVG: {round(sum(values)*1000/len(values),2)}, max: {round(max(values)*1000,2)}")



#function that saves in a FILE only the objects detected in the last N seconds, and refreshes the output file every N seconds,

#function using the multiprocessing.Manager().dict() to store and update object detection data in RAM, with one latest record per ID, updated every N seconds.




def box_center(x1, y1, x2, y2):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy


def getClassId(x, y, bboxes_class, tolerance=6, default=None):
    """
    Find the class_id for (x, y) by matching to stored centers within a tolerance.

    Args:
        x (int): x-coordinate to check
        y (int): y-coordinate to check
        bboxes_class (dict): mapping (xc, yc) -> class_id
        tolerance (int): maximum distance allowed for a match
        default: value to return if no match found

    Returns:
        int or None: class_id if a match is found, else default
    """
    for (xc, yc), class_id in bboxes_class.items():
        if abs(x - xc) <= tolerance and abs(y - yc) <= tolerance:
            return class_id
    return default




def nowTime_ms():
    """Return current time in milliseconds."""
    #print("TTTTTTTT", time.time())
    return int(time.time() * 1000)

def nowTime_ms_mod():
    """Return current time in milliseconds."""
    #print("TTTTTTTT", time.time())
    return int(time.time() * 1000) % 65536

    

#remove entries older than ttl ms from the current time
def clean_buffer(buf, ttl):
    """Remove entries older than ttl (ms) from the buffer."""
    now = nowTime_ms_mod()

    while buf and (now - buf[0]['time']) % 65536 >= ttl:
        old = buf.popleft()
        print(f"Removed old entry: id={old['id']} age={(now - old['time']) % 65536} ms")


def get_first_number(line: str):
    """
    Returns the first number in the line if it starts with digits.
    Returns None if the first token is not numeric.
    """
    if not line:
        return None
    line = line.strip()
    if not line:
        return None

    first_token = line.split()[0]
    return int(first_token) if first_token.isdigit() else None


nowTime0 = nowTime_ms_mod()
firstTime=True
timeMS=0
def run_tracking_and_share_recent_and_show_video_utm(yolo_model_path, shared_dict, detection_threshold=0.5):
    
    detection_threshold = DETECTION_THRESHOLD
    flush_interval = FLUSH_INTERVAL #/1000.0
    sampling_interval = SAMPLING_INTERVAL/1000.0

    global H
    global firstTime
    global timeMS
    global image_points
    
#Variables survives outside the if block because blocks don’t define scope in Python.
#The only scope boundaries are: Functions (def, lambda); Classes; Modules

    static=run_tracking_and_share_recent_and_show_video_utm
    
    if not hasattr(static, "clusters_set"): static.clusters_set = []  # initialize once


    if USE_REALSENSE:
         
         cameraId = "camera1" #"videocam1"
         
         print("OpenCV webcam setup")
         ctx = rs.context()
         devices = ctx.query_devices()
         if not devices:
          print("❌ No RealSense devices detected.")
          exit()
         else:
          print("✅ RealSense device(s) detected:")
          #for dev in devices:
          #         print(f" - {dev.get_info(rs.camera_info.name)}")
   
         # Declare pipeline and config
         pipeline = rs.pipeline()
         config = rs.config()
   
         # Enable device by serial number
         #device_serial = '141322251218'
        #config.enable_device(device_serial)
   
        # Enable both depth and color streams
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
####
        # Only enable color stream
        #config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # Use wrapper to resolve device conflicts
        #pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        #pipeline_profile = config.resolve(pipeline_wrapper)
######

   # Get the list of available devices and their sensors
# Initialize context and pipeline

         ctx = rs.context()
         devices = ctx.query_devices()
         if not devices:
          raise RuntimeError("No RealSense device found")

         serial = devices[0].get_info(rs.camera_info.serial_number)
         print("Device Serial Nr. =", serial)
         config.enable_device(serial)
         
         # Get the first connected device
         device = ctx.query_devices()[0]
         
         # List all supported profiles for the color sensor
         if 0:
           color_sensor = device.query_sensors()[1]  # [0]=depth, [1]=color (usually)
           for s in color_sensor.get_stream_profiles():
               v = s.as_video_stream_profile()
               print(f"Resolution: {v.width()}x{v.height()} @ {v.fps()}fps, Format: {v.format()}")

         width, height = 1280, 800 #640, 480
        
         # Properly configure the RGB stream
         config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30) #1280×720 @ 30 fps
         

         # Start pipeline
         profile=pipeline.start(config)
         print("Pipeline started with RGB stream")

         
         
         
         # Get the actual stream FPS
         color_stream = profile.get_stream(rs.stream.color)
         fps = 5 #color_stream.as_video_stream_profile().fps()
         print("RealSense camera FPS:", fps)

         out = cv2.VideoWriter(OUTPUT_AVI_FILE, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))

         # Grab one test frame to confirm
         frames = pipeline.wait_for_frames()
         color_frame = frames.get_color_frame()
         if not color_frame:
            print("❌ Could not access the RealSense camera.")
            sys.exit(0)
         frame = np.asanyarray(color_frame.get_data())
         # Convert to grayscale for faster processing and visualization
         #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

         ret = True
   
         #video_out_path = os.path.join('.', 'out.mp4')




    if USE_UVC:
         cameraId = "camera2" #"videocam2"
         # ✅ OpenCV webcam setup
         print("✅ OpenCV uvc webcam setup")
         
         if 0:
             for i in range(10):
               cap = cv2.VideoCapture(i)
               if cap.isOpened():
                  print(f"Camera found at index {i}")   
             cap.release(); exit()
         
         cap = cv2.VideoCapture(4)  # 0 is the laptop camera (2 is for laptop gray image); 
         # ls /dev/video*  --> /dev/video0  /dev/video1  /dev/video2 : each corresponds to a possible index (0, 1, 2…)
         
         # Set resolution and fps
         width, height, fps = 640, 480, 5 #supported modes
         cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
         cap.set(cv2.CAP_PROP_FPS, fps)
         
         #print("#######",cv2.getBuildInformation())
 
      
         # Get video properties
         fps = int(cap.get(cv2.CAP_PROP_FPS))
         #width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
         #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
         #fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
         #codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
         #print(f"Resolution: {int(width)}x{int(height)} @ {fps:.2f} fps, Codec: {codec}")
         
         print("###### FPS =", fps, flush=True)
         
         out = cv2.VideoWriter(OUTPUT_AVI_FILE, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))

         ret, frame = cap.read()
         if not ret:
          print("❌ Could not access the webcam.")
          print("Hint: check settings of USE_UVC and USE_REALSENSE\n")
          return

    
    global VIDEO_FILE

    
    if USE_SAVED_VIDEO:
        cameraId = "camera999" #"videocam999"
        logging.getLogger('ultralytics').setLevel(logging.ERROR)
        video_path = os.path.join('.', 'data', VIDEO_FILE)

        if GET_CPM_FROM_FILE:  
#When you assign to a variable inside a function, Python automatically treats it as a local variable within that function, even if there’s a global variable with the same name.
             video_path = VIDEO_FILE
             cpm_reader = read_cpm_from_file(INPUT_LOG_FILE)


        print("Video path:", video_path)
        if not os.path.exists(video_path): 
           print("❌ File not found!")
           sys.exit(0)
        
        #print("Log path:", INPUT_LOG_FILE)
        #if not os.path.exists(INPUT_LOG_FILE): 
        #   print("❌ File not found!")
        #   sys.exit(0)
    
        # Load the video
        cap = cv2.VideoCapture(video_path)
    
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        # Define the codec and create VideoWriter object
        out = cv2.VideoWriter(OUTPUT_AVI_FILE, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))

        

    
        red_flagged_cars=[]
        red_flagged_peds=[]
    
        # Jump to start_frame or start_time if provided
        start_time = None
        start_frame = START_FRAME
        if start_time is not None:
             cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        else:
             cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        ret, frame = cap.read()
        
        print("###### FPS=", fps, "width=", frame.shape[1], "height=", frame.shape[0], flush=True)
        

        if not ret:
             print("Error: Cannot read video.")
             sys.exit(0)

    


    #colours = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

    model = YOLO(yolo_model_path)
    tracker = Tracker()

    frame_rate = SAVED_VIDEO_FPS
    frame_idx = 0

    bufer = deque() #double-ended queue, provides fast appends and pops from both ends. It is a deque with multiple tracked objects
    cpm = deque() 
    object_buffers = {} # Dictionary to hold a buffer per object ID
         
    last_flush_time = time.time() # seconds since epoch
    last_flush_timestamp = 0.0  # Usar o timestamp do vídeo
         
    # Initialize storage for track_id → class_id mapping
    class_ids = {}
    frame_count=0
         
    red_flagged_cars=[]

    print("[INFO] Starting video processing with visualization... Press 'q' to quit.")

    north_frame = -50.8  # degrees
    north_point1 = (230, 290)
    north_point2 = (430, 45)
    north_point_arrow1 = (180, 110)
    north_point_arrow2 = (north_point2[0] - 200, north_point2[1])

    time0 = nowTime_ms() #time.time()
    lasttime=time.time()
    
    # Catch CTRL+C
    #signal.signal(signal.SIGINT, cleanup)
    
    if GET_POINTS_FROM_FRAME:
          #os.system('clear')
          get_ntp_info()
          if RESTART_AFTER_END: 
            img_already_georeferenced = 1
          else: 
            img_already_georeferenced = ask_yes_no("Image is already georeferenced (y/n)?  ")
          if img_already_georeferenced: 
                   image_points = IMAGE_X_Y_POINTS
                   print(); print("Used video:"); print(VIDEO_FILE)
                   print(); print("Used image points:"); print(image_points)
                   time.sleep(2)
          

#######################################

    try:

     while ret:
       #t=0.0
       #if frame_count>423: t=0.5
       #time.sleep(t)
       frame_count += 1
       frame_already_saved=0
        
       if USE_SAVED_VIDEO and frame_count >= NR_FRAMES_TO_DISPLAY:
            print("Reached last frame, stopping...")
            break
            
       print(); print((nowTime_ms_mod()- nowTime0)%65536, "frame", frame_count)
        
       if GET_CPM_FROM_FILE:
               
               line = next(cpm_reader)
               print(); print("FROM FILE:", line, flush=True)
               
               line = line.strip()
               num = get_first_number(line)
               if num is not None: timeMS = num

               if firstTime: 
                     firstTime = False
                     last_timeMS = timeMS

               
               sampling_condition = (timeMS - last_timeMS)%65536 >= SAMPLING_INTERVAL

               if sampling_condition:
                  nextLineStartsWithBrace, cpm_reader = next_line_starts_with_brace(cpm_reader)
                  print(); print("SAMPLE")
                  if nextLineStartsWithBrace:
                    line = next(cpm_reader)
                    cpm_dict = json.loads(line)
                    print(); print("FROM FILE:", cpm_dict["timestamp"], cpm_dict, flush=True)
                    last_timeMS = timeMS                          
                    red_flagged_cars, red_flagged_peds = find_red_flagged_objs(cpm_dict)
               
               
               
               
               timestamp = frame_idx / frame_rate *1.0  #seconds
               detectionResults = model(frame, verbose=False) #verbose=0 to remove YOLO’s internal logging messages
               detections = []
               extra_info = []   # to keep score and class_id    
               bboxes_class = {}  # mapping bbox tuple -> class_id    
               num_detected_objects_above_threshold = 0
               
        # --- Collect detections ---
               for result in detectionResults: #for detection in detections:   
                if result.boxes is None: continue
                num_boxes = len(result.boxes.data)
                for r in result.boxes.data.tolist():
                 x1, y1, x2, y2, score, class_id = r #score = 0.92; class_id = 0: means that 92% confident that the box contains a person
                 
                 if score > detection_threshold: num_detected_objects_above_threshold += 1  # ← count detection
                 if score > detection_threshold and int(class_id) in [0, 2]:  # person or car
                    # save class_id here
                    #detections.append([int(x1), int(y1), int(x2), int(y2), float(score), int(class_id)])
                    detections.append([int(x1), int(y1), int(x2), int(y2), float(score)])
                    #bbox = [int(x1), int(y1), int(x2), int(y2)]
                    xc,yc=box_center(x1, y1, x2, y2)
                    #detections.append(bbox)
                    bboxes_class[(int(xc),int(yc))] = int(class_id)
                    #class_ids[int(track.track_id)] = int(class_id)   # keep a dict of track_id → class_id
                    
                    # keep class_id mapping separately
                    #extra_info.append(int(class_id))
                    x1,y1 = box_center(x1, y1, x2, y2)
                    #print(class_id, "box_center", x1,y1)                  
               print("### Nr. detected objects=", num_boxes, ": Nr. detected objects above threshold=", num_detected_objects_above_threshold, ": Frame=", frame_count)
               tracker.update(frame, detections)
               
     

               ret, frame = cap.read()
               cv2.imshow('YOLO Tracking', frame) # color image
               delay = int(1000 / SAVED_VIDEO_FPS)  if SEE_VIDEO_WITH_MATCHED_FPS else 1
               if cv2.waitKey(delay) & 0xFF == ord('q'): break  #required to display each frame properly. Window refreshes every delay ms at maximum
              







       if not GET_CPM_FROM_FILE:
                    
        timestamp = frame_idx / frame_rate *1.0  #seconds
        
        detectionResults = model(frame, conf=detection_threshold, verbose=False) #verbose=0 to remove YOLO’s internal logging messages
        # lista de detectionResults contem apenas as deteccoes com score > detection_threshold
        
        #print("LLLLL",len(detectionResults), flush=True)
        detections = []
        extra_info = []   # to keep score and class_id    
        bboxes_class = {}  # mapping bbox tuple -> class_id    
        num_detected_objects_above_threshold = 0
        
        # --- Collect detections ---
        for result in detectionResults: #for detection in detections:
            if result.boxes is None: continue        
            for r in result.boxes.data.tolist():
                num_boxes = len(result.boxes.data)
                x1, y1, x2, y2, score, class_id = r #score = 0.92; class_id = 0: means that 92% confident that the box contains a person
                if score > detection_threshold: num_detected_objects_above_threshold += 1  # ← count detection
                #print("SSSSSS", int(class_id), score, flush=True)
                #if score > detection_threshold and int(class_id) in [0, 2]:  # person or car
                if int(class_id) in [0, 2]:  # person or car
                    # save class_id here
                    #detections.append([int(x1), int(y1), int(x2), int(y2), float(score), int(class_id)])
                    detections.append([int(x1), int(y1), int(x2), int(y2), float(score)])
                    #bbox = [int(x1), int(y1), int(x2), int(y2)]
                    xc,yc=box_center(x1, y1, x2, y2)
                    #detections.append(bbox)
                    bboxes_class[(int(xc),int(yc))] = int(class_id)
                    #class_ids[int(track.track_id)] = int(class_id)   # keep a dict of track_id → class_id
                    
                    # keep class_id mapping separately
                    #extra_info.append(int(class_id))
                    x1,y1=box_center(x1, y1, x2, y2)
                    #print(class_id, "box_center", x1,y1)                  
        print("###### Nr. detected objects=", num_boxes, ": Nr. detected objects above threshold=", num_detected_objects_above_threshold, ": Frame=", frame_count)
        tracker.update(frame, detections)
        



        # --- Timestamp encoding ---
        #timestamp_ms = int(round(timestamp * 1000))  # convert to ms
        generationModTime = nowTime_ms_mod() # it wraps every ~65536 seconds). It is relative to the station’s synchronized GNSS time.
        #print("TTTTTT", timestamp, time.time())
        cpm.append({
                    "version": 1.0,
                    "timestamp": generationModTime
                }) # //aaa
        
        
        if 1 and HIDE_CAR_PLATE:
                 color="gray"
                 if VIDEO_TYPE == "2peds2carsVideo":
                    x,y = get_x_y(frame_count, "2peds2carsVideo")
                    #print("GGGGG",frame_count, x, y)            
                    if frame_count>15 and frame_count <  40:
                      cv2.rectangle(frame, (x-20, y), (x+50, y-25), colors[color], -1)
                      
                    if frame_count >=  40 :#and frame_count < 60:
                      cv2.rectangle(frame, (x-20, y), (x+100, y-25), colors[color], -1)
                       
                 if VIDEO_TYPE == "2peds1carVideo":
                    x,y = get_x_y(frame_count, "2peds1carVideo")
                    #print("GGGGG",frame_count, x, y)     
                    if frame_count>=24 and frame_count <=  31:
                      cv2.rectangle(frame, (x-20, y), (x+50, y-50), colors[color], -1)                           
                    if frame_count>=32 and frame_count <=  36:
                      cv2.rectangle(frame, (x-20, y), (x+50, y-25), colors[color], -1)
                    if frame_count ==37:
                      cv2.rectangle(frame, (x-20, y), (x+100, y+20), colors[color], -1)                     
                    if frame_count >= 43 and frame_count <= 44:
                      cv2.rectangle(frame, (x-40, y), (x+80, y-50), colors[color], -1)                        
                    if frame_count>=45 and frame_count <=  46:
                      cv2.rectangle(frame, (x-20, y), (x+100, y-50), colors[color], -1)                      

        frame_offset=0
        if USE_SAVED_VIDEO: frame_offset = 20 #para evitar sobreposição do frame nr
        
        text="ONE PERSON PASSED BY A CAR"
        #cv2.putText(frame, text, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, colors["yellow"], 8)
        text="TWO PERSONS PASSED BY A CAR IN SUCCESSION"
        #cv2.putText(frame, text, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors["yellow"], 6)
        text="FORMATION AND BREAKUP OF A 4-MEMBER CLUSTER"
        #cv2.putText(frame, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors["yellow"], 6)
        


        # Draw detections
        for track in tracker.tracks:
          #if (track.track_id==4 or track.track_id==8 or track.track_id==6):
            if (track.track_id is not None and track.center is not None) and (hasattr(track, 'bbox') and track.bbox is not None):
                #print("YYYYY", track.track_id)
                # Create a new deque for this object if it doesn't exist yet
                obj_id = track.track_id
               # if obj_id not in object_buffers:
               #     object_buffers[obj_id] = deque(maxlen=30)  # keep last 30 entries for this object


                x, y = int(track.center[0]), int(track.center[1])
                
                x_center_utm, y_center_utm = convert_to_utm_from_homography((x,y), H)
                
                distance2camera = get_distance_to_camera(CAMERA_POINT, x_center_utm, y_center_utm)
                
                #print("UUUUUUUUUU id, dist, x, y, x_center_utm, y_center_utm:", obj_id, distance2camera,  x, y, x_center_utm, y_center_utm, flush=True)
                
                if distance2camera > MAX_DISTANCE_TO_CAMERA: continue
                
                x1, y1, x2, y2 = map(int, track.bbox)
                
                track.class_id=getClassId(x,y, bboxes_class, 6)
                
                
                # --- Store class_id persistently ---
                # track keeps its last known class ID automatically (avoiding None)
                if track.class_id is not None:
                      class_ids[track.track_id] = track.class_id
                else:
                    # Restore last known class if available
                    if track.track_id in class_ids:
                            track.class_id = class_ids[track.track_id]
                
                
                
                
                
                
                #print(track.track_id, "track_box_center", track.center[0], track.center[1])

                # Draw bounding box
                #cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                #if track.track_id in [4, 8]: cv2.rectangle(frame, (x1, y1), (x2, y2), colors["green"], 2)
                #if track.track_id==6: 
                #    if frame_count < CHANGE_COLOR_AT_FRAME_NR: cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                #    else: cv2.rectangle(frame, (x1, y1), (x2, y2), colors["green"], 2)
                #cv2.putText(frame, f'ID {track.track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["green"], 2)
                
                vel = int(track.speed*10)/10.0             
                #cv2.putText(frame, f"{track_id}#{vel if abs(vel) < 6.0 else 6.0*int(vel)/abs(int(vel))}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_PLAIN, 0.9, colors["green"], 1)
                #cv2.putText(frame, f"{track.track_id}#{int(track.mavg_speed*10)/10}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, colors["green"], 1)
                
               


                # Label with ID and class
                #cls_id = getattr(track, "class_id", -1) #gets the value of the attribute class_id of the object track
                cls_id=track.class_id #class_ids[track.track_id]
                cls_id = track.class_id if track.class_id is not None else class_ids.get(track.track_id, -1)
                offset = 35 if APPLY_LABEL_OFFSET else 10
                #print("CCCCCCCC", track.track_id, cls_id)
                if cls_id == pedClass: label = f"P{track.track_id}"
                elif cls_id == carClass: label = f"C{track.track_id}"
                else: label = f"X{track.track_id}"
                if PRINT_LABEL_IN_FRAME:
                  cv2.putText(frame, label, (x1, y1 - offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["green"], 1)

                # Draw center point at box center    
                #center_coordinates = (x, y)
                #cv2.circle(frame, center_coordinates, radius=3, color=colors["green"], thickness=-1) # Green point at center
                
                # Draw center point at box bottom line
                x=x1+int((x2-x1)/2); y=y2       
                center_coordinates = (x, y)
                cv2.circle(frame, center_coordinates, radius=3, color=CIRCLE_COLOR, thickness=-1) # Green point at bottom line
                #print("FFFFF",frame_count,x,y)  
                
                if HIDE_CAR_PLATE and cls_id == carClass:
                  if VIDEO_TYPE == "2peds2carsVideo":
                      x_offset=150; y_offset=20
                      if frame_count >= 32 and frame_count <=44:
                         x_offset=50; y_offset=20
                      if frame_count >= 47 and frame_count <=48:
                        x_offset=250; y_offset=-40
                      if frame_count >= 57 and frame_count <=80:
                         x_offset=25; y_offset=20
                      if frame_count > 82:
                        x_offset=250; y_offset=20                   

                  if VIDEO_TYPE == "2peds1carVideo":
                    x_offset=100; y_offset=20
                    if frame_count >= 41 and frame_count < 43:
                         x_offset=150; y_offset=60                    
                    if frame_count >= 44 and frame_count <= 45:
                         x_offset=100; y_offset=60                     

                  cv2.rectangle(frame, (x-25, y), (x+x_offset, y-y_offset), BOX_BLUR_COLOR, -1)
                  
                  
                  
                  
                #print("TTTTTT", track.track_id, center_coordinates)

                # Extra: If it's a person (class_id == 0), compute UTM
                #if hasattr(track, "class_id") and track.class_id == 0 and H is not None:
                if len(track.path) > 2: #class_ids.get(track.track_id, None) == 0 and H is not None:
                    #print(center_coordinates, "###")
                    utm_coords = convert_to_utm_from_homography(center_coordinates, H)
                    utm_coords_print = tuple(round(float(v), 2) for v in utm_coords) #para evitar os prints de np.float64()
                    print(f"Track {track.track_id} (x,y)={center_coordinates} UTM: {utm_coords_print}")
                    offset = 25 if APPLY_LABEL_OFFSET else 10
                    if PRINT_UTM_COORD_IN_FRAME:
                       cv2.putText(frame, f"{int(utm_coords[0])},{int(utm_coords[1])}", (x + 10, y-offset), cv2.FONT_HERSHEY_PLAIN, 0.7, colors["green"], 1)
                                
                if PRINT_OBJ_DISTANCE_TO_CAMERA:
                  x_center_utm, y_center_utm = convert_to_utm_from_homography((x,y), H)
                  dist = get_distance_to_camera(CAMERA_POINT, x_center_utm, y_center_utm)
                  if cls_id == pedClass: 
                     print(track.track_id, " DST TO CAM 🧍 =", round(dist,1)) # https://emojipedia.org/
                  if cls_id == carClass:
                     print(track.track_id, " DST TO CAM 🚙 =", round(dist,1))
                  if cls_id == biciClass: 
                     print("DST TO CAM 🚴🏾‍♂️ =", round(dist,1))
                  if cls_id == motoClass: 
                     print("DST TO CAM 🛵 =", round(dist,1)) #🏍️
                
                if DRAW_PATH: 
                  if track.track_id == 1 or track.track_id == 4: draw_path(frame, track.path)  
                           
                xn=[]; yn=[]
                azimuth=999.99
                buffer_size = 24 # 1 sec
                if len(track.path) > 2:#buffer_size*-1:
                     n = len(track.path) if len(track.path) < buffer_size else buffer_size
                     xn, yn = get_last_n_track_positions(n, track.path)
                     box_inside_frame = check_box_inside_frame(n, track.path)
                     person_going_down = 0 #calculate_going_down([i for i in range(size)], yn)       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                     #print("person_going_down",y, person_going_down) 
                     slope, intercept = np.polyfit(xn, yn, 1)
                     point1, point2 = get_two_points_of_line(slope, intercept)
                     if person_going_down: 
                        p1=point1; point1=point2; point2=p1
                     azimuth = calculate_azimuth(north_point1, north_point2, point1, point2)
                     if PRINT_AZIMUTH_IN_FRAME:
                       if(box_inside_frame): cv2.putText(frame, f"{round(azimuth,1)}", (int(x2) -40, int(y2) + 20), cv2.FONT_HERSHEY_PLAIN, 0.9, colors["green"], 1)

                x_utm, y_utm = convert_to_utm_from_homography((x,y), H)
                #print("BBBB",bufer, flush=True)
                
                if obj_id not in object_buffers:
                  # Create buffer and initialize detectionTime only once
                   detectionTime = time.time()
                   object_buffers[obj_id] = {
                        "detectionTime": detectionTime,
                        "history": deque(maxlen=30)
                   }
                else:
                    detectionTime = object_buffers[obj_id]["detectionTime"]
                    
              # Append the latest detection info of the id
              # Append new position/time entry to the per-object history
                object_buffers[obj_id]["history"].append(f"{int(x_utm)},{int(y_utm)},{round(timestamp, 3)}")
               
                # Store data in buffer
                #k=1e7
                bufer.append({
                    "originID": cameraId,
                    "id": int(track.track_id),
                    #"id2": f"{int(x_utm)},{int(y_utm)},{round(timestamp, 3)}",
                    "classID": cls_id,
                    #"position": {"x": int(x_utm), "y": int(y_utm)},
                    "position": {"x": (x_utm), "y": (y_utm)},   #novo                 
                    "azimuth": round(azimuth,2),
                    "speed": round(vel,2),
                    "time": generationModTime, #round(timestamp, 2)
                    "updateTime": time.time(), # seconds since epoch #round(timestamp, 2)
                    "detectionTime":  detectionTime
                })
                
                
                # Append the latest detection info of the id #id2
                #object_buffers[obj_id].append(f"{int(x_utm)},{int(y_utm)},{round(timestamp, 3)}")
                
                #if 1 or obj_id==1: print("TTTT", obj_id, ":", object_buffers[obj_id])
                
                # check if car is stopped
                #time_window=50*(1.0/fps)
                #is_stopped = is_car_stopped(object_buffers[obj_id], obj_id, cls_id, time_window)
                #print("OOOOOOOOO", obj_id, is_stopped)
                
              
                #print("$$$$$", track.track_id, red_flagged_cars, "frame=", frame_count)
                #if track.track_id == 6:
                
                #cv2.rectangle(frame, (x1, y1), (x2, y2), colors["green"], 2) #yellow
                
                if DRAW_OBJECT_BOXES:
                  if cls_id == carClass:                
                    if track.track_id in red_flagged_cars:
                      cv2.rectangle(frame, (x1, y1), (x2, y2), colors["red"], BOX_LINE_THICKNESS)
                    else:
                      cv2.rectangle(frame, (x1, y1), (x2, y2), colors["yellow"], BOX_LINE_THICKNESS)
                  
                  cluster_id = ONE_MEMBER_CLUSTER_ID
                  for cluster in static.clusters_set:
                    if cls_id == pedClass and str(obj_id) in cluster.get("members", []):
                       cluster_id = cluster["clusterID"]
                       break
                  
                  
                  if cls_id == pedClass and cluster_id == ONE_MEMBER_CLUSTER_ID:
                    if track.track_id in red_flagged_peds:
                       cv2.rectangle(frame, (x1, y1), (x2, y2), colors["red"], BOX_LINE_THICKNESS)
                    else:
                       cv2.rectangle(frame, (x1, y1), (x2, y2), colors["yellow"], BOX_LINE_THICKNESS)
                       
                    #print("CHECK: obj_id=", obj_id, "cls_id=", cls_id, "pedClass=", pedClass,"members=", cluster.get("members", []),"match=", str(obj_id) in cluster.get("members", [])) 
                                  

                if GET_CPM_FROM_FILE:
                    cv2.putText(frame, f"Frame: {str(frame_count)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 0.9, colors[FRAME_NR_COLOR_1], 1)
                    frame_already_saved=1
                else:
                  if PRINT_LABEL_IN_FRAME:       
                    cv2.putText(frame, f"FRAME: {str(frame_count)}", (10, 50+frame_offset), cv2.FONT_HERSHEY_PLAIN, 0.9, colors[FRAME_NR_COLOR_1], 1)
                  print(); print(nowTime_ms_mod(), "FRAME", frame_count)
               
                if USE_UVC or USE_REALSENSE: 
                        #out.write(frame)
                        frame_copy = frame.copy()
                        frame_already_saved=1
       
            
        if (USE_SAVED_VIDEO or USE_UVC or USE_REALSENSE) and SAVE_ALL_FRAMES and not frame_already_saved:
              if USE_SAVED_VIDEO or USE_UVC or USE_REALSENSE:
                   if PRINT_LABEL_IN_FRAME:       
                      cv2.putText(frame, f"FRAME: {str(frame_count)}", (10, 50+frame_offset), cv2.FONT_HERSHEY_PLAIN, 0.9, colors[FRAME_NR_COLOR_1], 1)
                      #out.write(frame)
                   frame_copy = frame.copy()                      
                   print(); print(nowTime_ms_mod(), "FRAME", frame_count)
              
                   
                      
        if not GET_CPM_FROM_FILE:
           if CALCULATE_STATS:
                print(STRING1,round(time.time()-lasttime, 6), flush=1) 
                lasttime=time.time()

        north_point1 = (852, 458); north_point2 = (374, 458)
     
        north_point_arrow1 = north_point1
        north_point_arrow2 = (north_point2[0]-0, north_point2[1])
        #perp_point1, perp_point2 = perpendicular_line_points(north_point1, north_point2)
        #print(f"Two points on the perpendicular line: {perp_point1}, {perp_point2}")
        #north_point_arrow1 = perp_point1; north_point_arrow2 = perp_point2
        #north_point1 = north_point_arrow1; north_point2 = north_point_arrow2
        
        #cv2.arrowedLine(frame, north_point_arrow1, north_point_arrow2, color=(255, 255, 255), thickness=2, line_type=cv2.LINE_AA, tipLength=0.1) #relative length of the arrow tip compared to the arrow length
     





        #print("BUFFER",bufer)
        err=0 #0.005
        # Remove old entries (older than flush interval)
        #print("TTTT", timestamp, bufer)
        while bufer and (nowTime_ms_mod() - bufer[0]["time"])%65536 >= flush_interval: #modulo arithmetic to handle timer rollover at 65 536 ms
            #print("REMOVED", bufer[0])
            bufer.popleft()


        # Update shared memory every flush_interval seconds (real time)
        current_time = time.time() # seconds since epoch
        sampling_condition = (current_time - last_flush_time >= sampling_interval - err)

# use_video_timestamp=1 --> Flush a cada 1s de vídeo (timestamp de vídeo)
  #mais realista e consistente com o vídeo
  #Se o vídeo "atrasar" ou a máquina ficar lenta, o flush continua coerente com a timeline do vídeo
  #Melhor para gravação/replay/sync com outras fontes baseadas no vídeo
  #Mais fácil depois para fazer comparações "por segundo de vídeo"
  #use_video_timestamp=0 --> Flush a cada 1s de relógio (tempo real dado por time.time())
  #útil se o processamento for feito "live" com uma câmera ou streaming em tempo real
  #Se processas um vídeo offline (ex.: mp4), pode ser inconsistente com o framerate real
  #Pode dar flushs com intervalos de frames irregulares se o processador estiver ocupado
  #Portanto:
  #Se estiver a processar um vídeo mp4 → flush com base em timestamp do vídeo é mais realista.
  #Se fosse um live camera stream (cv2.VideoCapture(0)) → time.time() seria aceitável.


        
        if USE_VIDEO_TIMESTAMP:
            sampling_condition = (timestamp - last_flush_timestamp >= sampling_interval - err)
            
        #print("CONDITION",sampling_condition, flush=True)

        if GET_CPM_FROM_FILE:
            sampling_condition = ((timeMS - last_timeMS)%65536 >= SAMPLING_INTERVAL)


        if sampling_condition:
            latest_by_id = {}
            for obj in reversed(bufer):  # Keep most recent entry
                if obj["id"] not in latest_by_id:
                    latest_by_id[obj["id"]] = obj
 

            shared_dict.clear()
            shared_dict.update(latest_by_id)
            
            last_flush_time = current_time
            last_flush_timestamp = timestamp

                        
            #print(f"[INFO] Updated shared memory with {len(shared_dict)} objects at t={round(timestamp, 2)}s")
            #print("[SHARED DICT]",shared_dict)
            #print("RRRR", red_flagged_cars)
               
            # Flush to JSON every N seconds (real-world time, not video time)
            
            output_file="recent_objects.json"
            output_proto_file="recent_objects_proto.json"
            current_time = time.time() # seconds since epoch
            #print("TIME=", current_time - last_flush_time)
            
            #retirei este bloco em 20-out-2025 porque estava em duplicado
            #latest_by_id = {}
            #for obj in reversed(bufer):  # reversed to keep the latest entry
            #  if obj["id"] not in latest_by_id:
            #     latest_by_id[obj["id"]] = obj
            #last_flush_time = current_time
            
            with open(output_file, "w") as f:
                if PRINT_PRETTY_MY_JSON:
                    json.dump(list(latest_by_id.values()), f, indent=2)
                if PRINT_MY_JSON:
                    json.dump(list(latest_by_id.values()), f, separators=(",", ":"))
               
            # remove "time" before saving   
            #cleaned_objects = []
            #for obj in latest_by_id.values():
            #   obj_copy = obj.copy()  # make a shallow copy
            #   obj_copy.pop("time", None)  # remove "time" if it exists
            #   cleaned_objects.append(obj_copy)
            #with open(output_file, "w") as f: json.dump(cleaned_objects, f, indent=2)
            
            #print(f"[INFO] Saved {len(latest_by_id)} unique objects to {output_file} at t={round(timestamp, 2)}s")
            
            if PRINT_MY_JSON or PRINT_PRETTY_MY_JSON: #########################################
            # Print the json file content
             with open(output_file, "r") as f:
               data = f.read()
               #print(f"[INFO] Content of {output_file}:\n{data}")
               #print(f"{frame_count}: {data}")
               print(nowTime_ms_mod(), data)
               
            #print("BUFFER", bufer)
            clean_buffer(bufer, BUFFER_TTL)
            #print("buffer", bufer)           
            
            # prepare cpm message
            cpm_dict = buffer_to_cpm_message(None, bufer, None, None, None, None, None, 0)
            #cpm.append(cpm_dict) #cpm is a deque of cpm_dict dictionaries  //aaa
            

            
            if GET_CPM_FROM_FILE:
               cpm_dict={"version": 1.0, "timestamp": 0, "objects": [], "events": []}
               nextLineStartsWithBrace, cpm_reader = next_line_starts_with_brace(cpm_reader)
               print(); print("SAMPLE")
               if nextLineStartsWithBrace:
                 line = next(cpm_reader)
                 cpm_dict = json.loads(line)
                 print(); print("FROM FILE:", cpm_dict["timestamp"], cpm_dict, flush=True)
               
               #for cpm_dict in read_cpm_from_file("cpm_log.txt"):
               #   print("Timestamp:", cpm_dict["timestamp"])
                  #cpm_dict = read_cpm_from_file()
            
            red_flagged_cars, red_flagged_peds = find_red_flagged_objs(cpm_dict)
            
            #print("CPM_DICT",cpm_dict)
            
            #clusters = find_pedestrian_clusters(cpm_dict, distance_threshold=5.0, min_cluster_size=2)
            #if clusters != []: print("CLUSTERS:", clusters)
            
            cpm_with_clusters = add_pedestrian_clusters_to_cpm(cpm_dict, MAX_DISTANCE_TO_BELONG_TO_CLUSTER, min_peds=2)
            #print("MMMMMM", cpm_with_clusters["clusters"])
            static.clusters_set = cpm_with_clusters["clusters"]
            red_flagged_cars_relative_to_clusters, red_flagged_clusters = find_red_flagged_clusters(cpm_with_clusters)
            #print("KKKKK clusters: redFlagCars & redFlagClusters", red_flagged_cars_relative_to_clusters, red_flagged_clusters)
            if DRAW_CLUSTER_BOX:
               frame = draw_boxes_around_pedestrian_clusters(frame, cpm_dict, red_flagged_clusters)
            


            #print(f"Found {len(clusters)} pedestrian clusters:")
            #for i, cluster in enumerate(clusters, 1):
            #     ids = [p['objectID'] for p in cluster]
            #     print(f"Cluster {i}: {ids}")
            
            ped_inside_area = None
            if USE_CASE == 2: ped_inside_area = check_pedestrian_inside_area(cpm_dict)
            
            cpm2send = buffer_to_cpm_message(cpm_with_clusters["clusters"], bufer, red_flagged_cars, red_flagged_peds, red_flagged_cars_relative_to_clusters, red_flagged_clusters, ped_inside_area, 1)
            #print("BBBBBB",bufer); print("bbbbbb",cpm2send)
            sendCPM=True
            if not cpm2send.get("objects"):  # True if [] or objects key missing
                print("CPM has no objects — skip sending.") #⚠️
                sendCPM=False

            if sendCPM:
              # Save to file
              with open("recent_objects_proto.json", "w") as outfile: 
                 json.dump(cpm2send, outfile, indent=2)

               
              if PRINT_PRETTY_PROTO: ##################################
                 # Print the json file content
                 with open(output_proto_file, "r") as infile:
                      data = infile.read()
                      print(f"[INFO] Content of {output_proto_file}:\n{data}")
            
              if 0:
                frame = mark_box_with_right_alert_color(frame, cpm2send, detections, static.clusters_set)
   
              send_message_to_udp_server(cpm2send)

            # Write the processed frame to file before taking the next one
            if (USE_SAVED_VIDEO or USE_UVC or USE_REALSENSE): out.write(frame)

        else: #not sampling_condition
          if (USE_SAVED_VIDEO or USE_UVC or USE_REALSENSE) and frame_copy is not None:
              out.write(frame_copy) # to avoid duplicate frame writes (only one frame per iteration is written).
        
        
        if GET_POINTS_FROM_FRAME:
          # Show the frame
          if frame_count == FREEZE_FRAME_NUMBER and not img_already_georeferenced:
             print(f"[INFO] Frame {FREEZE_FRAME_NUMBER} frozen")
             cv2.destroyAllWindows()
             image_points=getFrame(frame)
             # Calcular matriz de homografia
             H, _ = cv2.findHomography(image_points, utm_points)
             H_inv = np.linalg.inv(H)
        cv2.imshow('YOLO Tracking', frame) # color image
        #cv2.imshow("Gray Tracking", gray_frame) # gray image

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting video visualization...", flush=True) ################################################################################################
            break

        if USE_REALSENSE:
           frames = pipeline.wait_for_frames()
           color_frame = frames.get_color_frame()
           frame = np.asanyarray(color_frame.get_data())
           # Convert to grayscale
           #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
           ret = True
        else:
           ret, frame = cap.read()
           
           if GET_FRAME_NUMBER_FROM_IMAGE:
              frame_number = get_frame_number_from_image(frame)
              if frame_number is not None:
                  print("Detected frame number:", frame_number)
              else:
                  print("No frame number found.")
           
       
        frame_idx += 1

     cap.release()
     cv2.destroyAllWindows()
     print("[INFO] Video processing completed.")
     udp_sock.close()
     

    except KeyboardInterrupt:
      print("\n[INFO] CTRL+C detected, stopping recording...")

    finally: #run always, whether there is an error in the try block or not
      if USE_UVC: cleanup(cap, out)
      if USE_REALSENSE: 
             cleanup_realsense(pipeline, out)
      print(); print("Georeferenced points:")
      #print(); print(image_points.tolist())  # print with decimal 0
      int_image_points = [[round(x), round(y)] for x, y in image_points.tolist()]
      print(int_image_points)
      if CALCULATE_STATS:
         process_log_file(LOG_FILE)
         cpm_time_stats(LOG_FILE)
      get_ntp_info()
      print("Exit program")
      sys.exit(0)     #Exit program






def add_pedestrian_clusters_to_cpm(cpm, eps, min_peds=2):
    """
    Detects clusters of pedestrians (classID=0) in a CPM and adds them as a 'clusters' field.

    Args:
        cpm (dict): A CPM message containing an "objects" list.

        eps (float): Max distance (in meters) between pedestrians to be considered same cluster.
        min_peds (int): Minimum number of pedestrians required to form a cluster.

    Returns:
        dict: The same CPM dict, now with an added key:
              "clusters": [
                  {
                      "clusterID": int,
                      "members": [objectIDs],
                      "center_utm": (x, y)
                  }, ...
              ]
    """

    # ---- Defensive checks ----
    if not isinstance(cpm, dict):
        print("[WARN] CPM is not a dictionary.")
        return cpm
    if "objects" not in cpm or not isinstance(cpm["objects"], list):
        cpm["clusters"] = []
        return cpm

    # ---- Extract pedestrians ----
    peds = [obj for obj in cpm["objects"] if obj.get("classID") == 0]
    if len(peds) < min_peds:
        cpm["clusters"] = []
        return cpm

    coords = []
    obj_ids = []
    for p in peds:
        lon = p.get("lon")
        lat = p.get("lat")
        if lon is None or lat is None:
            continue
        try:
            lon_deg, lat_deg = lon / 1e7, lat / 1e7
            utm_x, utm_y = convert_wgs84_to_utm(lon_deg, lat_deg)
            utm_x, utm_y = lon, lat  #novo
            coords.append([utm_x, utm_y])
            obj_ids.append(str(p["objectID"]))
        except Exception as e:
            print(f"[WARN] UTM conversion failed for object {p.get('objectID')}: {e}")

    if len(coords) < min_peds:
        cpm["clusters"] = []
        return cpm

    coords = np.array(coords)

    # ---- Apply DBSCAN ---- Density-Based Spatial Clustering of Applications with Noise
    db = DBSCAN(eps, min_samples=min_peds, metric="euclidean").fit(coords)
    labels = db.labels_  # -1 means noise / unclustered points

    print("UUUU", coords, labels)

    clusters = []
    for label in set(labels):
        if label == -1:
            continue
        members_idx = np.where(labels == label)[0]
        member_ids = [obj_ids[i] for i in members_idx]
        cluster_points = coords[members_idx]
        center = cluster_points.mean(axis=0)
        wgs_x, wgs_y = convert_utm_to_wgs84(center[0], center[1])
        wgs_x, wgs_y = center[0]/1e7, center[1]/1e7  #novo
        clusters.append({
            "clusterID": int(label),
            "members": member_ids,
            #"center_utm": (float(center[0]), float(center[1]))
            "cLat":  round(wgs_y*1e7),
            "cLon":  round(wgs_x*1e7)
        })

    # ---- Add clusters to CPM ----
    cpm["clusters"] = clusters
    return cpm


def find_pedestrian_clusters(cpm_input, distance_threshold=5.0, min_cluster_size=2):
    """
    Finds pedestrian clusters (classID == 0) in a CPM or deque of CPMs.
    Converts WGS84 lat/lon to UTM using convert_wgs84_to_utm(lon, lat).
    """

    # Handle deque or list input
    if not isinstance(cpm_input, dict):
        # Try to get the last dict in the deque/list that has "objects"
        cpm = None
        for item in reversed(cpm_input):
            if isinstance(item, dict) and "objects" in item:
                cpm = item
                break
        if cpm is None:
            print("[WARN] No valid CPM with 'objects' found.")
            return []
    else:
        cpm = cpm_input

    # Extract pedestrians only
    peds = [obj for obj in cpm.get("objects", []) if obj.get("classID") == 0]
    if len(peds) < min_cluster_size:
        return []

    # Convert to UTM coordinates
    ped_coords = []
    for ped in peds:
        lat = ped["lat"]
        lon = ped["lon"]
        utm_x, utm_y = convert_wgs84_to_utm(lon/1e7, lat/1e7)
        utm_x, utm_y = lon, lat  #novo
        ped_coords.append((ped["objectID"], utm_x, utm_y))

    # Cluster pedestrians by proximity
    clusters = []
    visited = set()

    def distance(p1, p2):
        return math.hypot(p1[1] - p2[1], p1[2] - p2[2])

    def dfs(idx, cluster):
        visited.add(idx)
        cluster.append(ped_coords[idx])
        for j in range(len(ped_coords)):
            if j not in visited and distance(ped_coords[idx], ped_coords[j]) <= distance_threshold:
                dfs(j, cluster)

    for i in range(len(ped_coords)):
        if i not in visited:
            cluster = []
            dfs(i, cluster)
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

    return clusters




def draw_boxes_around_pedestrian_clusters(frame, cpm, redFlaggedClusters):

    color=NO_RISK_CLUSTER_BOX_COLOR

    if not cpm or "clusters" not in cpm or not cpm["clusters"]:
        #print("[INFO] No clusters to draw.")
        return frame

    for cluster in cpm["clusters"]:
        members = cluster.get("members", [])
        cluster_id = cluster.get("clusterID", ONE_MEMBER_CLUSTER_ID)
        if cluster_id in redFlaggedClusters: color=RISK_CLUSTER_BOX_COLOR
        if not members:
            continue

        cluster_points = []
        for obj in cpm.get("objects", []):
            if str(obj["objectID"]) in members:
                lon = obj.get("lon")
                lat = obj.get("lat")
                if lon is None or lat is None:
                    continue
                try:
                    lon_deg, lat_deg = lon / 1e7, lat / 1e7
                    utm_x, utm_y = convert_wgs84_to_utm(lon_deg, lat_deg)
                    utm_x, utm_y = lon, lat  #novo
                    px, py = convert_utm_to_image_pixel(utm_x, utm_y, H_inv)
                    #print("PPPP", obj["objectID"], utm_x, utm_y, px, py)
                    #cluster_points.append((int(px), int(py)))
                    cluster_points.append(((px), (py)))  #novo
                    #print(f"[DEBUG] Cluster {cluster_id} member {obj['objectID']} -> ({px:.1f}, {py:.1f}) UTM=({utm_x:.2f},{utm_y:.2f})")
                except Exception as e:
                    print(f"[WARN] Pixel conversion failed for object {obj['objectID']}: {e}")

        if not cluster_points:
            #print(f"[WARN] Cluster {cluster_id} has no valid pixel points.")
            continue

        # Compute bounding box around cluster points
        xs = [p[0] for p in cluster_points]
        ys = [p[1] for p in cluster_points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_min, x_max = int(min(xs)), int(max(xs)) #novo
        y_min, y_max = int(min(ys)), int(max(ys)) #novo
        #x_avg = int(sum(xs) / len(xs))
        #y_avg = int(sum(ys) / len(ys))        


        d= int((x_max - x_min)*0.1) #estava /len(xs))
        print(f"[INFO] Cluster {cluster_id}: bbox=({x_min},{y_min})→({x_max},{y_max}) with {len(members)} members", d)
        
        label = f"CT {cluster_id}({len(members)})"
        # Draw rectangle and label
        cv2.rectangle(frame, (x_min-d, y_max), (x_max+d, y_min - 75), color, BOX_LINE_THICKNESS)
        if PRINT_LABEL_IN_FRAME:
          cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CLUSTER_FONT_COLOR, 2)


    return frame


def convert_utm_to_image_pixel(x_utm, y_utm, H_inv=None): #novo
    """
    Convert UTM coordinates to image pixel coordinates.
    If H_inv (inverse of homography) is not provided, it will be computed.
    """
    global H

    if H is None:
        raise ValueError("Homography matrix H not set. Compute it first.")

    if H.shape != (3, 3):
        raise ValueError("Homography matrix H must be 3x3")

    if H_inv is None:
        H_inv = np.linalg.inv(H)

    utm_point = np.array([[x_utm, y_utm, 1]], dtype=np.float32).T
    pixel_point = H_inv @ utm_point
    pixel_point /= pixel_point[2, 0]
    #print("PPPPP", x_utm, y_utm, float(pixel_point[0, 0]), float(pixel_point[1, 0]))
    return float(pixel_point[0, 0]), float(pixel_point[1, 0])


#perform the inverse transformation (UTM → image) by doing the inverse of the homography.
def convert_to_image_from_utm(utm_point, H):
    """
    Converts a UTM point (world coordinates) to image pixel coordinates
    using the inverse of the homography matrix H.
    
    Args:
        utm_point (tuple): (x_utm, y_utm)
        H (np.array): 3x3 homography matrix (image -> UTM)

    Returns:
        (x_img, y_img): pixel coordinates in the image
    """
    H_inv = np.linalg.inv(H) # do the inverse of the homography
    point = np.array([utm_point[0], utm_point[1], 1.0]).reshape((3, 1))
    image_point = H_inv @ point
    image_point /= image_point[2]
    return (image_point[0][0], image_point[1][0])
    
    




def mark_box_with_right_alert_color(frame, cpm_dict, detections, clustersSet):
    """
    Draws bounding boxes for detected objects using CPM event types.
    The boxes are red if eventType is 'collisionRisk' and green if 'None'.
    
    Args:
        frame (np.array): Current video frame.
        cpm_dict (dict): CPM data structure containing 'objects' and 'events'.
        detections (list): List of detections [[x1, y1, x2, y2, score], ...]
    """
    
    # Create map of origin -> eventType
    event_map = {str(e["origin"]): e["eventType"] for e in cpm_dict.get("events", [])}
    
    #print("EEEEEE", event_map)

    #cluster_id = ONE_MEMBER_CLUSTER_ID
    #for cluster in clustersSet:
    #  if clsID == pedClass and str(obj_id) in cluster.get("members", []):
    #       cluster_id = cluster["clusterID"]
    #       break
    
    #print("CCCCC",cpm_dict)           
                  
    #if (clsId == pedClass and cluster_id == ONE_MEMBER_CLUSTER_ID) or clsId == carClass:

    # Iterate through CPM objects and detections in parallel
    for idx, obj in enumerate(cpm_dict.get("objects", [])):
        #print("EEEEE", idx, len(detections), obj)
        if idx >= len(detections):
            break  # No corresponding detection for this object
            
        if 0:
           if obj["clusterID"] == ONE_MEMBER_CLUSTER_ID: continue #if ped belongs to a cluster, skip the rest of this loop iteration for not drawing a box around him
        
        x1, y1, x2, y2, score = detections[idx]
        obj_id = str(obj["objectID"])

        event_type_str = event_map.get(obj_id, "null")
        draw=True
        if event_type_str == "collisionRiskEnd":  #"antes era None"
            #print("CCCC green")
            color = colors["green"]
            thick=10
        elif event_type_str == "collisionRisk":
            color = colors["red"]  #(0, 0, 255)  # red (alert)
            thick=10
        else:
            draw=False
            #color = colors["yellow"]
            #thick=2
            

        # Draw rectangle and label
        if draw: cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
       # cv2.putText(frame,  f"ID:{obj_id} {event_type}", (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame



          




def get_frame_number_from_image(frame):
    # Crop where text appears (adjust as needed)
    roi = frame[0:200, 0:500]   # top-left corner
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Improve contrast
    gray = cv2.convertScaleAbs(gray, alpha=2, beta=0)
    
    # Handle both yellow and black text: invert if background is light
    mean_val = np.mean(gray)
    if mean_val > 128:
        gray = 255 - gray  # invert colors
    
    # Threshold to make text crisp
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Optionally enlarge for better OCR accuracy
    thresh = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Debug (optional)
    # cv2.imshow("OCR Region", thresh)
    # cv2.waitKey(1)
    
    # OCR
    text = pytesseract.image_to_string(thresh, config="--psm 6").strip()
    print("OCR raw:", repr(text))
    return None
    # Extract frame number
    match = re.search(r"frame[:\s]*([0-9]+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    else:
        print("No frame number found.")
        return None




def next_line_starts_with_brace(gen):
    # Peek using lookahead via next() + chain, not tee()
    try:
        next_line = next(gen)
    except StopIteration:
        return False, gen  # no next line

    starts_with_brace = next_line.lstrip().startswith('{')
    # Put the line back at the front of the generator
    gen = itertools.chain([next_line], gen)
    return starts_with_brace, gen
    

def next_line_starts_with_brace_old(gen):
    """
    Check if the next line in a generator starts with '{' without consuming it.

    Args:
        gen: a generator yielding lines (strings)

    Returns:
        True if the next line starts with '{', False otherwise.
    """
    peek, _ = itertools.tee(gen)  # duplicate iterator
    next_line = next(peek, None)
    if next_line is None:
        return False  # generator is exhausted
    return next_line.lstrip().startswith('{')
    
       
def cleanup(cap, out):
    print("\n[INFO] Cleaning up...", flush=True)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    

def cleanup_realsense(pipeline, out):
    """
    Cleans up RealSense pipeline, video writer, and OpenCV windows.
    """
    print("\n[INFO] Cleaning up...", flush=True)

    # Stop RealSense pipeline
    if pipeline is not None:
        pipeline.stop()
        print("[INFO] RealSense pipeline stopped.", flush=True)

    # Release video writer if used
    if out is not None:
        out.release()
        print("[INFO] VideoWriter released.", flush=True)

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()
    print("[INFO] OpenCV windows destroyed.", flush=True)




def read_cpm_from_file(filepath="videoLog_onlyCPMs.txt"):
    """
    Generator that yields one CPM dictionary at a time
    from a file containing one JSON CPM per line.
    """
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip blank lines
            yield line
            #try:
            #    yield json.loads(line)
            #except json.JSONDecodeError as e:
            #    print(f"[WARN] Skipping invalid JSON line: {e}")






#Calculate distance between the camera reference point and an object.
def get_distance_to_camera(cameraPoint, x, y):
    cx, cy = cameraPoint
    return math.sqrt((x - cx) ** 2 + (y - cy) ** 2)



def is_car_stopped(buffer, objID, classId, time_window, distance_threshold=6.0, speed_threshold=0.5):
    """
    Detect if a car is stopped based on its recent trajectory.

    Args:
        buffer (deque or list): each entry is "x,y,t" (string).
        distance_threshold (float): max displacement (m) allowed to still consider 'stopped'.
        time_window (float): seconds of history to check.
        speed_threshold (float): max avg speed (m/s) allowed to still consider 'stopped'.

    Returns:
        bool: True if car is stopped, False otherwise.
    """
    
    if classId==0: return False # for pedestrians
    
    
    if len(buffer) < 2:
        return False  # not enough data

    # Parse
    history = [tuple(map(float, entry.split(','))) for entry in buffer]
    
    # history is a list of float tuples: [(550147.0, 4601312.0, 1.733), (550147.0, 4601312.0, 1.767), (550144.0, 4601310.0, 2.200), ...]
    
    
    _, _, t_latest = history[-1] #ignore the first two elements (x and y) of the tuple; get timestamp of the most recent point

    # Keep only last `time_window` seconds
    recent = [p for p in history if p[2] >= t_latest - time_window]
    # recent = [(x1, y1, t1), (x2, y2, t2), (x3, y3, t3), ...]
    
    if len(recent) < 2:
        return False

    xs, ys, ts = zip(*recent)
    
    #xs = (x1, x2, x3, ...)
    #ys = (y1, y2, y3, ...)
    #ts = (t1, t2, t3, ...)
    
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    displacement = math.hypot(dx, dy)

    duration = max(ts) - min(ts)
    avg_speed = displacement / duration if duration > 0 else 0
    

    # 🚗 Stopped if BOTH displacement and speed are below thresholds
    res = displacement < distance_threshold #and avg_speed < speed_threshold
    
    if 1 or objID==1: 
     print("xxxxx", objID, displacement, res, flush=True); 
     #print("XXXXX", objID, displacement, avg_speed, res, flush=True); 
     #print(recent)
     #print(buffer)

    
    return res



def buffer_to_cpm_message(cluster, bufffer, redFlaggedCars, pedFlaggedCars, redFlaggedCarsRelativeToClusters, redFlaggedClusters, pedInsideArea, incCounter):
    """
    Convert a bufer containing multiple tracked objects into a single CPM-style message.

    Parameters:
        bufer: deque or list of dicts, each with keys:
            'id', 'classId', 'position' {'x','y'}, 'azimuth', 'speed', 'time'
        event_type: string, type of CPM event to attach per object

    Returns:
        dict: CPM-style message
    """
    # emulate a static variable
    static=buffer_to_cpm_message # just another reference to the buffer_to_cpm_message object
    
    if not hasattr(static, "counter"): static.counter = 0  # initialize once
    
    if not hasattr(static, "last_counter"): static.last_counter = 0 
        
    if not hasattr(static, "event_counter"): static.event_counter = 1
    
    if not hasattr(static, "cpm_counter"): static.cpm_counter = 0
    
    if not hasattr(static, "nrSucessiveCollisionRisks"): static.nrSucessiveCollisionRisks = 0  
    
    if not hasattr(static, "nrSucessiveCollisionRiskEnds"): static.nrSucessiveCollisionRiskEnds = 0  
   
    if not hasattr(static, "newEvent"): static.newEvent = True  
    
    if not hasattr(static, "thereWasAlreadyOneCollisionRisk_state"): static.thereWasAlreadyOneCollisionRisk_state = {}  # {obj_id: bool}
    #static dictionary that maps each objectID → boolean. Used to avoid collisionRiskEnd before any previous collisionRisk.
    #como o carro só pode estar em collisionRisk e collisionRiskEnd, o carro enviava inicialmente sempre collisionRiskEnd pois não estava em collisionRisk
    # O thereWasAlreadyOneCollisionRisk_state evita que tal aconteça
        
    if not bufffer or USE_ONLY_FOR_OBJECT_DETECTION:
        # If bufffer is empty, return an empty CPM message
        return {
            "version": 1.0,
            "timestamp": 0,
            "messageID": static.cpm_counter,
            "objects": [],
            "events": [],
            #"clusters": []
        }

        
    latest_by_id = {}
    
    nrPeds=0; nrCars=0
    
    # Keep only the most recent entry per object ID
    for obj in reversed(bufffer):
        if obj["id"] not in latest_by_id:
            latest_by_id[obj["id"]] = obj
            if obj["classID"] == pedClass: nrPeds+=1
            if obj["classID"] == carClass: nrCars+=1
          

    # Build CPM objects
    cpm_objects = []
    cpm_events = []

    
    if cluster == None: cluster = []
    if redFlaggedCarsRelativeToClusters == None: redFlaggedCarsRelativeToClusters = []
    if redFlaggedClusters == None: redFlaggedClusters = []
    #if redFlaggedCars == None: redFlaggedCars = []
    #if pedFlaggedCars == None: pedFlaggedCars = []
     
    
        
    cpm_clusters = cluster
    
# [{'clusterID': 0, 'members': ['11', '10', '7', '4', '6', '5'], 'cLat': 415615203, 'cLon': -83994006}]    
    
    nrObjects = len(latest_by_id)
    nrEvents = len(cpm_events) or 1 # "or" returns the first truthy value it finds. If len(cpm_events)=0, is False, so "or" evaluates the next value (1) and assigns it

    clusterSize = 1 if cpm_clusters == [] else len(cpm_clusters[0]["members"])
    clusterSize = 0 if nrPeds == 0 else clusterSize
    
    #print("CCCCCCCCC", cpm_clusters, clusterSize)

#CPM transmission should be suppressed if: i) No object is detected, and ii) No change in the local perception environment is observed. (ETSI TR 103 562, section 5.2)

    
    if incCounter: static.cpm_counter+=1


    for obj_id, obj in latest_by_id.items():

        # --- Find cluster ID for this object ---
        cluster_id = ONE_MEMBER_CLUSTER_ID  # Default: not in any cluster
        for cluster in cpm_clusters:
            if str(obj_id) in cluster.get("members", []):
                cluster_id = cluster.get("clusterID", ONE_MEMBER_CLUSTER_ID)
                break

        # --- Coordinates utm ---
        x_utm, y_utm = obj["position"]["x"], obj["position"]["y"] #novo
        
        if static.newEvent: 
                static.counter += 1
                static.newEvent = False

        azimuth = int(obj["azimuth"] / 0.1)
        if azimuth > 3600:
             azimuth = AZIMUTH_UNAVAILABLE # field uses 14 bits: 16383 --> max=590 km/h, 16383/3 = 5461; 16383/2 = 8191

        speed = int(round(obj["speed"] * 100))
        if speed > 8191:
            speed = SPEED_UNAVAILABLE
        if USE_ZERO_SPEED:
            speed = 0

        altitude = 21100 + random_int(0, 5)  # centimetros
        
        # --- Ensure persistent state exists ---
        state = static.thereWasAlreadyOneCollisionRisk_state
        if obj_id not in state:
            state[obj_id] = False

        cpm_objects.append({
            "originID": obj["originID"],
            "objectID": str(obj_id),
            "classID": obj["classID"],
            "speed": speed,       # cm/s units
            "speedConfidence": 127, #p.56
            "positionConfidence": 7,
            "heading": azimuth,      # CPM heading in 0.1 degree units
            "headingConfidence": 127, #p.38
            "lat": (y_utm * 1e0), #micro degrees  Y center point coord at box bottom line  #novo  1e7 para 1e0
            "lon": (x_utm * 1e0), #micro degrees  X center point coord at box bottom line  #novo  1e7 para 1e0
            "altitude": altitude, # cm
            "altitudeConfidence": 8, # accuracy <= 5 metres
            "lane": 1,
            #"laneConfidence": 1
            #"referenceTime": obj["updateTime"],
            #"detectionTime": obj["detectionTime"],
            "clusterID": cluster_id
        })
      

        if redFlaggedCars is not None and pedFlaggedCars is not None:
        # ---- ONLY ADD EVENTS FOR VEHICLES (classID == 2) ----
         if SEND_ONLY_CAR_EVENTS:
           condition1 = (obj_id in redFlaggedCars and obj["classID"] == carClass)
           if obj["classID"] == carClass:
              
              if condition1 == False:
                  static.nrSucessiveCollisionRiskEnds += 1
                  if static.nrSucessiveCollisionRiskEnds > nrEvents * NR_MAX_SUCESSIVE_CollisionRiskEnd_EVENTS: 
                      static.newEvent = True
                      static.nrSucessiveCollisionRisks = 0
                      continue
              else: 
                  static.nrSucessiveCollisionRiskEnds = 0
                  static.newEvent = False
                  static.nrSucessiveCollisionRisks += 1
                  if static.nrSucessiveCollisionRisks > nrEvents * NR_MAX_SUCESSIVE_CollisionRisk_EVENTS: continue
              
              #if static.counter > static.last_counter: 
              #   static.event_counter += 1
              #   static.last_counter = static.counter
              
              evento={
              "eventID": static.counter,
              "eventType": "collisionRisk" if condition1 == True else "collisionRiskEnd", #"antes era None"
              "origin": str(obj_id),
              "lat": (y_utm * 1e0),
              "lon": (x_utm * 1e0),
              "altitude": altitude, # centimetros
              "referenceTime": obj["updateTime"],
              "detectionTime": obj["detectionTime"],
              "clusterSize": clusterSize              
              }
              
              # --- Update persistent state ---
              if condition1:
                        state[obj_id] = True # thereWasAlreadyOneCollisionRisk
              if state[obj_id]:
                        cpm_events.append(evento)
                                
              #print("OOOOO",obj_id, state[obj_id])
             
             
        #print("RRRRRRR",redFlaggedCarsRelativeToClusters, redFlaggedClusters)    
        if redFlaggedCarsRelativeToClusters != [] and redFlaggedClusters != [] and not condition1: #para evitar evento duplicado
        # ---- ONLY ADD EVENTS FOR VEHICLES (classID == 2) ----
         if SEND_ONLY_CAR_EVENTS:
           if obj["classID"] == carClass:
              condition2 = (obj_id in redFlaggedCarsRelativeToClusters and obj["classID"] == carClass)
              if condition2 == False:
                  static.nrSucessiveCollisionRiskEnds += 1
                  static.newEvent = True
                  static.nrSucessiveCollisionRisks = 0
                  if static.nrSucessiveCollisionRiskEnds > nrEvents * NR_MAX_SUCESSIVE_CollisionRiskEnd_EVENTS: continue
              else:         
                  static.nrSucessiveCollisionRiskEnds = 0
                  static.newEvent = False
                  static.nrSucessiveCollisionRisks += 1
                  if static.nrSucessiveCollisionRisks > nrEvents * NR_MAX_SUCESSIVE_CollisionRisk_EVENTS: continue

              
              cpm_events.append({
              "eventID": static.counter,
              "eventType": "collisionRisk" if condition2 == True else "collisionRiskEnd", #"antes era None"
              "origin": str(obj_id),
              "lat": (y_utm * 1e0),
              "lon": (x_utm * 1e0),
              "altitude": altitude, # centimetros
              "referenceTime": obj["updateTime"],              
              "detectionTime": obj["detectionTime"],
              "clusterSize": clusterSize
             })
             
                
         if not SEND_ONLY_CAR_EVENTS:  #Add an event per object (ped and vehicle)
            condition = (obj_id in redFlaggedCars and obj["classID"] == carClass) or (obj_id in pedFlaggedCars and obj["classID"] == pedClass) and len(redFlaggedCars)>1
            if condition == False: 
               static.nrSucessiveCollisionRiskEnds += 1
               if static.nrSucessiveCollisionRiskEnds > nrEvents * NR_MAX_SUCESSIVE_CollisionRiskEnd_EVENTS: continue
            else:
                  static.nrSucessiveCollisionRiskEnds = 0
                  static.newEvent = False
                  static.nrSucessiveCollisionRisks += 1
                  if static.nrSucessiveCollisionRisks > nrEvents * NR_MAX_SUCESSIVE_CollisionRisk_EVENTS: continue
            
          
            cpm_events.append({
              "eventID": static.counter,
              "eventType": "collisionRisk" if condition == True else "collisionRiskEnd", #"antes era None"
              "origin": str(obj_id),
              "lat": (y_utm * 1e0),
              "lon": (x_utm * 1e0), 
              "altitude": altitude, # centimetros
              "referenceTime": obj["updateTime"],              
              "detectionTime": obj["detectionTime"],
              "clusterSize": clusterSize
             })




    
    #global AREA_POLYGON_UTM  
    if USE_CASE == 2:
            static.counter+=1
            center_x_utm, center_y_utm = polygon_center_utm(AREA_POLYGON_UTM)
            center_x_utm, center_y_utm = convert_utm_to_wgs84(center_x_utm, center_y_utm)
            center_x_utm, center_y_utm = center_x_utm/1e7, center_y_utm/1e7  #novo
            cpm_events.append({
              "eventID": static.counter,  # or a unique event counter
              "eventType": "pedAtCrossing" if pedInsideArea else "noPedAtCrossing",
              "lat": (center_y_utm * 1e0),
              "lon": (center_x_utm * 1e0),
              "altitude": altitude, # centimetros
              "referenceTime": obj["updateTime"],              
              "detectionTime": obj["detectionTime"],
              "clusterSize": clusterSize
            })
    #print("ccccccc",cpm_objects)
    
    # Build final CPM message
    if SEND_CPM_EVENTS: 
     if cpm_events != []:
       cpm_message = {
         "version": 1.0,
         #"timestamp": int(round(bufer[-1]["time"] * 1000)) % 65536,  # last timestamp in bufer
         "timestamp": time.time(), #bufer[-1]["time"],  # last timestamp in bufer
         "messageID": static.cpm_counter,
         "objects": cpm_objects,
         #"clusters": cpm_clusters,
         "events": cpm_events
       }
     else:
      cpm_message = {
        "version": 1.0,
        #"timestamp": int(round(bufer[-1]["time"] * 1000)) % 65536,  # last timestamp in bufer
        "timestamp": time.time(), #bufer[-1]["time"],  # last timestamp in bufer
        "messageID": static.cpm_counter,
        "objects": cpm_objects,
        #"clusters": cpm_clusters
      }
     


    return cpm_message


def random_int(n, m):
    """Return a random integer between n and m (inclusive both endpoints)."""
    return random.randint(n, m)

#Get the centroid (lat/lon) of a given cluster
def get_cluster_coordinates(cpm, cluster_id):
    """
    Returns (lat, lon) in degrees for the given cluster_id, if available.
    """
    for cluster in cpm.get("clusters", []):
        if cluster.get("clusterID") == cluster_id:
            lat = cluster.get("cLat")
            lon = cluster.get("cLon")
            if lat is not None and lon is not None:
                # convert from 1e-7 integer representation to degrees
                return lat / 1e7, lon / 1e7
    return None

def polygon_center_utm(area_coords_utm):
    """
    Calculates the UTM coordinates (easting, northing) of the center (centroid)
    of a polygon defined by UTM coordinates.

    Parameters:
    - area_coords_utm (list of tuples): [(easting, northing), ...]

    Returns:
    - (center_easting, center_northing): tuple of floats
    """
    poly = Polygon(area_coords_utm)
    centroid = poly.centroid
    return (centroid.x, centroid.y)

#Function to convert (lon, lat) → (utm_x, utm_y).
def convert_wgs84_to_utm(lon, lat):
    """
    Convert WGS84 geographic coordinates (lon, lat) → UTM (x, y).
    
    Args:
        lon (float): Longitude in degrees
        lat (float): Latitude in degrees
    
    Returns:
        (float, float): UTM (x, y) in meters
    """
    # WGS84 (EPSG:4326) → UTM Zone 29N (EPSG:32629)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32629", always_xy=True)
    
    utm_x, utm_y = transformer.transform(lon, lat)
    return utm_x, utm_y
    
    
#convert from UTM (EPSG:32629 for northern hemisphere) to WGS84 (EPSG:4326)
def convert_utm_to_wgs84(utm_x, utm_y):

   # Define transformer: UTM northern hemisphere → WGS84 lat/lon
   transformer = Transformer.from_crs("EPSG:32629", "EPSG:4326", always_xy=True)

   lon, lat = transformer.transform(utm_x, utm_y)

   #print(f"Lat_utm: {utm_x}, Long_utm: {utm_y}")
   #print(f"Latitude: {lat}, Longitude: {lon}")
   
   return lon, lat


#Send a CPM message to a UDP server using an existing socket.
def send_message_to_udp_server(cpm, sock=udp_sock, udp_ip=UDP_IP, udp_port=UDP_PORT):

    LAT=415616150; LON=-83991518
    
    if SENT_WGS84_IN_UDP_JSON: K=1e7 
    else: K=1
    
    for obj in cpm.get("objects", []):
       wgs_x, wgs_y = convert_utm_to_wgs84(obj["lon"], obj["lat"])
       obj["lat"] = int(wgs_y*K) if not USE_PREDEFINED_COORDS else LAT
       obj["lon"] = int(wgs_x*K) if not USE_PREDEFINED_COORDS else LON

    # Safely update lat/lon in CPM events (if any)
    for ev in cpm.get("events", []):
       wgs_x, wgs_y = convert_utm_to_wgs84(ev["lon"], ev["lat"])
       ev["lat"] = int(wgs_y*K) if not USE_PREDEFINED_COORDS else LAT
       ev["lon"] = int(wgs_x*K) if not USE_PREDEFINED_COORDS else LON



    prettyPrint=0
    
    static=send_message_to_udp_server
    if not hasattr(static, "counter_cpm"): static.counter_cpm = 0  # initialize once
    
    
    if prettyPrint:
           msg_str = json.dumps(cpm, indent=2)  # pretty-print JSON
    else:
           msg_str = json.dumps(cpm, separators=(",", ":"))  # dump (i.e. makes) JSON in a single compact line (string)
        #if msg_str == '{"version":1.0,"timestamp":0,"objects":[],"events":[]}': return
    msg_bytes = json.dumps(cpm).encode('utf-8') #MSG_ENCODING
    if PRINT_SENT_PROTO:
          #print(nowTime_ms_mod(), f"Send CPM message to {udp_ip}:{udp_port}:", msg_str)
          static.counter_cpm += 1
          last_nowTime_ms_mod = nowTime_ms_mod()
          print(static.counter_cpm, nowTime_ms_mod(), "CPM tx", msg_str, flush=True)
    try:
        sock.sendto(msg_bytes, (udp_ip, udp_port))        
    except Exception as e:
        print(f"[ERROR] Failed to send CPM message: {e}")


        

def unix_socket_reader(socket_path, shared_queue):
    print(f"[UNIX-READER] unix_socket_reader started...")
    client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    # Retry loop
    while not os.path.exists(socket_path):
        print(f"[UNIX-READER] Waiting for socket {socket_path} to appear...", flush=True)
        time.sleep(0.1)

    connected = False
    while not connected:
        try:
            client_socket.connect(socket_path)
            connected = True
            print(f"[UNIX CLIENT] Connected to {socket_path}")
        except socket.error as e:
            print(f"[UNIX-READER] Connect failed ({e}), retrying...", flush=True)
            time.sleep(0.1)

    bufer = b""
    try:
        while True:
            rlist, _, _ = select.select([client_socket], [], [], 0.1)
            if rlist:
                #print("[UNIX-READER] Ready to recv", flush=True)
                data = client_socket.recv(1024)
                #print(f"[UNIX-READER] Received raw data: {data!r}", flush=True)
                if not data:
                    print("[UNIX CLIENT] Server disconnected.")
                    break
                bufer += data
                while b"\n" in bufer:
                    line, bufer = bufer.split(b"\n", 1)
                    msg = line.decode().strip()
                    print(f"[UNIX-READER] Received: {msg}", flush=True)
                    shared_queue.append(msg)
            else:
                time.sleep(0.01)
    except Exception as e:
        print(f"[UNIX-READER] Exception: {e}")
    finally:
        client_socket.close()


       



def send_messages_to_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp_client_socket:
        try:
            tcp_client_socket.connect((REMOTE_SERVER_IP, REMOTE_SERVER_PORT))
            print(f"[TCP CLIENT] Connected to {REMOTE_SERVER_IP}:{REMOTE_SERVER_PORT}")

            while True:
                msg = input("Type message to send to RSU server ('q!' to quit): ")
                if msg.lower() == 'q!':
                    print("Closing connection...")
                    break
                tcp_client_socket.sendall((msg + '\n').encode())
                print(f"[TCP CLIENT] Sent: {msg}")

        except Exception as e:
            print(f"[TCP CLIENT] Exception: {e}")
            
            

def check_and_cut_wrong_path(path):  
       max_dist = 10
       for i in range(1, len(path)):
           if path[i][0] == None: continue
           point_x= path[i][0]
           point_x_prev= path[i-1][0]
           #point_y= path[i][1]
           #point_y_prev= path[i-1][1]
           if abs(point_x - point_x_prev) > max_dist:
                 #print("DDDDDDD", i, point_x, point_x_prev, abs(point_x - point_x_prev), "$$$$$", path, "######", path[i:])
                 if point_x > point_x_prev: path = path[i:] 
                 else: path = path[:i] 
                 return path
       return path     

def get_last_n_track_positions(n, track_path):
     x=[]; y=[]
     for i in range(len(track_path)-n, len(track_path)):
           if track_path[i][0] == None: continue
           point_x= track_path[i][0]
           point_y= track_path[i][1]
           if point_x < 0 or point_y < 0: box_inside_frame=0
           x += [point_x]
           y += [point_y]
           #print ("x=",x)
           #print("y=",y)
     return x, y


def check_box_inside_frame(n, track_path):
     return 1
     box_inside_frame = 1
     for i in range(len(track_path)-n, len(track_path)):
           if track_path[i][0] == None: continue
           point_x= track_path[i][0]
           point_y= track_path[i][1]
           if not(0 < point_x < 640) or not(0 < point_y < 360): 
              box_inside_frame=0
              break
     return box_inside_frame   


def calculate_going_down(x, y):
   slope, intercept = np.polyfit(x, y, 1)
   return (int(slope/abs(slope)) > 0)


def get_two_points_of_line(slope, intercept):
   #First point 
   x1 = 0; y1 = slope * x1 + intercept
   # Second point
   x2 = 200 #frame_width
   y2 = slope * x2 + intercept
   #print("TTTTTTT",(x1, y1), (x2, y2))
   # Return the two points as tuples
   return (x1, y1), (x2, y2)

def get_end_points_of_track(track_path): #unused
   start_point_x= track_path[i-1][0]
   start_point_y= track_path[i-1][1]
   end_point_x = track_path[i][0]
   end_point_y = track_path[i][1]
   #declive= (end_point_y - start_point_y) / (end_point_x - start_point_x)
   #print("DDDDD", declive, north_frame - math.atan(declive)*180/math.pi)
   return (start_point_x, start_point_y), (end_point_x, end_point_y)
   
####################

# Define the two points of the original line north_point1 = (230, 290) north_point2 = (430, 45) 
# Function to find the slope of a line given two points 
def calculate_slope(x1, y1, x2, y2): 
   if x2 - x1 == 0: # Vertical line case 
        return None 
   return (y2 - y1) / (x2 - x1) 

 
# Calculate two points on a perpendicular line at a specific distance
def perpendicular_line_points(pt1, pt2, distance=100): 
   # Calculate the slope of the original line 
   slope = calculate_slope(pt1[0], pt1[1], pt2[0], pt2[1]) 
   # Calculate the slope of the perpendicular line 
   if slope is None: 
   # Original line is vertical, perpendicular line will be horizontal 
      perp_slope = 0 
   elif slope == 0: # Original line is horizontal, perpendicular line will be vertical 
      perp_slope = None 
   else: perp_slope = -1 / slope 
   # Midpoint of the original line 
   midpoint = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2) 
   # If perpendicular slope is None (vertical line) 
   if perp_slope is None: 
   # Move up and down along the y-axis (since perpendicular is vertical) 
      point1 = (midpoint[0], midpoint[1] + distance) 
      point2 = (midpoint[0], midpoint[1] - distance) 
   else: # Move along the perpendicular slope to find two points 
      dx = distance / math.sqrt(1 + perp_slope ** 2) 
      dy = perp_slope * dx 
      point1 = (int(midpoint[0] + dx), int(midpoint[1] + dy)) #opencv usa inteiros
      point2 = (int(midpoint[0] - dx), int(midpoint[1] - dy)) 
   return point1, point2 




#######################

def calculate_azimuth(north_point1, north_point2, center_point1, center_point2): 
# Invert the y-coordinates to account for the top-left origin of image coordinates 
   north_point1 = (north_point1[0], -north_point1[1]) 
   north_point2 = (north_point2[0], -north_point2[1]) 
   center_point1 = (center_point1[0], -center_point1[1]) 
   center_point2 = (center_point2[0], -center_point2[1]) 
   # Define the vectors (north and movement) 
   north_vector = np.array([north_point2[0] - north_point1[0], north_point2[1] - north_point1[1]]) 
   movement_vector = np.array([center_point2[0] - center_point1[0], center_point2[1] - center_point1[1]]) 
   # Normalize the vectors 
   north_vector = north_vector / np.linalg.norm(north_vector) 
   movement_vector = movement_vector / np.linalg.norm(movement_vector) 
   # Calculate the angle between the vectors using the dot product and arctan2 
   angle_radians = np.arctan2(movement_vector[1], movement_vector[0]) - np.arctan2(north_vector[1], north_vector[0]) 
   # Convert to degrees 
   angle_degrees = -np.degrees(angle_radians) # minus beacuse the y-axis increases downwards from the top-left corner (origin)
   # Normalize the angle to be within [0, 360] degrees 
   azimuth = (angle_degrees + 360) % 360 
   return azimuth


def draw_path(frame, path): # Draw the line connecting successive center points
           for i in range(1, len(path)): 
              start_point = (int(path[i - 1][0]), int(path[i - 1][1])) 
              end_point = (int(path[i][0]), int(path[i][1])) 
              cv2.line(frame, start_point, end_point, color=(255, 0, 0), thickness=2) # Blue line 
              #print("XXXXX",track_id, start_point, end_point)

#----------------------------



previous_distances = {}

def euclidean_distance(pos1, pos2):
    #print("DDDD",int(pos1['x']-550000), int(pos2['x']-550000), ":", int(pos1['y']-4600000), int(pos2['y']-4600000), ": d=", round(math.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['y'] - pos2['y'])**2),1), round(time.time()-1758225800,5))
    #return abs(pos1['y'] - pos2['y'])
    return math.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['y'] - pos2['y'])**2)



# Inicializa o tracking_state (uma vez)
tracking_state = {}





def check_pedestrian_inside_area(cpm):
    """
    Returns 1 if any pedestrian (classID == 0) is inside the given UTM area, else 0.

    Parameters:
    - cpm (str or dict): CPM message (JSON string or parsed dict)
    - area_coords_utm (list of tuples): list of (easting, northing) coordinates defining the area polygon
    """

    # Parse JSON if needed
    #if isinstance(cpm, str):
    #    cpm_data = json.loads(cpm)
    #else:
    #    cpm_data = cpm

    cpm_data = cpm
    
    global AREA_POLYGON_UTM
    
    
    # Build polygon in UTM coordinates
    area_poly = Polygon(AREA_POLYGON_UTM)

    for obj in cpm_data.get("objects", []):
        # classID == 0 → pedestrian
        if obj.get("classID") == pedClass:
            easting = obj["lon"]
            northing = obj["lat"]

            point = Point(easting, northing)

            if area_poly.contains(point):
                return True  # pedestrian found inside area

    return False  # no pedestrian inside area



def check_approaching_objects(peds_list, cars_list, tracking_state,
    history_len=HISTORY_LEN,                     # N = nº de valores para a média
    #history_len1=5
):


    parsed_peds = []
    parsed_cars = []


    for ped in peds_list:
        try:
            ped_id = int(ped["id"])
            ped_pos = {
                "x": float(ped["position"]["x"]),
                "y": float(ped["position"]["y"])
            }
            time_stamp = float(ped.get("time", 0.0))

            parsed_peds.append({
                "id": ped_id,
                "position": ped_pos,
                "time": time_stamp,
                "approaching_car": 0 # reset at start 
            })
        except Exception as e:
            print(f"[WARN] Invalid pedestrian message: {ped} ({e})")

   
    
    for car_msg in cars_list:
        try:
            car_id = int(car_msg["id"]) 
            class_id = str(car_msg["classID"])
            x = float(car_msg["position"]["x"])
            y = float(car_msg["position"]["y"])
            time_stamp = float(car_msg.get("time", 0.0)) #If "time" does not exist, returns 0.0.

            parsed_cars.append({
                "id": car_id,
                "classID": class_id,
                "position": {"x": x, "y": y},
                "time": time_stamp,
                "approaching_pedestrian": 0 #estava 1
            })

        except Exception as e:
            print(f"[WARN] Invalid car message: {car_msg} ({e})")
    

    
    # Processa cada par carro-pedestre
    for car in parsed_cars:
        car_id = car['id']
        car_pos = car['position']

        for ped in parsed_peds:
            ped_id = ped["id"]
            ped_pos = ped['position']
            current_distance = euclidean_distance(car_pos, ped_pos)
            key = (car_id, ped_id)

            # Se não existir entrada para este par, inicializar
            if key not in tracking_state:
                tracking_state[key] = {
                    'distances': deque(maxlen=history_len),
                    #'distances1': deque(maxlen=history_len1), #adi
                    'reductions': deque(maxlen=history_len),
                    'previous_car_pos': {'x': car_pos['x'], 'y': car_pos['y']},
                    'previous_distance': current_distance + 1
                }

            state = tracking_state[key]

            if history_len == 1:
                # --- Original behavior ---
                previous_distance = state['previous_distance']
                diff = previous_distance - current_distance
                #state['distances1'].append(diff)
                #avg_current_distance = np.mean(state['distances1'])
                car_approaching_ped = avg_current_distance > 0
                #print("######## ", "diff=", diff, "avg=", avg_current_distance, "approach=", car_approaching_ped, flush=True)
                if car_approaching_ped and current_distance <= previous_distance:
                    car['approaching_pedestrian'] = 1
                    ped['approaching_car'] = 1  # once any car sets it, it stays 1
                else: #current_distance > previous_distance
                    car['approaching_pedestrian'] = 0

                state['previous_distance'] = current_distance

            else:
                # --- Smoothed behavior with history ---
                state['distances'].append(current_distance)

                avg_current_distance = np.mean(state['distances'])

                if len(state['distances']) > 1:
                    reduction = state['distances'][-2] - state['distances'][-1]
                    state['reductions'].append(reduction)

                    dist_list = list(state['distances'])
                    #print("LLLL", dist_list)
                    avg_previous_distance = np.mean(dist_list[:-1])
                else:
                    avg_previous_distance = avg_current_distance

                #print("DDDDD", avg_previous_distance, avg_current_distance, avg_previous_distance - avg_current_distance) # aparece nos logs dos testes
                if round(avg_current_distance,2) < round(avg_previous_distance + DISTANCE_MARGIN_ADD_UP,2):
                    car['approaching_pedestrian'] = 1
                    ped['approaching_car'] = 1  # once any car sets it, it stays 1
                    #print("OLAOLAOLA", ped['id'], ped['approaching_car'], parsed_peds)
                #if avg_current_distance < avg_previous_distance:
                #elif avg_current_distance - avg_previous_distance < DISTANCE_MARGIN_ADD_UP:
                #    car['approaching_pedestrian'] = 1
                #    ped['approaching_car'] = 1
                else:  
                    car['approaching_pedestrian'] = 0
                    #ped['approaching_car'] = 0 #commented because if one car is approaching and another is not, the ped might incorrectly end up with 0.

            # Guardar posição para o próximo frame
            state['previous_car_pos'] = {'x': car_pos['x'], 'y': car_pos['y']}
            
    #print("PPPPPPPPP",parsed_cars, parsed_peds)
    return parsed_cars, parsed_peds




def check_approaching_clusters(
    clusters_list,
    cars_list,
    tracking_state,
    history_len=5
):
    parsed_clusters = []
    parsed_cars = []

    #print("CLUSTERS LIST", clusters_list)
    #print("CARS LIST", cars_list)
    # --- Parse clusters ---
    for cluster in clusters_list:
        try:
            cluster_id = int(cluster["clusterID"])
            members = cluster.get("members", [])
            #print("CLUSTERS LIST_2", cluster_id, members)
            
            # Compute cluster center (average of member positions)
            #member_positions = [
            #    (float(m["position"]["x"]), float(m["position"]["y"]))
            #    for m in members if "position" in m
            #]
            #print("CLUSTERS LIST_3", member_positions)
            #if not member_positions:
            #    continue

            #cx = np.mean([m[0] for m in member_positions])
            #cy = np.mean([m[1] for m in member_positions])

            wgs84_cx=cluster["cLon"]
            wgs84_cy=cluster["cLat"]
            
            utm_cx, utm_cy = convert_wgs84_to_utm(wgs84_cx/1e7, wgs84_cy/1e7)
            utm_cx, utm_cy = wgs84_cx, wgs84_cy  #novo
            
            parsed_clusters.append({
                "id": cluster_id,
                "center": {"x": utm_cx, "y": utm_cy},
                "members": members,
                "approached_by_car": 0
            })
            #print("CLUSTER PARSED",parsed_clusters)
        except Exception as e:
            print(f"[WARN] Invalid cluster: {cluster} ({e})")

    # --- Parse cars ---
    for car_msg in cars_list:
        try:
            car_id = int(car_msg["id"])
            class_id = str(car_msg.get("classID", ""))
            x = float(car_msg["position"]["x"])
            y = float(car_msg["position"]["y"])
            time_stamp = float(car_msg.get("time", 0.0))

            parsed_cars.append({
                "id": car_id,
                "classID": class_id,
                "position": {"x": x, "y": y},
                "time": time_stamp,
                "approaching_cluster": 0
            })
        except Exception as e:
            print(f"[WARN] Invalid car message: {car_msg} ({e})")

    # --- Compare each car vs. each cluster ---
    for car in parsed_cars:
        car_id = car['id']
        car_pos = car['position']

        for cluster in parsed_clusters:
            cluster_id = cluster["id"]
            cluster_pos = cluster["center"]
            current_distance = euclidean_distance(car_pos, cluster_pos)
            key = (car_id, cluster_id)

            # Initialize tracking
            if key not in tracking_state:
                tracking_state[key] = {
                    'distances': deque(maxlen=history_len),
                    'reductions': deque(maxlen=history_len),
                    'previous_distance': current_distance + 1
                }

            state = tracking_state[key]
            state['distances'].append(current_distance)

            # Compute smoothed distance change
            if len(state['distances']) > 1:
                reduction = state['distances'][-2] - state['distances'][-1]
                state['reductions'].append(reduction)
                avg_prev = np.mean(list(state['distances'])[:-1])
            else:
                avg_prev = current_distance

            avg_current = np.mean(state['distances'])
            #print("AAAAAA", avg_current,avg_prev)
            # Check approach condition
            if round(avg_current,2) < round(avg_prev + DISTANCE_MARGIN_ADD_UP,2):
                car['approaching_cluster'] = 1
                cluster['approached_by_car'] = 1
            else:
                car['approaching_cluster'] = 0

            state['previous_distance'] = current_distance

    return parsed_cars, parsed_clusters



def split_by_classId(objects):
    """
    Split tracked objects into vehicles and pedestrians lists.

    Args:
        objects (dict): mapping id -> object info dict
        vehicle_class (int): classId representing vehicles
        pedestrian_class (int): classId representing pedestrians

    Returns:
        (list, list): (vehicles, pedestrians)
    """
 
    vehicles = []
    pedestrians = []

    for obj in objects.values():
        if obj.get("classID") == carClass:
            vehicles.append(obj)
        elif obj.get("classID") == pedClass:
            pedestrians.append(obj)
    #print("OOOOOOOO", objects, "\n", pedestrians, vehicles)
    return pedestrians, vehicles

#Check if a string is contained in a deque buffer.
# Return True if query_str is in buffer, False otherwise
def is_in_buffer(buf, query_str):
    return query_str in buf

def cpm_to_shared_dict(cpm):
    """
    Convert CPM dict to shared dict format.
    Directly maps CPM fields without artificial conversions.
    """
    shared = {}
    timestamp = cpm.get("timestamp", 0)
    updateTime = cpm.get("updateTime", 0)
    detectionTime = cpm.get("detectionTime", 0)
    
    #w_x,w_y=convert_utm_to_wgs84(550094, 4601274)
    #u_x, u_y = convert_wgs84_to_utm(w_x,w_y)
    #print("UUUU", 550094, 4601274, u_x, u_y)
    
    for obj in cpm.get("objects", []):
        obj_id = int(obj.get("objectID", -1))

        wgs84_x = obj.get("lon")
        wgs84_y = obj.get("lat")
        
        utm_x, utm_y = convert_wgs84_to_utm(wgs84_x/1e7, wgs84_y/1e7)
        utm_x, utm_y = wgs84_x, wgs84_y  #novo
        azimuth = np.float64(obj.get("heading", 0))
        speed = obj.get("speed", 0)

        shared[obj_id] = {
            "id": obj_id,
            #"id2": f"{x},{y},{timestamp}",
            "classID": obj.get("classID"),
            #"position": {"x": int(round(utm_x,0)), "y": int(round(utm_y,0))},
            "position": {"x": utm_x, "y": utm_y}, #novo
            "azimuth": azimuth,
            "speed": speed,
            "time": timestamp,
            "updateTime": updateTime,
            "detectionTime": detectionTime
        }

    return shared



#returns only the IDs of flagged cars
def find_red_flagged_objs(cpm):

#[SHARED DICT] {6: {'id': 6, 'id2': "550099,4601276,1.667", 'classId': 2, 'position': {'x': 550099, 'y': 4601276}, 'azimuth': np.float64(191.62), 'speed': 5.2, 'time': 1.67}, 
#               4: {'id': 4, 'id2': "550090,4601275,1.667", 'classId': 0, 'position': {'x': 550090, 'y': 4601275}, 'azimuth': np.float64(218.87), 'speed': 2.1, 'time': 1.67}}
    sharedDict = cpm_to_shared_dict(cpm)
    #print("ECCCCCCcpm   ", cpm); print("DDDDDDDshared", sharedDict)

    peds_list, cars_list = split_by_classId(sharedDict)

    cars_flagged, peds_flagged = check_approaching_objects(peds_list, cars_list, tracking_state)
    
    t=nowTime_ms_mod()
    
    car_red_flagged_ids = []
    ped_red_flagged_ids = []

    for car in cars_flagged:
        if car['approaching_pedestrian'] == 1:

            print(t, f"🔴🚗 {car['id']}",flush=True)
            car_red_flagged_ids.append(car['id'])
        else:
            print(t, f"🟢🚗 {car['id']}",flush=True)
           


    for ped in peds_flagged:
        if ped['approaching_car'] == 1:
            if PRINT_PED_ICON: print(t, f"🔴🧍 {ped['id']}",flush=True)
            ped_red_flagged_ids.append(ped['id'])
        else:
            if PRINT_PED_ICON: print(t, f"🟢🧍 {ped['id']}",flush=True)

    #print("RRRRRR", car_red_flagged_ids, ped_red_flagged_ids)
    return car_red_flagged_ids, ped_red_flagged_ids


def find_red_flagged_clusters(cpm):
    """
    Detects cars approaching pedestrian clusters using CPM data.
    Returns two lists: car IDs and cluster IDs flagged as 'approaching'.
    """

    sharedDict = cpm_to_shared_dict(cpm)

    # Split objects by class type
    peds_list, cars_list = split_by_classId(sharedDict)

    # Build clusters (they should already be part of the CPM if clustering is done earlier)
    clusters_list = cpm.get("clusters", [])

    # Compute approach events between cars and clusters
    cars_flagged, clusters_flagged = check_approaching_clusters(clusters_list, cars_list, tracking_state)

    #print("CLUSTER FLAGS:", cars_flagged, clusters_flagged)

    t = nowTime_ms_mod()
    car_red_flagged_ids = []
    cluster_red_flagged_ids = []

    # --- Print and collect car alerts ---
    for car in cars_flagged:
        if car.get('approaching_cluster', 0) == 1:
            print(t, f"🔴🚗 Car {car['id']} approaching cluster", flush=True)
            car_red_flagged_ids.append(car['id'])
        else:
            print(t, f"🟢🚗 Car {car['id']} safe regarding cluster", flush=True)

    # --- Print and collect cluster alerts ---
    for cluster in clusters_flagged:
        if cluster.get('approached_by_car', 0) == 1:
            print(t, f"🔴👥 Cluster {cluster['id']} approached by car", flush=True)
            cluster_red_flagged_ids.append(cluster['id'])
        else:
            print(t, f"🟢👥 Cluster {cluster['id']} safe", flush=True)

    return car_red_flagged_ids, cluster_red_flagged_ids



##############################

clicked_points = []
current_cursor = [0, 0]  # will store current mouse position
next_point=0; ready=0

def getPoint():
    global next_point, ready
    time.sleep(1 if next_point > -1 else 0)
    os.system('clear')
    next_point += 1
    print(f"{NR_TOT_POINTS+1-next_point} - choose point at:")

    if USE_CASE == 1:
    #dict of descriptions
     descriptions = {
        1: "grelha de drenagem esq., canto mais afastado",
        2: "banco no passeio, aresta mais afastada",
        3: "grelha de drenagem dir., canto mais afastado", #"tampa de águas pluviais",
        4: "poste de sinal de trânsito",
        5: "poste de luz",
        6: "estrutura da tampa no fim da berma dir., aresta mais afastada",
        7: "marco castanho em frente da escola de direiro",
        8: "poste do ponto de encontro"
     }


    if USE_CASE == 2:
     descriptions = {
        1: "tampo das águas pluviais no passeio do cruzamento",
        2: "primeiro poste de luz",
        3: "grelha de drenagem no passeio esq. do cruzamento, perto do mapa",
        4: "banco mais próximo no passeio esq., vértice mais afastado",
        5: "tampo das águas pluviais no meio da estrada, o mais afastado",
        6: "grelha de drenagem no passeio dir.",
        7: "grelha de drenagem no passeio esq, oposta à do passeio dir.",
        8: "banco mais distante no passeio esq., vértice mais afastado",
        9: "balde do lixo no fundo do passeio esq."
     }


    if USE_CASE == 3:
     descriptions = {
        1: "ponto 1 dir.",
        2: "ponto 2 dir.",
        3: "ponto 3 dir.",
        4: "ponto 1 esq.",
        5: "ponto 2 esq.",
        6: "ponto 3 esq.",
     }
    
    #print("size", len(descriptions))
    
    if next_point in descriptions:
        print(descriptions[next_point])
    else:
        print("All points collected !!!")
        ready = 1   # <-- sets global flag
    return ready


def mouse_move(event, x, y, flags, param):
    global current_cursor, clicked_points, ready
    if event == cv2.EVENT_MOUSEMOVE:
        current_cursor = [x, y]
    elif event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinate: ({x}, {y})")
        #x+=550000; y+=4600000
        clicked_points.append((x, y))
        getPoint()   # updates global ready if needed


def getFrame(frame):
    global ready
    try:
        if frame is None:
            print("No frame provided.")
            return
        #stringg="Clicar para selecionar ponto. Terminar recolha com a tecla 'q'"
        stringg="CLICK WITH MOUSE AT A POINT"
        cv2.imshow(stringg, frame)
        cv2.setMouseCallback(stringg, mouse_move)

        #print("Pressione 's' para salvar coordenadas do cursor.")
        
        #os.system('clear')
        #print("👉 ALL POINTS MUST BE DIFFERENT !!!")
        #time.sleep(1)
        
        getPoint()

        while True:
            
            key = cv2.waitKey(1) & 0xFF

            # 's' → save current cursor coordinates
            if key == ord('s'):
                clicked_points.append(tuple(current_cursor))
                print(f"Saved cursor coordinates: {tuple(current_cursor)}")
                getPoint()
                

            # 'q' → quit
            elif key == ord('q'):
                print("\Clicked points:")
                for i, pt in enumerate(clicked_points):
                    print(f"Point {i}: {pt}")
                break
                
            if ready:
                os.system('clear')
                print("All points collected.")
                time.sleep(1)
                break
          

    except KeyboardInterrupt:
        print("Shutting down frame selection...")
        sys.exit(0)
    finally:
        cv2.destroyAllWindows()

    # After exiting, build the numpy array
    image_points = np.array(clicked_points, dtype=np.float32)
    size=int(image_points.size/2)
    print("Nr. points =", size)
    print("\n", image_points)
    time.sleep(2)
    if size != NR_TOT_POINTS:
       print("\nERROR: nr. points not ok !!!!") 
       exit(0)
    #nr_dups = find_duplicate_points()
    #print("DDDDDdddDDDD", nr_dups, flush=True)
    #if nr_dups != 0:
    #   print(f"\nERROR: {nr_dups} duplicated points !!!!", flush=True) 
    #   exit(0)
    return image_points

               
def find_duplicate_points(image_points):
    """
    Detects duplicate (x, y) points in a NumPy array.

    Parameters:
        image_points (np.ndarray): Array of shape (N, 2) with float or int coordinates.

    Returns:
        list of tuples: List of duplicate points found.
    """
    # Convert to tuples for hashable comparison
    points_as_tuples = [tuple(pt) for pt in image_points]
    
    seen = set()
    duplicates = set()
    
    for pt in points_as_tuples:
        if pt in seen:
            duplicates.add(pt)
        else:
            seen.add(pt)
    print(duplicates)
    #return list(duplicates)
    return len(duplicates)


def getFrame_savedVideo(frame_number):
    # Abre o vídeo
    video = cv2.VideoCapture(VIDEO_FILE)

    # Posiciona no frame desejado
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = video.read()
    video.release()

    if not ret:
        print("Não foi possível ler o frame do vídeo.")
        return None
    return frame

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordenadas do clique: ({x}, {y})")


def extract_frames(video_path, t1, t2, output_folder="frames"):
    """
    Extrai todas as frames de um vídeo entre os tempos t1 e t2 (em segundos).

    Args:
        video_path (str): caminho para o ficheiro de vídeo.
        t1 (float): tempo inicial em segundos.
        t2 (float): tempo final em segundos.
        output_folder (str): pasta para guardar as frames.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(t1 * fps)
    end_frame = int(t2 * fps)

    if start_frame >= total_frames or end_frame > total_frames:
        raise ValueError("Os tempos indicados estão fora da duração do vídeo.")

    # Vai diretamente para o frame inicial
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    saved = 0

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Guarda cada frame como imagem
        filename = f"{output_folder}/frame_{frame_idx}.jpg"
        cv2.imwrite(filename, frame)
        saved += 1

        frame_idx += 1

    cap.release()
    print(f"[INFO] Guardadas {saved} frames em '{output_folder}'")


def get_x_y(frameNr, video):

  if video == "2peds2carsVideo":
    data = {
     1: (481, 324),  2: (483, 327),  3: (485, 330),  4: (488, 332),  5: (490, 335),  6: (492, 338),  7: (495, 341),  8: (497, 344),  9: (499, 347),  10: (502, 349),  11: (504, 352),  12: (506, 355),  13: (509, 358),  14: (511, 361),  15: (513, 364),  16: (516, 366),  17: (518, 369),  18: (520, 372),  19: (523, 375),  20: (525, 378),  21: (527, 381),  22: (530, 383),  23: (532, 386),  24: (534, 389),  25: (537, 392),  26: (539, 395),  27: (541, 398),  28: (544, 400),  29: (546, 403),  30: (548, 406),  31: (551, 409),  32: (555, 417),

 32: (555, 417), 33: (558, 422), 34: (563, 430), 35: (570, 438), 36: (574, 447), 37: (582, 457), 38: (593, 469), 39: (606, 479), 40: (620, 495), 41: (631, 509), 42: (645, 527), 43: (660, 548), 44: (679, 574), 45: (701, 605), 46: (732, 651), 47: (776, 694), 48: (837, 756), 49: (902, 783), 50: (960, 792), 51: (979, 796), 52: (992, 796), 53: (1022, 796), 54: (1049, 796), 55: (1069, 808),   

    56: (527, 381),     57: (530, 383),     58: (532, 386),     59: (534, 389),     60: (537, 392),     61: (539, 395),     62: (541, 398),     63: (544, 400),     64: (546, 403),     65: (548, 406),     66: (551, 409),     67: (555, 417),
   
  68: (561, 419), 69: (567, 426), 70: (572, 433), 71: (579, 441), 72: (587, 448), 73: (597, 458), 74: (607, 467), 75: (617, 480), 76: (628, 493), 77: (643, 510), 78: (662, 532), 79: (681, 552), 80: (698, 575), 81: (728, 610), 82: (755, 646), 83: (804, 701), 84: (856, 758), 85: (925, 783), 86: (958, 794), 87: (981, 800), 88: (996, 801), 89: (1027, 800), 90: (1048, 798), 91: (1072, 795), 92: (1091, 806),
 }
 
  if video == "2peds1carVideo":
   data = {
1: (490, 320), 2: (492, 323), 3: (494, 326), 4: (496, 329),
5: (498, 332),
6: (500, 335),
7: (502, 338),
8: (504, 341),
9: (506, 344),
10: (508, 347),
11: (510, 350),
12: (512, 353),
13: (514, 356),
14: (516, 359),
15: (518, 362),
16: (520, 365),
17: (522, 368),
18: (524, 371),
19: (526, 374),
20: (528, 377),
21: (530, 380),
22: (532, 383),
23: (534, 386),
24: (536, 389),
25: (538, 392),
26: (540, 395),
27: (542, 398),
28: (544, 401),
29: (546, 404),
30: (548, 407),
31: (551, 410),
32: (553, 413),
33: (555, 416),
34: (557, 419),
35: (559, 422),
36: (561, 425),
37: (563, 428),
38: (565, 461),
39: (581, 475),
40: (577, 495),
41: (590, 516),
42: (599, 528),
43: (711, 608),

44: (820, 615),
45: (822, 680),
46: (947, 778),
47: (960, 788),
48: (973, 798),
49: (1004, 808),
50: (1027, 807),
51: (1046, 804),
52: (1084, 829)
}

 
 
 
  x, y = data.get(frameNr, (0, 0))  # If frameNr exists → unpack x,y; else return (0,0)
  return x, y #date[frameNr].[0], date[frameNr].[1]


###################################

def start_yolo_process(shared_dict, stop_event):
    process = None
    try:
        while not stop_event.is_set():
            print("Starting YOLO process ...")
            process = Process(
                target=run_tracking_and_share_recent_and_show_video_utm,
                args=("yolov10n.pt", shared_dict),
                kwargs={"detection_threshold": 0.4}
            )
            process.start()
            print("YOLO process up")

            # Wait for the process to finish
            while process.is_alive() and not stop_event.is_set():
                time.sleep(0.1)

            if process.is_alive():
                process.terminate()
                process.join()

            print("YOLO process ended. Restarting in 1 second...")
            time.sleep(1)
            if not RESTART_AFTER_END:
               sys.exit(0)

    except KeyboardInterrupt:
        print("\n[INFO] CTRL+C detected, stopping YOLO process...")
        #cleanup()
        stop_event.set()
        if process is not None and process.is_alive():
            process.terminate()
            process.join()

#usar subprocesso
if RESTART_AFTER_END:
 if __name__ == "__main__":
    print("Starting Manager ...")
    manager = Manager()
    shared_dict = manager.dict()
    stop_event = Event()

    # Optional: frame point selection
    if GET_POINTS_FROM_FRAME:
        try:
            FRAME_NR = 120  # frame number to use
            image = getFrame_savedVideo(FRAME_NR)
            if image is not None:
                cv2.imshow("Clique para selecionar ponto", image)
                cv2.setMouseCallback("Clique para selecionar ponto", click_event)

                while True:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            print("Shutting down frame selection...")
            sys.exit(0)
        finally:
            cv2.destroyAllWindows()

    # Start YOLO process in main loop
    try:
        start_yolo_process(shared_dict, stop_event)

    except KeyboardInterrupt:
        print("\n[INFO] Main CTRL+C detected, stopping everything...")
        stop_event.set()

    print("[INFO] Program finished.")
    


################################################

#usar main process
if not RESTART_AFTER_END:
 if __name__ == "__main__":
    #extract_frames("tiempos_nuevos.mp4", t1=45, t2=55, output_folder="output_frames"); exit()
    print("Starting Manager ...")
    manager = Manager()
    shared_dict = manager.dict()
    shared_fifo_data = manager.list()
    stop_event = Event()

    # Optional: point selection from a frame
    if 0: #GET_POINTS_FROM_FRAME and USE_SAVED_VIDEO:
        try:
            FRAME_NR = 120  # frame number to select
            image = getFrame_savedVideo(FRAME_NR)
            if image is not None:
                cv2.imshow("Clique para selecionar ponto", image)
                cv2.setMouseCallback("Clique para selecionar ponto", click_event)
                
                while True:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        except KeyboardInterrupt:
            print("[INFO] Shutting down...")
            sys.exit(0)
        finally:
            cv2.destroyAllWindows()

    # Run YOLO tracking directly in main process
    try:
        print("[INFO] Starting YOLO tracking in main process...")
        run_tracking_and_share_recent_and_show_video_utm("yolov10n.pt", shared_dict, detection_threshold=0.4)

    except KeyboardInterrupt:
        print("\n[INFO] Main CTRL+C detected, stopping...")
        stop_event.set()
        cv2.destroyAllWindows()

    print("[INFO] Program finished.")
    if CALCULATE_STATS:
       process_log_file(LOG_FILE)




# linux:  v4l2-ctl --list-formats-ext
# Test resolutions and fps based on your v4l2-ctl output
resolutions = [
    (1280, 720),
    (960, 540),
    (848, 480),
    (640, 480),
    (640, 360),
    (424, 240),
    (320, 240),
    (320, 180),
]

fps_options = [15, 30]

def test_camera(device=0):
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    working_modes = []

    for w, h in resolutions:
        for fps in fps_options:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, fps)

            # Read one frame to verify
            ret, frame = cap.read()
            if not ret:
                continue

            # Get what the camera actually set
            actual_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)

            # Only count if OpenCV reports what we requested (or close)
            if abs(actual_w - w) <= 10 and abs(actual_h - h) <= 10:
                working_modes.append((actual_w, actual_h, round(actual_fps)))
                print(f"[OK] {w}x{h} @ {fps}fps -> camera reports {actual_w}x{actual_h} @ {actual_fps:.1f}fps")
            else:
                print(f"[FAIL] {w}x{h} @ {fps}fps not supported")

    cap.release()
    return working_modes


if 0: #__name__ == "__main__":
    modes = test_camera(0)  # device 0 (default webcam)
    print("\n=== Working modes detected ===")
    for m in modes:
        print(f"{m[0]}x{m[1]} @ {m[2]} fps")








if 0: #__name__ == "__main__": 
  print("Starting Manager ...")
  manager = Manager()
  print("Manager up")
  shared_dict = manager.dict()
  shared_fifo_data = manager.list()   # <-- também se pode usar para UNIX socket 
  # Cria process do tracking
  process = Process(
    #target=run_tracking_and_share_recent,
    target=run_tracking_and_share_recent_and_show_video_utm,
    args=("yolov10n.pt", shared_dict),
    kwargs={"detection_threshold": 0.4}
  )


  
  
  print("Starting Yolo process ...")
  process.start()
  print("Yolo process up")

  # Cria thread do UNIX socket reader

  if USE_UNIX_SOCKETS:
    socket_path = UNIX_SOCKET_PATH
    print(f"[MAIN] Launching unix_socket_reader on {socket_path}...")
    fifo_thread = threading.Thread(target=unix_socket_reader, args=(socket_path, shared_fifo_data))
    fifo_thread.start()
    #usa-se args=(client_socket,) quando já tenho um socket criado fora, e quero-o passar para uma função em thread,
  
    outgoing_queue = manager.list()
    threading.Thread(target=send_messages_to_server, daemon=True).start()


  # Loop principal
  try:
   #GREEN = '\033[92m'
   
    while True:  # <-- keep running until Ctrl-C
            # Example: just show the shared_dict contents
            print("Shared objects:", dict(shared_dict), flush=True)
            time.sleep(1)
            
            
  except KeyboardInterrupt:
    process.terminate()
    process.join()

#YOLOv10-N: Nano version for extremely resource-constrained environments.
#YOLOv10-S: Small version balancing speed and accuracy.
#YOLOv10-M: Medium version for general-purpose use.
#YOLOv10-B: Balanced version with increased width for higher accuracy.
#YOLOv10-L: Large version for higher accuracy at the cost of increased computational resources.
#YOLOv10-X: Extra-large version for maximum accuracy and performance.



#Real-Time Object Tracking using YOLO10 and DeepSORT 
#https://www.youtube.com/watch?v=7HGfEPC6pf4

#https://www.memorycow.co.uk/digital-camera/samsung/e-series/samsung-es17-digital-camera


#Found existing installation: typing_extensions 4.12.2
#    Uninstalling typing_extensions-4.12.2:
#    Uninstalling numpy-1.24.4:

#https://www.youtube.com/watch?v=Jx6oLBfDxRo&t=5s

# https://medium.com/@Mert.A/how-to-use-yolov9-for-object-detection-93598ad88d7d

#find . -name "*.py" -exec grep "Speed" {} \; -print

# grep -E '(events":\[{"eventID.*){2,}' SAVED_LOG.txt  #matches lines that contain the string at least twice

#video
#{'min': 0.0065004825592041016, 'max': 0.26665806770324707, 'avg': 0.1175774131809269, 'count': 222}
#{'min': 0.005774259567260742, 'max': 0.2785003185272217, 'avg': 0.1171982309839747, 'count': 222}

#realsense
#{'min': 0.1652510166168213, 'max': 0.24361109733581543, 'avg': 0.19094588807834093, 'count': 186}

#uvc
#{'min': 0.16796636581420898, 'max': 0.24310874938964844, 'avg': 0.19816632550131827, 'count': 239}

# 1,80 m comprimento do cabo do uvc

#https://www.scribd.com/document/495646133/A-Menina-Gotinha-de-Agua

# https://www.olx.pt/d/anuncio/serras-de-valongo-estudo-de-geomorfologia-fernando-rebelo-IDIpSc3.html

# https://pt-pt.topographic-map.com/

