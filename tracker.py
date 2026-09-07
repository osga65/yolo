from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker 
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np
import math


class Tracker:
    tracker = None
    encoder = None
    tracks  = None

    def __init__(self): #constructor
        max_cosine_distance = 0.4
        nn_budget = None

        encoder_model_filename = 'model_data/mars-small128.pb'

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)



    def update(self, frame, detections):
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])  
            self.update_tracks()
            return

        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]

        features = self.encoder(frame, bboxes)

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()

    def update_tracks(self):
        tracks = []
#for each detected object (i.e., each confirmed track in `self.tracker.tracks`), a new instance of the `Track` class is created
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()

            id = track.track_id

            # Create or update the Track object with speed calculation 
            tracks.append(Track(id, bbox, track))
            #tracks.append(Track(id, bbox)) #original
        self.tracks = tracks


class Track:
    track_id = None
    bbox = None
    speed = 0.0
    center = [None,None]
    prev_center = [None, None]
    state_dict = {} #o state já é um atributo do track_obj 
    mavg_speed = 0.0
    velocity=[0]*10


    # constructor
    def __init__(self, id, bbox, track_obj): 
        self.track_id = id #originalmente só tinha este
        self.bbox = bbox #e este. Argumento não tinha o track_obj

        self.path = [] # Store the centers to draw the path
        #print("TTTTT", self.track_id)
        #print("PPPPP",track_obj)
        # Calculate current center of the bounding box
        self.center = self.calculate_center(bbox, id)
        # Store the current center in the path list for drawing the path 
        self.path += self.center #self.path.append(self.center)
        #print("UUUUUU",id, self.path)

        if not hasattr(track_obj, 'state_dict'): track_obj.state_dict = {} 
        # Initialize state_dict as an empty dictionary if not present
        size=10
        if not hasattr(track_obj, 'speed'): track_obj.speed = [0]*size

        if not hasattr(track_obj, 'path'): track_obj.path = []
        else:  
           self.path = self.add_path_to_buffer(self.path, track_obj)
           #self.path = self.check_and_clean_wrong_path(self.path, track_obj)

        if track_obj.state_dict.get('previous_center') is not None: self.prev_center = track_obj.state_dict['previous_center']
        else: track_obj.state_dict['previous_center'] = None

        # If previous center exists, calculate speed 
        if track_obj.state_dict['previous_center'] is not None: 
            self.prev_center = track_obj.state_dict['previous_center'] 
            self.distance = self.calculate_distance(self.center, self.prev_center)
            self.speed = self.calculate_speed(self.distance)
            self.save_speed_in_buffer(self.speed, track_obj)
            self.mavg_speed = self.calculate_moving_avg_speed(id, track_obj)
        else: self.speed = 0.0 
        # Update the track object's state_dict to store the current center for future calculations 
        track_obj.state_dict['previous_center'] = self.center 

    def calculate_center(self, bbox, ident):
       """ Calculate the center point of the bounding box """ 
       x1, y1, x2, y2 = bbox 
       center_x = (x1 + x2) / 2 
       center_y = (y1 + y2) / 2
       #if(ident!=-1): print ("BBBBB",center_x, center_y)
       return [center_x, center_y] 

    def calculate_distance(self, current_center, previous_center): 
       """ Calculate Euclidean distance between the previous and current center points """ 
       #print("PPPPP", current_center, previous_center)
       dx = current_center[0] - previous_center[0] 
       dy = current_center[1] - previous_center[1] 
       distance = math.sqrt(dx ** 2 + dy ** 2) 
       return distance 
       # Speed here is in pixel/frame, can be adjusted for real-world distance/time

    def calculate_speed(self, distance):
       meter_per_px = 0.026
       frame_rate = 24 #frames/s
       speed = distance*meter_per_px*frame_rate*3.6 #km/h
       return speed


    def save_speed_in_buffer(self, vel, track_obj):
       size=10 # alterar em velocity=[0]*size
       if track_obj.speed is not None:
              for i in range (size-1, 0,-1): 
                  track_obj.speed[i]=track_obj.speed[i-1]
              
              track_obj.speed[0] = vel if abs(vel) < 6.0 else 6.0*int(vel)/abs(int(vel))		
       else: 
              track_obj.speed = [0]*10       


    def calculate_moving_avg_speed(self,  ident, track_obj):
       size=10 # alterar em velocity=[0]*size
       for i in range (size-1, 0,-1): 
          #if ident != -1: print("BBBB",i, track_obj.speed[i])
          self.velocity[i]=self.velocity[i-1]
 
       #self.velocity[0]=vel #if int(vel)>-4 and int(vel)<4 else 4.0

       # window_average = round(sum(window) / window_size, 2)
       tot_speed = sum(track_obj.speed)
       mov_avg_speed=tot_speed/size
       return mov_avg_speed


    def add_path_to_buffer(self, path, track_obj):
       #if track_obj.path is not None:
       track_obj.path.append(path)	
       #else: track_obj.path = []
       #print("GGGGGGG", path, track_obj.path)
       return track_obj.path
              
    def check_and_clean_wrong_path(self, path, track_obj):  
       max_dist = 5
       for i in range(len(path)):
           if path[i][0] == None: continue
           point_x= path[i][0]
           #point_y= path[i][1]
           point_x_prev= path[i-1][0]
           #point_y_prev= path[i-1][1]
           if abs(point_x - point_x_prev) > max_dist:
                 path = track_obj.path
                 return path
       return path     












