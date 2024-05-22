import cv2
import numpy as np

from filterpy.kalman import KalmanFilter

class KalmanTracker:
    
    count:int = 0
    
    def __init__(self, bounding_box):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])
        
        self.kf.R *= 10.
        self.kf.P *= 10.
        self.kf.Q *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bounding_box[:4])
        self.confident = bounding_box[-2]
        self.cls = bounding_box[-1]

        self.time_since_update = 0
        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))
        
    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]
        
    def get_state(self):
        return np.hstack((self.convert_x_to_bbox(self.kf.x), [[self.confident]], [[self.cls]]))

    @staticmethod
    def convert_bbox_to_z(bbox):
        """
        Convert a bounding box in the form [x1, y1, x2, y2] to [cx, cy, s, r]
        where cx, cy is the center of the box, s is the scale/area, and r is the
        aspect ratio.
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))
    
    @staticmethod
    def convert_x_to_bbox(x):
        """
        Convert the bounding box from [cx, cy, s, r] to [x1, y1, x2, y2]
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w

        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))