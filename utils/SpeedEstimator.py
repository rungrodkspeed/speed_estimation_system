import cv2
import numpy as np

from typing import List, Tuple

class SpeedEstimator:
    
    def __init__(self, reg_pts: List[Tuple[int]] = [(20, 400), (1260, 400)], spdl_dist_thresh: int = 10):
        self.im0 = None
        self.annotator = None
        self.reg_pts = reg_pts
        self.spdl_dist_thresh = spdl_dist_thresh
        
    
    