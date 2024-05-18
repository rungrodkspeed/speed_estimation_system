import cv2
import atexit
import numpy as np

from utils import ImageProcessor
from engine import TrtEngine

inference_engine = TrtEngine("assets/yolov8n.plan")
ip = ImageProcessor()
atexit.register(inference_engine.clear)

image = cv2.imread("testcase_0.jpg")
preprocessed_im = ip(image, task="preprocess")
print(inference_engine, type(inference_engine))
print(preprocessed_im, preprocessed_im.shape)

res = inference_engine([preprocessed_im])

print(res, res.shape)
