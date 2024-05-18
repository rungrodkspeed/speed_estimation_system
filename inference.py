import cv2
import torch

from utils import ImageProcessor

ip = ImageProcessor(device="cuda")

image_path = "testcase_0.jpg"
im = cv2.imread(image_path)

ckpt = torch.load("yolov8n.pt")
model = ckpt["model"].float().to("cuda").eval()

tensor = ip(im, task="preprocess")

print(tensor.shape)

res = model(tensor)

print(res.shape)