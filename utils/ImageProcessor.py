import cv2
import time
import numpy as np

from functools import reduce, partial
from typing import Any, Callable, Tuple, TypeVar, Optional, List

T = TypeVar('T')

class ImageProcessor:
    
    shape: Optional[Tuple[int]]
    preprocess_shape: Optional[Tuple[int]]
    
    def __init__(self):
        self.tasks = ["preprocess", "postprocess", "draw"]
        self.shape = None
        self.new_unpad = None
        
    def validate_inputs(self, sources, task) -> None:
        
        if not isinstance(sources, np.ndarray):
            raise TypeError("sources must be numpy array")
        
        if isinstance(task, str):
            if task not in self.tasks:
                raise ValueError(f"not have task: {task}")
        else:
            raise TypeError("task must be str")
        
    def __call__(self, source: np.ndarray, task: str) -> Any:
        
        self.validate_inputs(source, task)

        impl = self.task_map(task)
        outs = impl(source)
        return outs

    def task_map(self, task: str) -> Callable:
        return {
            "preprocess": self.preprocess,
            "postprocess": self.postprocess,
            "draw": self.draw
        }[task]
        
    def pipe(self, data: T, *functions: Callable[[Any], Any]) -> Any:
        return reduce(lambda x, f: f(x), functions, data)
        
    def preprocess(self, im: np.ndarray) -> np.ndarray:
        tensor = self.pipe(im, self.resize, self.bgr_to_rgb, self.image_to_tensor)
        tensor = np.ascontiguousarray(tensor)
        tensor = tensor.astype(np.float32)
        tensor /= 255.
        return tensor
        
    def resize(self, im: np.ndarray, new_shape: Tuple[int] = (640, 640), stride: int = 32) -> np.ndarray:

        self.shape = im.shape[:2]

        r = min(new_shape[0] / self.shape[0], new_shape[1] / self.shape[1])
        new_unpad = int(round(self.shape[1] * r)), int(round(self.shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

        dw /= 2
        dh /= 2

        if self.shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder( im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114) )
        
        self.preprocess_shape = im.shape[:2]
        
        return im

    def bgr_to_rgb(self, im: np.ndarray) -> np.ndarray:
        return im[..., ::-1]

    def image_to_tensor(self, im: np.ndarray) -> np.ndarray:
        return im.transpose(2, 0, 1)
    
    #boxes, confident, classes
    def postprocess(self, preds: np.ndarray) -> List[np.ndarray]:
                
        assert (self.shape is not None) and (self.preprocess_shape is not None), "Maybe must preprocess before using postprocess" 
        
        custom_nms = partial(self.non_max_suppression, conf_thres=0.2, classes=[2], max_nms=100, max_wh=5040)
        return self.pipe(preds, custom_nms, self.impl_post)
    
    def impl_post(self, preds: List[np.ndarray]) -> List[np.ndarray]:
        res = []
        for pred in preds:
            pred[:, :4] = self.scale_boxes(self.preprocess_shape, pred[:, :4], self.shape)
            res.append(pred)
        return res
    
    def xywh2xyxy(self, x):
        
        y = np.empty_like(x)
        dw = x[..., 2] / 2
        dh = x[..., 3] / 2
        y[..., 0] = x[..., 0] - dw
        y[..., 1] = x[..., 1] - dh
        y[..., 2] = x[..., 0] + dw
        y[..., 3] = x[..., 1] + dh
        return y

    def nms(self, boxes, scores, iou_threshold):

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def non_max_suppression(
        self,
        prediction,
        conf_thres=0.1,
        iou_thres=0.7,
        classes=None,
        max_det=300,
        nc=0,
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
    ):

        assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

        if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        bs = prediction.shape[0]  # batch size
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = prediction[:, 4:mi].max(1) > conf_thres  # candidates

        # Settings
        time_limit = 2.0 + max_time_img * bs  # seconds to quit after

        prediction = np.transpose(prediction, (0, 2, 1))  # shape (1, 84, 6300) to shape (1, 6300, 84)

        prediction[..., :4] = self.xywh2xyxy(prediction[..., :4])

        output = [np.zeros((0, 6 + nm)) for _ in range(bs)]
        start_time = time.time()
        
        for xi, x in enumerate(prediction):  # image index, image inference
            if not np.any(xc[xi]):
                continue

            # Apply constraints
            x = x[xc[xi]]  # confidence

            if not x.shape[0]:
                continue

            box, cls, mask = np.split(x, (4, mi), axis=1)

            conf = np.max(cls, axis=1, keepdims=True)
            j = np.argmax(cls, axis=1, keepdims=True)
            x = np.concatenate((box, conf, j.astype(float), mask), axis=1)[conf.flatten() > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[np.isin(x[:, 5], classes)]

            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            if n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * max_wh  # classes
            boxes = x[:, :4] + c  # boxes (offset by class)
            scores = x[:, 4]  # scores

            keep = self.nms(boxes, scores, iou_thres)
            keep = keep[:max_det]  # limit detections

            output[xi] = x[keep]

            # Check time limit
            if time.time() - start_time > time_limit:
                break

        return output

    def clip_boxes(self, boxes, shape):
        
        boxes[:, 0] = np.clip(boxes[:, 0], 0, shape[1])  # x1
        boxes[:, 1] = np.clip(boxes[:, 1], 0, shape[0])  # y1
        boxes[:, 2] = np.clip(boxes[:, 2], 0, shape[1])  # x2
        boxes[:, 3] = np.clip(boxes[:, 3], 0, shape[0])  # y2
        return boxes

    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
        if ratio_pad is None:
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            pad = (
                round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
                round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
            )
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        if padding:
            boxes[..., 0] -= pad[0]
            boxes[..., 1] -= pad[1]
            if not xywh:
                boxes[..., 2] -= pad[0]
                boxes[..., 3] -= pad[1]

        boxes[..., :4] /= gain
        return self.clip_boxes(boxes, img0_shape)

    
    def draw(self):
        print("draw")