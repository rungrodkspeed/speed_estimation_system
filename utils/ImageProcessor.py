import cv2
import torch
import numpy as np

from functools import reduce
from typing import Any, Callable, Tuple, TypeVar

T = TypeVar('T')

class ImageProcessor:
    
    def __init__(self):
        self.tasks = ["preprocess_fromfile", "preprocess", "postprocess", "draw"]
        
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

        call_task = self.task_map(task)
        outs = call_task(source)
        return outs

    def pipe(self, data: T, *functions: Callable[[Any], Any]) -> Any:
        return reduce(lambda x, f: f(x), functions, data)

    def task_map(self, task: str) -> Callable:
        return {
            "preprocess": self.preprocess,
            "postprocess": self.postprocess,
            "draw": self.draw
        }[task]
        
    def preprocess(self, im : np.ndarray) -> torch.tensor:
        tensor = self.pipe(im, self.resize, self.bgr_to_rgb, self.image_to_tensor)
        tensor = np.ascontiguousarray(tensor)
        tensor = tensor.astype(np.float32)
        tensor /= 255.
        return tensor
        
    def resize(self, im: np.ndarray, new_shape: Tuple[int] = (640, 640), stride: int = 32) -> np.ndarray:

        shape = im.shape[:2]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder( im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114) )
        
        return im

    def bgr_to_rgb(self, im: np.ndarray) -> np.ndarray:
        return im[..., ::-1]

    def image_to_tensor(self, im: np.ndarray) -> np.ndarray:
        return im.transpose(2, 0, 1)
    
    def postprocess(self):
        print("postprocess")
    
    def draw(self):
        print("draw")