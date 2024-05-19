import gc
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

import pycuda.autoinit

from math import ceil
from typing import Optional, List, Union

class TrtEngine:
    
    buffer: Optional[List[cuda.DeviceAllocation]]
    context: Optional[trt.tensorrt.IExecutionContext]
    host_output: Optional[np.ndarray]
    
    def __init__(self, path: str):
        
        self.LOGGER = trt.Logger()
        self.buffer = None
        self.context = None
        self.rf = 32
        self.ppc = 21
        self.engine = self.load_engine(path)
        self.context = self.engine.create_execution_context()
        
        self.host_output = None
        
    def __call__(self, images: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        assert not (self.context is None), "Context is None."
        
        if isinstance(images, list):
            batch = len(images)
            c, h, w = images[0].shape
            input_shape = (batch, c, h, w)
            input_dtype = images[0].dtype
        else:
            input_shape = images.shape
            input_dtype = images.dtype
        
        host_pinned = cuda.pagelocked_empty(input_shape, dtype=input_dtype)
        host_pinned[:] = images
        
        self.binding_data(host_pinned)
        self.context.execute_v2(bindings=self.buffer)
        
        cuda.memcpy_dtoh_async(self.host_output, self.buffer[-1])
        
        return self.host_output
        
    def load_engine(self, path: str):
        
        with open(path, "rb") as f, trt.Runtime(self.LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            
        return engine
        
    def binding_data(self, data: Union[List[np.ndarray], np.ndarray]):
        
        if isinstance(data, list):
            batch = len(data)
            _, height, width = data[0].shape
        else:
            batch, _, height, width = data.shape
        
        fmh, fmw = ceil(height / self.rf), ceil(width / self.rf) # feature map shape
        output_shape=(batch, 84, self.ppc * fmh * fmw)
        
        buffer = []
        for binding in self.engine:
            mode = self.engine.get_tensor_mode(binding)
            if mode.name == "INPUT":
                self.context.set_input_shape(binding, data.shape)
                size = trt.volume(data.shape) * data.dtype.itemsize
                device_inpts = cuda.mem_alloc(size)
                cuda.memcpy_htod_async(device_inpts, data.tobytes())
                buffer.append(device_inpts)
            else:
                size = trt.volume(output_shape) * data.dtype.itemsize
                device_outs = cuda.mem_alloc(size)
                buffer.append(device_outs)
                
        self.buffer = buffer
        self.host_output = cuda.pagelocked_empty(output_shape, dtype=data.dtype)
    
    def clear(self):
        
        if self.buffer is not None:
            for mem_allocate in self.buffer:
                mem_allocate.free()
            
        self.buffer = None
        self.context = None
        self.engine = None
        
        gc.collect()