import gc
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

import pycuda.autoinit


def load_engine(path):
    LOGGER = trt.Logger()
    with open(path, "rb") as f, trt.Runtime(LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        
    return engine

engine = load_engine("assets/yolov8n.plan")
context = engine.create_execution_context()

for _, binding in enumerate(engine):
    mode = engine.get_tensor_mode(binding)
    print(mode, mode.name, binding, context.get_tensor_shape(binding))
    
    
# device_inpts = cuda.mem_alloc(1024)
temp = cuda.pagelocked_empty((1,3,3), dtype=np.float16)
print(type(temp))

del context
del engine
gc.collect()