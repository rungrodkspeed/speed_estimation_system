import numpy as np
import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints

from typing import Optional, Union, List, Dict

class OpenVinoEngine:
    
    model: Optional[ov.runtime.ie_api.CompiledModel]
    request: Optional[ov.runtime.ie_api.InferRequest]
    
    def __init__(self, path: str, device: str = "AUTO"):
        self.core = ov.Core()
        self.core.set_property(device, {hints.execution_mode: hints.ExecutionMode.PERFORMANCE})
        
        self.device = device
        self.network = self.core.read_model(model=path)
        self.model = None
        self.request = None
        
    def __call__(self, images: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        
        if isinstance(images, list):
            images = np.array(images)
            
        input_tensor = ov.Tensor(array=images, shared_memory=True)
        self.request.set_input_tensor(input_tensor)
        
        self.request.start_async()
        self.request.wait()
        
        output = self.request.get_output_tensor()
        return output.data
    
    def compile_model(self, cfg: Optional[Dict] = None, with_optimal: bool = False):

        if with_optimal:
            if cfg is None:
                cfg = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
                       props.inference_num_threads: "4"}
                
        self.model = self.core.compile_model(self.network, self.device, cfg)
        self.request = self.model.create_infer_request()