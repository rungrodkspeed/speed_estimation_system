
import numpy as np
import onnxruntime as ort


def load_model(path):
    
    providers=['CPUExecutionProvider']

    return ort.InferenceSession(path, providers=providers)

# def inference(model, sample):
    
#     ort_input = [{model.get_inputs()[0].name: s} for s in sample] 
#     ort_out = [model.run(None, inpt)[0] for inpt in ort_input]
    
#     for outs in ort_out:
#         print(outs)

onnx_model = load_model("assets/yolov8n.onnx")

for inputs in onnx_model.get_inputs():
    print(inputs)
    
dummy = np.random.random((1, 3, 352, 352)).astype(np.float32)
input_dummy = {onnx_model.get_inputs()[0].name: dummy}

outs = onnx_model.run(None, input_dummy)[0]
print(outs.shape)