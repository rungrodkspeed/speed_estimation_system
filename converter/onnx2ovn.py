import openvino as ov

core = ov.Core()
onnx_model_path = "../assets/yolov8n.onnx"

model_onnx = core.read_model(model=onnx_model_path)
compiled_model_onnx = core.compile_model(model=model_onnx, device_name="CPU")