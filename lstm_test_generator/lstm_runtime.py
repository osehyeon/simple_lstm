import onnxruntime
import numpy as np 

ort_session = onnxruntime.InferenceSession("lstm_dense.onnx")

# Run inference using some random input data
# x_data = np.random.rand(1, 10, 1).astype(np.float32)
x_data = np.linspace(0, 2*np.pi, 10).reshape(1, 10, 1).astype(np.float32)
ort_inputs = {ort_session.get_inputs()[0].name: x_data}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs)