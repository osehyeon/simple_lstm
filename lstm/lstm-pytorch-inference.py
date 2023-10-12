import numpy as np
import onnxruntime as ort

# Constants
PI = 3.14159265358979323846

# 1. onnxruntime 세션 생성하기
ort_session = ort.InferenceSession("lstm-pytorch-10.onnx")

# 2. 예측 및 검증을 위한 데이터 생성하기
def generate_data():
    tensor_X = np.zeros((1, 10, 1), dtype=np.float32)
    delta = 2.0 * PI / 9.0
    for i in range(10):
        tensor_X[0, i, 0] = i * delta
    return tensor_X



inputs = generate_data()

#print(inputs)
# ONNX 모델을 사용하여 예측하기
ort_inputs = {ort_session.get_inputs()[0].name: inputs}
ort_outs = ort_session.run(None, ort_inputs)
predictions = np.array(ort_outs).squeeze()

#print(f"Input Data: {inputs.squeeze()}")
print(f"Predictions: {predictions}")
