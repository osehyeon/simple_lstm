import numpy as np
import onnxruntime as ort

# Constants
PI = 3.14159265358979323846

# 1. onnxruntime 세션 생성하기
ort_session = ort.InferenceSession("lstm-pytorch-10-1.onnx")

# 2. 예측 및 검증을 위한 데이터 생성하기
def generate_data(seq_start=0):
    x_vals = np.linspace(seq_start, seq_start + 9, 10)  # 10개의 연속된 정수 값 생성
    y_vals = np.sin(x_vals)
    tensor_X = y_vals.reshape(1, -1, 1).astype(np.float32)
    return tensor_X
inputs = generate_data()

print(inputs)

# ONNX 모델을 사용하여 예측하기
ort_inputs = {ort_session.get_inputs()[0].name: inputs}
ort_outs = ort_session.run(None, ort_inputs)
predictions = np.array(ort_outs).squeeze()

#print(f"Input Data: {inputs.squeeze()}")
print(f"Predictions: {predictions}")
f