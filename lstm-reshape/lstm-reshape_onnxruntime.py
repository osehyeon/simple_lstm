import numpy as np
import onnxruntime as rt

# 모델 불러오기
sess = rt.InferenceSession("lstm-reshape.onnx")

# 입력 데이터 준비
PI = 3.14159265358979323846
delta = 2.0 * PI / 9.0
tensor_X = np.array([[i * delta] for i in range(10)], dtype=np.float32).reshape(10, 1, 1)

# 모델 추론
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

predicted_output = sess.run([output_name], {input_name: tensor_X})[0]

# 결과 출력
for i in range(128):
    print("Result:", predicted_output[0][i])