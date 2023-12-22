import onnxruntime as rt
import numpy as np

dir_path = './Model'
file_name = 'lstm_test.onnx'

# 입력 데이터 준비
Tensor_X = np.zeros((1, 128, 1), dtype=np.float32)
for i in range(128):
    Tensor_X[0, i, 0] = float(i)

# ONNX 런타임 세션 로드
sess = rt.InferenceSession(dir_path + file_name)

# 입력 이름과 출력 이름 가져오기
input_name = sess.get_inputs()[0].name
output_names = [output.name for output in sess.get_outputs()]

desired_output_index = output_names.index('hn4')

# 모델 추론
pred_onx = sess.run([output_names[desired_output_index]], {input_name: Tensor_X})[0]

        
print(pred_onx)