import onnxruntime as ort
import numpy as np

# 모델 파일 경로
model_path = 'yirang_lstm_tf.onnx'

# ONNX Runtime 세션 초기화
sess = ort.InferenceSession(model_path)

# normal.txt 파일에서 데이터 읽기
# ../Data/anormal.txt 로 지정 시 비정상 데이터 추론 결과 확인 가능 
with open('../Data/normal.txt', 'r') as file: 
    data = [float(line.strip()) for line in file]

# 입력 데이터를 1x128x1 형태의 배열로 변환
tensor_X_data = np.array(data).reshape(1, 128, 1).astype(np.float32)


# 입력 이름과 출력 이름 가져오기
input_name = sess.get_inputs()[0].name
output_names = [output.name for output in sess.get_outputs()]
desired_output_index = output_names.index('lstm_9')

# 모델 추론
pred_onx = sess.run([output_names[desired_output_index]], {input_name: tensor_X_data})[0]


print(pred_onx[0])