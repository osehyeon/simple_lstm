import tensorflow as tf
import numpy as np

# h5 모델 파일 경로
model_path = 'DH_epoch1_512_model0.h5'

# 모델 로드
model = tf.keras.models.load_model(model_path)

# normal.txt 파일에서 데이터 읽기
with open('../Data/normalization/anormal.txt', 'r') as file: 
    data = [float(line.strip()) for line in file]

# 입력 데이터를 1x128x1 형태의 배열로 변환
tensor_X_data = np.array(data).reshape(1, 128, 1).astype(np.float32)

# 모델 추론
pred = model.predict(tensor_X_data)


# lstm_7 레이어의 출력을 포함하는 새로운 모델 생성
lstm_output_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('lstm_7').output)

# 새 모델을 사용하여 lstm_7 레이어의 출력 얻기
lstm_output = lstm_output_model.predict(tensor_X_data)

print("Predicted Values from Full Model:")
print(pred[0])

print("Output from 'lstm_7' layer:")
print(lstm_output)