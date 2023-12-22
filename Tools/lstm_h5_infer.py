import tensorflow as tf
import numpy as np

# h5 모델 파일 경로

dir_path = './Model/'
file_name = 'lstm_dehs_epoch1_hidden128_v1.h5'
model_path = dir_path + file_name

# 모델 로드
model = tf.keras.models.load_model(model_path)

# normal.txt 파일에서 데이터 읽기
with open('../Data/anormal.txt', 'r') as file: 
    data = [float(line.strip()) for line in file]

# 테스트 데이터 생성 
#data = np.zeros((1, 128, 1), dtype=np.float32)
#for i in range(128):
#    data[0, i, 0] = float(i)

# 입력 데이터를 1x128x1 형태의 배열로 변환
tensor_X_data = np.array(data).reshape(1, 128, 1).astype(np.float32)

# 모델 추론
pred = model.predict(tensor_X_data)

print(pred[0])