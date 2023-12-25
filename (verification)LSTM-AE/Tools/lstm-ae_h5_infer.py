import tensorflow as tf
import numpy as np

# 유사도 계산을 위한 함수 정의 (Mean Squared Error)
def calculate_mse(original, prediction):
    return np.mean(np.square(original - prediction))


# h5 모델 파일 경로
model_path = 'lstm-ae_test.h5'

# 모델 로드
model = tf.keras.models.load_model(model_path)

# normal.txt 파일에서 데이터 읽기
with open('../Data/normalization/anormal.txt', 'r') as file: 
    data = [float(line.strip()) for line in file]

# 입력 데이터를 1x128x1 형태의 배열로 변환
tensor_X_data = np.array(data).reshape(1, 128, 1).astype(np.float32)

# 모델 추론
pred = model.predict(tensor_X_data)

# 유사도 계산
mse = calculate_mse(tensor_X_data.flatten(), pred.flatten())

print("Predicted Values:")
print(pred[0])
print("\nMean Squared Error:")
print(mse)
