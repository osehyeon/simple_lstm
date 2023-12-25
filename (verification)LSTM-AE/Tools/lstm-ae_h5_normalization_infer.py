import tensorflow as tf
import numpy as np

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# 역 정규화 함수
def denormalize_data(normalized_data, original_data):
    return (normalized_data * (np.max(original_data) - np.min(original_data))) + np.min(original_data)

# 유사도 계산을 위한 함수 정의 (Mean Squared Error)
def calculate_mse(original, prediction):
    return np.mean(np.square(original - prediction))

# h5 모델 파일 경로
model_path = 'lstm-ae_test.h5'

# 모델 로드
model = tf.keras.models.load_model(model_path)

# normal.txt 파일에서 데이터 읽기
with open('../Data/anormal.txt', 'r') as file: 
    data = [float(line.strip()) for line in file]


# 원본 데이터 저장
original_data = np.array(data).reshape(1, 128, 1).astype(np.float32)

# 입력 데이터 정규화
normalized_data = normalize_data(original_data).reshape(1, 128, 1).astype(np.float32)

# 모델 추론 (정규화된 데이터 사용)
normalized_pred  = model.predict(normalized_data)

# 역 정규화 (예측값을 원래 스케일로 변환)
pred = denormalize_data(normalized_pred.flatten(), original_data)

# 유사도 계산 (원본 데이터와 역 정규화된 예측값 사용)
mse = calculate_mse(original_data, pred)

print("Predicted Values (Denormalized):")
print(pred)
print("\nMean Squared Error:")
print(mse)