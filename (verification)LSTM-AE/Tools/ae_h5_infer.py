import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 데이터 정규화 함수
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# 역 정규화 함수
def denormalize_data(normalized_data, original_data):
    return (normalized_data * (np.max(original_data) - np.min(original_data))) + np.min(original_data)

# 결과 시각화 함수
def plot_results(original, predicted, title):
    plt.figure(figsize=(10, 4))
    plt.plot(original, label='Actual Data')
    plt.plot(predicted, label='Predicted Data', color='red')
    plt.title(title)
    plt.legend()
    plt.show()

# 데이터 생성 함수
def generate_data(length=128, noise_factor=0.5):
    x = np.linspace(-2 * np.pi, 2 * np.pi, length)
    sin_wave = np.sin(x)
    cos_wave = np.cos(x)
    combined_wave = sin_wave + cos_wave
    
    # 정상 데이터
    normal_data = combined_wave

    # 비정상 데이터: 노이즈 추가
    noise = np.random.normal(0, noise_factor, combined_wave.shape)
    abnormal_data = combined_wave + noise
    
    return normal_data, abnormal_data

# 데이터 생성 및 정규화
normal_data, abnormal_data = generate_data()
normal_data_normalized = normalize_data(normal_data).reshape(1, 128, 1).astype(np.float32)
abnormal_data_normalized = normalize_data(abnormal_data).reshape(1, 128, 1).astype(np.float32)

# 모델 로드
model_path = 'lstm-ae_test.h5'  # 저장된 모델의 경로
model = tf.keras.models.load_model(model_path)

# 정상 데이터에 대한 모델 추론 및 역 정규화
normal_pred_normalized = model.predict(normal_data_normalized)
normal_pred = denormalize_data(normal_pred_normalized.reshape(128), normal_data)
plot_results(normal_data, normal_pred, 'Normal Data Prediction')

# 비정상 데이터에 대한 모델 추론 및 역 정규화
abnormal_pred_normalized = model.predict(abnormal_data_normalized)
abnormal_pred = denormalize_data(abnormal_pred_normalized.reshape(128), abnormal_data)
plot_results(abnormal_data, abnormal_pred, 'Abnormal Data Prediction')
