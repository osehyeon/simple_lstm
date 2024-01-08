import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape

# 데이터 정규화 함수
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# 파형 시각화 함수
def plot_waves(original_wave, normalized_wave, x_values):
    plt.figure(figsize=(14, 6))

    # 원본 파형 시각화
    plt.subplot(1, 2, 1)
    plt.title("Combined Sine and Cosine Waves")
    plt.plot(x_values, original_wave, label='Combined Wave')
    plt.legend()

    # 정규화된 파형 시각화
    plt.subplot(1, 2, 2)
    plt.title("Normalized Combined Wave")
    plt.plot(x_values, normalized_wave, label='Normalized Wave', color='orange')
    plt.legend()

    plt.show()

# 데이터 생성 
x = np.linspace(-2 * np.pi, 2 * np.pi, 128)  # -2π에서 2π까지 128개의 값
sin_wave = np.sin(x)
cos_wave = np.cos(x)
combined_wave = sin_wave + cos_wave

# 데이터 정규화 
normalized_wave = normalize_data(combined_wave)

# 파형 시각화
plot_waves(combined_wave, normalized_wave, x)

# 입력 데이터의 형태를 모델에 맞게 변환 
x_train = normalized_wave.reshape((1, 128, 1))
y_train = normalized_wave.reshape((1, 128, 1))

# Define the model
model = Sequential()
model.add(LSTM(128, input_shape=(128, 1)))
model.add(Reshape((128, 1))) 
model.add(Dense(32, activation='relu', input_shape=(128, 1)))
model.add(Dense(16, activation='relu')) 
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  

# 모델 컴파일 
model.compile(optimizer='adam', loss='mse')

# 모델 요약 
model.summary()

# 모델 훈련
model.fit(x_train, y_train, epochs=1024)

# 모델 저장 
model.save('lstm-ae_test.h5')