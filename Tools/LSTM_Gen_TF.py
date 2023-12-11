import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM

# 모델 정의
model = Sequential([
    LSTM(2, input_shape=(128, 1))  # 입력 차원이 (128, 1)이며, 출력 차원이 2인 LSTM 레이어
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# 모델 요약 출력
model.summary()

# 모델을 h5 형식으로 저장
model.save('test_lstm_128_2.h5')