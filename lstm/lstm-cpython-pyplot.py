import numpy as np
import matplotlib.pyplot as plt
import ctypes

# 1. C shared library 불러오기
lstm_lib = ctypes.CDLL('./lstm-weight-update.so')
array_3d_float = np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, flags='CONTIGUOUS')
lstm_lib.entry.argtypes = [array_3d_float, array_3d_float]
lstm_lib.entry.restype = ctypes.c_void_p

def call_entry(tensor_X, Y_h):
    lstm_lib.entry(tensor_X, Y_h)

# 2. 예측 및 검증을 위한 데이터 생성하기
def generate_data(seq_length=10, num_samples=1000):
    x = np.linspace(0, num_samples, num_samples)
    y = np.sin(x)
    input_data, target_data = [], []
    for i in range(len(y) - 2*seq_length):
        input_data.append(y[i:i+seq_length])
        target_data.append(y[i+seq_length:i+2*seq_length])
    input_data = np.array(input_data, dtype=np.float32).reshape(-1, seq_length, 1)
    target_data = np.array(target_data, dtype=np.float32).reshape(-1, seq_length, 1)
    return input_data, target_data

inputs, targets = generate_data()

# 3. 그래프를 실시간으로 업데이트하기 위한 설정
plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
ax1.set_xlim(0, 20)
ax1.set_title("Input Data")
ax2.set_title("Prediction")
ax3.set_title("Ground Truth")
ax3.set_xlabel("Time Steps")
for ax in [ax1, ax2, ax3]:
    ax.set_ylim(-1.5, 1.5)

for i in range(len(inputs)-10):
    # C shared library를 사용하여 예측하기
    Y_h = np.empty((1, 1, 10), dtype=np.float32)
    call_entry(inputs[i:i+1], Y_h)
    predictions = Y_h.squeeze()
    
    # 실시간 그래프 업데이트
    x_input = np.linspace(i, i+10, 10)
    x_pred = np.linspace(i+10, i+20, 10)
    
    ax1.plot(x_input, inputs[i].squeeze(), color='b')
    ax2.plot(x_pred, predictions, color='r')
    ax3.plot(x_pred, targets[i].squeeze(), color='g')
    
    # x축 범위 업데이트
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(i, i + 30)
    
    plt.pause(0.5)

plt.ioff()
plt.tight_layout()
plt.show()
