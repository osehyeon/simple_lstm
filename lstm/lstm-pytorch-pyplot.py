import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

# 1. onnxruntime 세션 생성하기
ort_session = ort.InferenceSession("lstm-pytorch-10.onnx")

# 2. 예측 및 검증을 위한 데이터 생성하기
def generate_data(seq_length=10, num_samples=1000):
    x = np.linspace(0, num_samples, num_samples)
    y = np.sin(x)
    input_data, target_data = [], []
    for i in range(len(y) - 2*seq_length):
        input_data.append(y[i:i+seq_length])
        target_data.append(y[i+seq_length:i+2*seq_length])
    input_data = np.array(input_data, dtype=np.float32).reshape(-1, seq_length, 1)
    target_data = np.array(target_data, dtype=np.float32).reshape(-1, 10)
    return input_data, target_data

inputs, targets = generate_data()

# 3. 그래프를 실시간으로 업데이트하기 위한 설정
plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
ax1.set_xlim(0, 20)  # 처음에는 첫 20개 데이터만 보여줍니다.
ax1.set_title("Input Data")
ax2.set_title("Prediction")
ax3.set_title("Ground Truth")
ax3.set_xlabel("Time Steps")
for ax in [ax1, ax2, ax3]:
    ax.set_ylim(-1.5, 1.5)

for i in range(len(inputs)-10):
    # ONNX 모델을 사용하여 예측하기
    ort_inputs = {ort_session.get_inputs()[0].name: inputs[i:i+1]}
    ort_outs = ort_session.run(None, ort_inputs)
    predictions = np.array(ort_outs).squeeze()
    
    # 실시간 그래프 업데이트
    x_input = np.linspace(i, i+10, 10)
    x_pred = np.linspace(i+10, i+20, 10)
    
    ax1.plot(x_input, inputs[i].squeeze(), color='b')
    ax2.plot(x_pred, predictions, color='r')
    ax3.plot(x_pred, targets[i], color='g')
    
    # x축 범위 업데이트
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(i, i + 30)
    
    plt.pause(0.5)  # 그래프 업데이트 간격을 0.5초로 설정합니다.

plt.ioff()
plt.tight_layout()
plt.show()
