
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 1. LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return hn

# 2. 데이터셋 준비하기
def generate_data(seq_length=10, num_samples=1000):
    x = np.linspace(0, num_samples, num_samples)
    y = np.sin(x)
    input_data, target_data = [], []
    for i in range(len(y) - 2*seq_length):
        input_data.append(y[i:i+seq_length])
        target_data.append(y[i+seq_length:i+2*seq_length])
    input_data = np.array(input_data, dtype=np.float32).reshape(-1, seq_length, 1)
    target_data = np.array(target_data, dtype=np.float32).reshape(-1, 10)
    return torch.from_numpy(input_data), torch.from_numpy(target_data)

inputs, targets = generate_data()
train_data = torch.utils.data.TensorDataset(inputs, targets)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

# 3. 모델 학습하기
model = LSTMModel(input_dim=1, hidden_dim=10)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):  # For simplicity, we just train for 10 epochs.
    for batch_inputs, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_inputs).squeeze()
        loss = loss_function(outputs, batch_targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 4. 학습된 모델을 ONNX로 저장하기
dummy_input = torch.randn(1, 10, 1)  # 입력의 형태를 정의합니다.
torch.onnx.export(model, dummy_input, "lstm_model.onnx")
