import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as quant

sequence_length = 10
hidden_dim = 1

# 1. LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)  # 출력 형태를 맞추기 위한 Linear layer 추가
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        _, (hn, _) = self.lstm(x)
        output = self.linear(hn.squeeze(0))  # Linear layer 통과
        return self.dequant(output)

# 2. 데이터셋 준비하기
def generate_integer_data(seq_length=10, num_samples=1000, scale_factor=127):
    x = np.linspace(0, num_samples, num_samples)
    y = np.sin(x)
    y = (y * scale_factor).clip(-128, 127).astype(np.int8)
    input_data, target_data = [], []
    for i in range(len(y) - 2*seq_length):
        input_data.append(y[i:i+seq_length])
        target_data.append(y[i+seq_length:i+2*seq_length])
    input_data = np.array(input_data, dtype=np.int8).reshape(-1, seq_length, 1)
    target_data = np.array(target_data, dtype=np.int8).reshape(-1, seq_length)
    return torch.from_numpy(input_data).float(), torch.from_numpy(target_data).float()

inputs, targets = generate_integer_data()
train_data = torch.utils.data.TensorDataset(inputs, targets)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

# 3. 모델 학습하기
model = LSTMModel(input_dim=1, hidden_dim=hidden_dim, output_dim=sequence_length)  # output_dim 추가
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for batch_inputs, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = loss_function(outputs, batch_targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

model.qconfig = quant.default_qconfig
model = quant.prepare(model, inplace=True)

# 양자화 준비된 모델 수정
for batch_inputs, _ in train_loader:
    model(batch_inputs)

# 양자화 실행
model = quant.convert(model, inplace=True)

# 양자화된 모델을 ONNX로 저장하기
dummy_input = torch.randn(1, sequence_length, 1)
torch.onnx.export(model, dummy_input, "quantized_lstm_model.onnx")
