import onnx
from onnx2pytorch import ConvertModel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

onnx_model = onnx.load("lstm-reshape-gemm.onnx")
pytorch_model = ConvertModel(onnx_model)
def generate_sin_data(seq_length, num_samples):
    t = np.linspace(0, 4 * math.pi, seq_length * num_samples + num_samples)  # + num_samples for the targets
    y = np.sin(t)
    input_data = []
    target_data = []
    for i in range(num_samples):
        start_idx = i * seq_length
        input_data.append(y[start_idx:start_idx+seq_length])
        target_data.append(y[start_idx+seq_length])  # Predicting the next data point
    return np.array(input_data), np.array(target_data)

seq_length = 10
num_samples = 1000
X, y = generate_sin_data(seq_length, num_samples)

X_train = torch.tensor(X).float().view(num_samples, seq_length, 1)
y_train = torch.tensor(y).float().view(num_samples, 1)

loss_function = nn.MSELoss()
optimizer = optim.Adam(pytorch_model.parameters(), lr=0.01)

for epoch in range(100):  # Or another desired number of epochs
    optimizer.zero_grad()
    outputs = pytorch_model(X_train)
    loss = loss_function(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        
torch.onnx.export(pytorch_model, X_train, "updated-lstm-reshape-gemm.onnx")
