import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=128, num_layers=1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

model = SimpleModel()
criterion = nn.MSELoss()  # Mean squared error for regression tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generate a sin signal
seq_length = 10
num_samples = 1000
sin_signal = torch.sin(torch.linspace(0, 2 * np.pi * num_samples / seq_length, num_samples)).unsqueeze(-1)

# Prepare the data
input_seq = [sin_signal[i:i+seq_length] for i in range(num_samples - seq_length - 1)]
target_seq = [sin_signal[i+seq_length:i+seq_length+1] for i in range(num_samples - seq_length - 1)]

input_seq = torch.stack(input_seq).squeeze(-1)
target_seq = torch.stack(target_seq).squeeze(-1)

# Training loop
epochs = 100
for epoch in range(epochs):
    for i in range(len(input_seq)):
        optimizer.zero_grad()
        
        output = model(input_seq[i].unsqueeze(0).unsqueeze(-1))
        loss = criterion(output, target_seq[i][-1].unsqueeze(0).unsqueeze(-1))

        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


dummy_input = torch.randn(1, 10, 1)  # [batch, sequence, feature]

# Save the model in ONNX format
torch.onnx.export(model,               # model being run
                  dummy_input,         # model input (or a tuple for multiple inputs)
                  "simple_model.onnx", # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=11,    # the ONNX version to export the model to
                  do_constant_folding=True)  # whether to execute constant folding for optimization



