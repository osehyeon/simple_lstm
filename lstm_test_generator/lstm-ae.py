import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# LSTM-AE Model
class LSTMAE(nn.Module):
    def __init__(self):
        super(LSTMAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.LSTM(1, 20, batch_first=True),
            nn.Linear(20, 30),
            nn.Linear(30, 1)
        )
        
        self.decoder = nn.Sequential(
            nn.LSTM(1, 10, batch_first=True),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        # Encoder
        x, _ = self.encoder[0](x)
        x = x[:, -1, :]
        x = self.encoder[1](x)
        x = self.encoder[2](x)
        x = x.unsqueeze(1)
        
        # Decoder
        x, _ = self.decoder[0](x)
        x = self.decoder[1](x)
        x = x.squeeze(1)
        
        return x

model = LSTMAE()
print(model)

# Generate sin wave data
timesteps = 10
x = np.linspace(0, 4*np.pi, timesteps)
y = np.sin(x)

# Training parameters
lr = 0.001
epochs = 1000
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    inputs = torch.tensor(y, dtype=torch.float32).view(1, timesteps, 1)
    outputs = model(inputs)
    loss = loss_fn(outputs, inputs)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Inference
model.eval()
with torch.no_grad():
    inputs = torch.tensor(y, dtype=torch.float32).view(1, timesteps, 1)
    decoded = model(inputs)

# Visualization
plt.figure(figsize=(12, 5))
plt.plot(y, label='Original sin wave', marker='o')
plt.plot(decoded[0].numpy(), label='Decoded sin wave', marker='x')
plt.legend()
plt.show()

dummy_input = torch.randn(1, timesteps, 1)  # Create a dummy input for ONNX export
torch.onnx.export(model, dummy_input, "lstm_ae.onnx")

print("Model saved to lstm_ae.onnx")
