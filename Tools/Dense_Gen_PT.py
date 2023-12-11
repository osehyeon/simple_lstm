import torch
import torch.nn as nn

class CustomDense(nn.Module):
    def __init__(self):
        super(CustomDense, self).__init__()
        # Define a fully connected layer with 128 input features and 64 output features
        self.fc = nn.Linear(in_features=128, out_features=64)

    def forward(self, x):
        # Pass the input through the fully connected layer
        x = self.fc(x)
        return x

# Create an instance of the model
model_dense = CustomDense()

# Create a dummy input with 1 sample, 128 features
dummy_input = torch.randn(1, 128)

# Define the ONNX file name
onnx_name = "test_dense_pt.onnx"

# Export the model to ONNX format
torch.onnx.export(model_dense, dummy_input, onnx_name)
