import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import onnxruntime as ort
import matplotlib.pyplot as plt

seq_length = 10

# Load ONNX model using onnxruntime
ort_session = ort.InferenceSession("simple_model.onnx")

# Generate some test data
test_samples = 200
test_signal = torch.sin(torch.linspace(0, 2 * np.pi * test_samples / seq_length, test_samples)).unsqueeze(-1)

# Prepare the test data
test_input_seq = [test_signal[i:i+seq_length] for i in range(test_samples - seq_length - 1)]

test_input_seq = torch.stack(test_input_seq).squeeze(-1)

# Perform inference and collect predictions
predictions = []

for i in range(len(test_input_seq)):
    # Convert torch tensor to numpy array
    ort_inputs = {ort_session.get_inputs()[0].name: test_input_seq[i].unsqueeze(0).unsqueeze(-1).numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    predictions.append(ort_outs[0][0][0])

# Convert predictions list to torch tensor for easier plotting
predictions = torch.tensor(predictions)

# Plot actual vs predicted
plt.figure(figsize=(15, 6))
plt.plot(test_signal[seq_length+1:], label="Actual", color="blue")
plt.plot(predictions, label="Predicted", color="red", linestyle="dashed")
plt.legend()
plt.title("Actual vs Predicted Sin Signal")
plt.xlabel("Time step")
plt.ylabel("Amplitude")
plt.show()
