import onnx
from onnx import numpy_helper

# Load the ONNX model
model_onnx = onnx.load("lstm-pytorch-10.onnx")

# Initialize empty dictionaries to store weights
lstm_weights = {}

# Extract LSTM weights directly from initializers
for initializer in model_onnx.graph.initializer:
    if initializer.name == 'onnx::LSTM_109':  # w
        lstm_weights['W'] = numpy_helper.to_array(initializer)
    elif initializer.name == 'onnx::LSTM_110':  # R
        lstm_weights['R'] = numpy_helper.to_array(initializer)
    elif initializer.name == 'onnx::LSTM_111':  # B
        lstm_weights['B'] = numpy_helper.to_array(initializer)

# Save weights to individual text files
for key, arr in lstm_weights.items():
    # Flatten the numpy array to 1D for easier extraction of elements
    flattened_array = arr.flatten()
    
    # Create file path for each weight
    file_name = f"{key}_weights.txt"
    
    # Write to file
    with open(file_name, 'w') as f:
        for val in flattened_array:
            f.write(f"{val}\n")
