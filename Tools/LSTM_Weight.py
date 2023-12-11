import onnx
from onnx import numpy_helper
import numpy as np

# Load the ONNX model
model_onnx = onnx.load("yirang_lstm.onnx")

# Initialize empty dictionaries to store weights
lstm_weights = {}

# Extract LSTM weights directly from initializers
for initializer in model_onnx.graph.initializer:
    if initializer.name == 'onnx::LSTM_383':  
        lstm_weights['tensor_W_1_16'] = numpy_helper.to_array(initializer)
    elif initializer.name == 'onnx::LSTM_384':  
        lstm_weights['tensor_R_1_16'] = numpy_helper.to_array(initializer)
    elif initializer.name == 'onnx::LSTM_385':  
        lstm_weights['tensor_B_1_16'] = numpy_helper.to_array(initializer)
        
    elif initializer.name == 'onnx::LSTM_405':  
        lstm_weights['tensor_W_2_16'] = numpy_helper.to_array(initializer)
    elif initializer.name == 'onnx::LSTM_406':  
        lstm_weights['tensor_R_2_16'] = numpy_helper.to_array(initializer)
    elif initializer.name == 'onnx::LSTM_407':  
        lstm_weights['tensor_B_2_16'] = numpy_helper.to_array(initializer)
        
    elif initializer.name == 'onnx::LSTM_427':  
        lstm_weights['tensor_W_3_16'] = numpy_helper.to_array(initializer)
    elif initializer.name == 'onnx::LSTM_428':  
        lstm_weights['tensor_R_3_16'] = numpy_helper.to_array(initializer)
    elif initializer.name == 'onnx::LSTM_429':  
        lstm_weights['tensor_B_3_16'] = numpy_helper.to_array(initializer)
        
    elif initializer.name == 'onnx::LSTM_449':  
        lstm_weights['tensor_W_4_1'] = numpy_helper.to_array(initializer)
    elif initializer.name == 'onnx::LSTM_450':  
        lstm_weights['tensor_R_4_1'] = numpy_helper.to_array(initializer)
    elif initializer.name == 'onnx::LSTM_451':  
        lstm_weights['tensor_B_4_1'] = numpy_helper.to_array(initializer)
        
    
for weight_name, weight_array in lstm_weights.items():
    file_path = f"../Weight/{weight_name}.txt"
    
    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Check if the array is 3D
        if weight_array.ndim == 3:
            # Iterate over each element in the 3D array
            for data_slice in weight_array:
                for row in data_slice:
                    for element in row:
                        file.write(f"{element}\n")
        # Check if the array is 2D
        elif weight_array.ndim == 2:
            # Iterate over each element in the 2D array
            for row in weight_array:
                for element in row:
                    file.write(f"{element}\n")

for weight_name, weight_array in lstm_weights.items():
    file_path = f"../Weight/{weight_name}.bin"  # 변경: .txt -> .bin
    
    # Saving the array in binary format
    weight_array.tofile(file_path)  # numpy 배열을 바이너리 파일로 저장