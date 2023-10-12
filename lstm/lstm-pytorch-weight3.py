import onnx
from onnx import numpy_helper

# Load the ONNX model
model_onnx = onnx.load("lstm-pytorch-10-1.onnx")

# Initialize empty dictionaries to store weights
lstm_weights = {}
gemm_weights = {}

# Extract LSTM weights directly from initializers
for initializer in model_onnx.graph.initializer:
    if initializer.name == 'onnx::LSTM_109':  # w
        lstm_weights['W'] = numpy_helper.to_array(initializer)
    elif initializer.name == 'onnx::LSTM_110':  # R
        lstm_weights['R'] = numpy_helper.to_array(initializer)
    elif initializer.name == 'onnx::LSTM_111':  # B
        lstm_weights['B'] = numpy_helper.to_array(initializer)
    elif initializer.name == '/lstm/Expand_output_0':  # 
        lstm_weights['Y_h'] = numpy_helper.to_array(initializer)
    elif initializer.name == '/lstm/Expand_1_output_0':  # B
        lstm_weights['Y_c'] = numpy_helper.to_array(initializer)

# Print the extracted weightstensor_W
print("LSTM Weights:")
print("W:", lstm_weights['W'])
print("R:", lstm_weights['R'])
print("B:", lstm_weights['B'])
#print("Y_h:", lstm_weights['Y_h'])
#print("Y_c:", lstm_weights['Y_c'])



def array_to_c_braces(arr):
    """
    Convert a numpy array to a nested brace format for C multidimensional arrays.
    """
    if arr.ndim == 1:
        return "{" + ", ".join(map(str, arr)) + "}"
    else:
        return "{" + ", ".join(map(array_to_c_braces, arr)) + "}"

def numpy_to_c_array_string(arr, var_name):
    """
    Convert a numpy array to a C-style multi-dimensional array string.
    """
    # Convert array dimensions to C-style
    c_dims = "][".join(map(str, arr.shape))
    c_data = array_to_c_braces(arr)
    return f"float {var_name}[{c_dims}] = {c_data};"

c_w = numpy_to_c_array_string(lstm_weights['W'], "tensor_W")
c_r = numpy_to_c_array_string(lstm_weights['R'], "tensor_R")
c_b = numpy_to_c_array_string(lstm_weights['B'], "tensor_B")
#c_y_h = numpy_to_c_array_string(lstm_weights['Y_h'], "Y_h")
#c_y_c = numpy_to_c_array_string(lstm_weights['Y_c'], "Y_c")

with open("weights-in-c-10-1.txt", "w") as f:
    f.write(c_w + "\n\n")
    f.write(c_r + "\n\n")
    f.write(c_b + "\n\n")
    #f.write(c_y_h + "\n\n")
    #f.write(c_y_c + "\n\n")
