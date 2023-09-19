import onnx
from onnx import numpy_helper

# Load the ONNX model
model_onnx = onnx.load("simple_model.onnx")

# Initialize empty dictionaries to store weights
lstm_weights = {}
gemm_weights = {}

# Extract LSTM weights directly from initializers
for initializer in model_onnx.graph.initializer:
    if initializer.name == '108':  # w
        lstm_weights['W'] = numpy_helper.to_array(initializer)
    elif initializer.name == '109':  # R
        lstm_weights['R'] = numpy_helper.to_array(initializer)
    elif initializer.name == '110':  # B
        lstm_weights['B'] = numpy_helper.to_array(initializer)
    elif initializer.name == 'fc.weight':
        gemm_weights['B'] = numpy_helper.to_array(initializer)
    elif initializer.name == 'fc.bias':
        gemm_weights['C'] = numpy_helper.to_array(initializer)

# Print the extracted weights
print("LSTM Weights:")
print("W:", lstm_weights['W'])
print("R:", lstm_weights['R'])
print("B:", lstm_weights['B'])

print("\nGemm Weights:")
print("B:", gemm_weights['B'])
print("C:", gemm_weights['C'])

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

c_w = numpy_to_c_array_string(lstm_weights['W'], "W")
c_r = numpy_to_c_array_string(lstm_weights['R'], "R")
c_b = numpy_to_c_array_string(lstm_weights['B'], "B")
c_gemmb = numpy_to_c_array_string(gemm_weights['B'], "gemmB")
c_gemmc = numpy_to_c_array_string(gemm_weights['C'], "gemmC")

with open("weights_in_c.txt", "w") as f:
    f.write(c_w + "\n\n")
    f.write(c_r + "\n\n")
    f.write(c_b + "\n\n")
    f.write(c_gemmb + "\n\n")
    f.write(c_gemmc + "\n\n")
