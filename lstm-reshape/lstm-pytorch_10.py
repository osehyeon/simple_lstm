import onnx
import onnxruntime
from onnx import helper, numpy_helper
from onnx import shape_inference
import numpy as np

# LSTM's input and weight definition
X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 10, 1])
W_shape = [1, 4*10, 1]
R_shape = [1, 4*10, 10]
B_shape = [1, 8*10]
W = numpy_helper.from_array(np.random.randn(*W_shape).astype(np.float32), "W")
R = numpy_helper.from_array(np.random.randn(*R_shape).astype(np.float32), "R")
B = numpy_helper.from_array(np.random.randn(*B_shape).astype(np.float32), "B")

# LSTM node definition
lstm_node = onnx.helper.make_node(
    'LSTM',
    inputs=['X', 'W', 'R', 'B'],
    outputs=['Y', 'Y_h', 'Y_c'],
    hidden_size=10,
    #activations=['sigmoid', 'tanh', 'tanh'],
    direction='forward',
    input_forget=0
)



# Construct the graph
graph = helper.make_graph(
    [lstm_node],
    "LSTM_Reshape_Example",
    [X],
    [helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, 1, 10, 10])],
    [W, R, B]
)


# Create the model
model = helper.make_model(graph, producer_name='onnx-lstm-reshape-pytorch')

# Add shape inference
inferred_model = shape_inference.infer_shapes(model)

# Save the ONNX model
onnx.save(inferred_model, "lstm-pytorch.onnx")

# Load the model and validate by running inference
ort_session = onnxruntime.InferenceSession("lstm-pytorch.onnx")

# Run inference using some random input data
x_data = np.random.rand(1, 10, 1).astype(np.float32)
ort_inputs = {ort_session.get_inputs()[0].name: x_data}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs)
