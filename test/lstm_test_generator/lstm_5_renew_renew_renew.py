import onnx
import onnxruntime
from onnx import helper, numpy_helper
from onnx import shape_inference
import numpy as np

# LSTM's input size and hidden layer definitions
batch_size = 1
sequence_length = 10
feature_size = 1
hidden_size = 128
hidden_size_2 = 64
hidden_size_4 = 32

# LSTM's input and weight definition
X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [batch_size, sequence_length, feature_size])
W_shape = [1, 4*hidden_size, feature_size]
R_shape = [1, 4*hidden_size, hidden_size]
B_shape = [1, 8*hidden_size]
W = numpy_helper.from_array(np.random.randn(*W_shape).astype(np.float32), "W")
R = numpy_helper.from_array(np.random.randn(*R_shape).astype(np.float32), "R")
B = numpy_helper.from_array(np.random.randn(*B_shape).astype(np.float32), "B")

# LSTM node definition
lstm_node = onnx.helper.make_node(
    'LSTM',
    inputs=['X', 'W', 'R', 'B'],
    outputs=['Y', 'Y_h', 'Y_c'],
    hidden_size=hidden_size,
    direction='forward',
    input_forget=0
)

# Use Slice to get the last hidden state
starts = np.array([0, sequence_length-1, 0], dtype=np.int64)
ends = np.array([1, sequence_length, hidden_size], dtype=np.int64)
axes = np.array([0, 1, 2], dtype=np.int64)

slice_node = helper.make_node(
    'Slice',
    inputs=['Y_h', 'starts', 'ends', 'axes'],
    outputs=['last_hidden'],
)
starts_init = numpy_helper.from_array(starts, "starts")
ends_init = numpy_helper.from_array(ends, "ends")
axes_init = numpy_helper.from_array(axes, "axes")

# Reshape the sliced output to be [batch_size, hidden_size]
shape_tensor = numpy_helper.from_array(np.array([batch_size, hidden_size], dtype=np.int64), "shape_tensor")
reshape_node = helper.make_node(
    'Reshape',
    inputs=['last_hidden', 'shape_tensor'],
    outputs=['reshaped_last_hidden']
)

# Fully Connected (Dense) layer weights and bias definition
dense_weight = numpy_helper.from_array(np.random.randn(hidden_size_2, hidden_size).astype(np.float32), "dense_weight")
dense_bias = numpy_helper.from_array(np.random.randn(hidden_size_2).astype(np.float32), "dense_bias")

# Gemm (Fully Connected) node definition
gemm_node = helper.make_node(
    'Gemm',
    inputs=['reshaped_last_hidden', 'dense_weight', 'dense_bias'],
    outputs=['dense_out0'],
    transB=1
)

relu_node = helper.make_node(
    'Relu',
    inputs=['dense_out0'],
    outputs=['relu_output0']
)

# Fully Connected (Dense) layer weights and bias definition
dense_weight1 = numpy_helper.from_array(np.random.randn(hidden_size_4, hidden_size_2).astype(np.float32), "dense_weight1")
dense_bias1 = numpy_helper.from_array(np.random.randn(hidden_size_4).astype(np.float32), "dense_bias1")

# Gemm (Fully Connected) node definition
gemm_node1 = helper.make_node(
    'Gemm',
    inputs=['relu_output0', 'dense_weight1', 'dense_bias1'],
    outputs=['dense_out1'],
    transB=1
)

relu_node1 = helper.make_node(
    'Relu',
    inputs=['dense_out1'],
    outputs=['relu_output1']
)


# Fully Connected (Dense) layer weights and bias definition
dense_weight2 = numpy_helper.from_array(np.random.randn(hidden_size_2, hidden_size_4).astype(np.float32), "dense_weight2")
dense_bias2 = numpy_helper.from_array(np.random.randn(hidden_size_2).astype(np.float32), "dense_bias2")

# Gemm (Fully Connected) node definition
gemm_node2 = helper.make_node(
    'Gemm',
    inputs=['relu_output1', 'dense_weight2', 'dense_bias2'],
    outputs=['dense_out2'],
    transB=1
)

relu_node2 = helper.make_node(
    'Relu',
    inputs=['dense_out2'],
    outputs=['relu_output2']
)


# Fully Connected (Dense) layer weights and bias definition
dense_weight3 = numpy_helper.from_array(np.random.randn(hidden_size, hidden_size_2).astype(np.float32), "dense_weight3")
dense_bias3 = numpy_helper.from_array(np.random.randn(hidden_size).astype(np.float32), "dense_bias3")

# Gemm (Fully Connected) node definition
gemm_node3 = helper.make_node(
    'Gemm',
    inputs=['relu_output2', 'dense_weight3', 'dense_bias3'],
    outputs=['dense_out3'],
    transB=1
)

relu_node3 = helper.make_node(
    'Relu',
    inputs=['dense_out3'],
    outputs=['relu_output3']
)

# Fully Connected (Dense) layer weights and bias definition
dense_weight4 = numpy_helper.from_array(np.random.randn(1, hidden_size).astype(np.float32), "dense_weight4")
dense_bias4 = numpy_helper.from_array(np.random.randn(1).astype(np.float32), "dense_bias4")

# Gemm (Fully Connected) node definition
gemm_node4 = helper.make_node(
    'Gemm',
    inputs=['relu_output3', 'dense_weight4', 'dense_bias4'],
    outputs=['dense_out'],
    transB=1
)

# Construct the graph
graph = helper.make_graph(
    [lstm_node, slice_node, reshape_node, gemm_node, relu_node, gemm_node1,relu_node1, gemm_node2, relu_node2, gemm_node3,relu_node3, gemm_node4],
    "LSTM_Dense_Example",
    [X],
    [helper.make_tensor_value_info('dense_out', onnx.TensorProto.FLOAT, [batch_size, 1])],
    [W, R, B, dense_weight, dense_bias, starts_init, ends_init, axes_init, shape_tensor, dense_weight1, dense_bias1
    , dense_weight2, dense_bias2, dense_weight3, dense_bias3, dense_weight4, dense_bias4]
)

# Create the model
model = helper.make_model(graph, producer_name='onnx-lstm-dense')

# Save the ONNX model
inferred_model = shape_inference.infer_shapes(model)
onnx.save(inferred_model, "lstm_dense.onnx")


# Load the model and validate by running inference
ort_session = onnxruntime.InferenceSession("lstm_dense.onnx")

# Run inference using some random input data
x_data = np.random.rand(batch_size, sequence_length, feature_size).astype(np.float32)
ort_inputs = {ort_session.get_inputs()[0].name: x_data}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs)
