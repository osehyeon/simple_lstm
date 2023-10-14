import onnx
import onnxruntime
from onnx import helper, numpy_helper
from onnx import shape_inference
import numpy as np

def create_and_save_rnn_model(sequence=10, hidden=10):
    # RNN's input and weight definition
    X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [sequence, 1, 1])
    W_shape = [1, hidden, 1]  # For basic RNN, just one weight matrix
    R_shape = [1, hidden, hidden]  # Recurrent weight matrix
    B_shape = [1, 2*hidden]  # 2 biases for each neuron: one for input, one for recurrent connection
    W = numpy_helper.from_array(np.random.randn(*W_shape).astype(np.float32), "W")
    R = numpy_helper.from_array(np.random.randn(*R_shape).astype(np.float32), "R")
    B = numpy_helper.from_array(np.random.randn(*B_shape).astype(np.float32), "B")

    # RNN node definition
    rnn_node = onnx.helper.make_node(
        'RNN',
        inputs=['X', 'W', 'R', 'B'],
        outputs=['Y', 'Y_h'],
        hidden_size=hidden,
        direction='forward',
        activations=['Tanh'] # default activation function for RNN is tanh
    )

    # Construct the graph
    graph = helper.make_graph(
        [rnn_node],
        "RNN_Example",
        [X],
        [helper.make_tensor_value_info('Y_h', onnx.TensorProto.FLOAT, [1, 1, hidden])],
        [W, R, B]
    )

    # Create the model
    model = helper.make_model(graph, producer_name='onnx-rnn')

    # Add shape inference
    inferred_model = shape_inference.infer_shapes(model)

    # Save the ONNX model
    file_name = f"rnn-{sequence}-{hidden}.onnx"
    onnx.save(inferred_model, file_name)
    print(f"Saved model to {file_name}")

    # Load the model and validate by running inference
    ort_session = onnxruntime.InferenceSession(file_name)

    # Run inference using some random input data
    x_data = np.random.rand(sequence, 1, 1).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: x_data}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)

# Example usage
create_and_save_rnn_model(sequence=10, hidden=10)
