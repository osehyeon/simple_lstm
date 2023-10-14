import onnx
import onnxruntime
from onnx import helper, numpy_helper
from onnx import shape_inference
import numpy as np

def create_and_save_gru_model(sequence=10, hidden=10):
    # GRU's input and weight definition
    X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [sequence, 1, 1])
    W_shape = [1, 3*hidden, 1]  # For GRU, 3 gates (reset, update, and new_memory)
    R_shape = [1, 3*hidden, hidden]
    B_shape = [1, 6*hidden]  # 2 biases for each gate
    W = numpy_helper.from_array(np.random.randn(*W_shape).astype(np.float32), "W")
    R = numpy_helper.from_array(np.random.randn(*R_shape).astype(np.float32), "R")
    B = numpy_helper.from_array(np.random.randn(*B_shape).astype(np.float32), "B")

    # GRU node definition
    gru_node = onnx.helper.make_node(
        'GRU',
        inputs=['X', 'W', 'R', 'B'],
        outputs=['Y', 'Y_h'],
        hidden_size=hidden,
        direction='forward'
    )

    # Construct the graph
    graph = helper.make_graph(
        [gru_node],
        "GRU_Example",
        [X],
        [helper.make_tensor_value_info('Y_h', onnx.TensorProto.FLOAT, [1, 1, hidden])],
        [W, R, B]
    )

    # Create the model
    model = helper.make_model(graph, producer_name='onnx-gru')

    # Add shape inference
    inferred_model = shape_inference.infer_shapes(model)

    # Save the ONNX model
    file_name = f"gru-{sequence}-{hidden}.onnx"
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
create_and_save_gru_model(sequence=10, hidden=10)
