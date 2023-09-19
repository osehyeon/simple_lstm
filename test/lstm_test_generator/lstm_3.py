import onnx
import onnxruntime
from onnx import helper, numpy_helper
import numpy as np

# 입력 텐서 정의
X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 10, 1])

# LSTM 가중치 및 바이어스 정의
W_shape = [1, 4*20, 1]  # 4: 숫자의 게이트 (i, o, f, c)
R_shape = [1, 4*20, 20]
B_shape = [1, 8*20]

W = numpy_helper.from_array(np.random.randn(*W_shape).astype(np.float32), "W")
R = numpy_helper.from_array(np.random.randn(*R_shape).astype(np.float32), "R")
B = numpy_helper.from_array(np.random.randn(*B_shape).astype(np.float32), "B")

# 출력 텐서 정의
Y = helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, 10, 20])

# LSTM 노드 정의
node = onnx.helper.make_node(
    'LSTM',
    inputs=['X', 'W', 'R', 'B'],
    outputs=['Y'],
    hidden_size=20,
    activations=['sigmoid', 'tanh', 'tanh'],
    direction='forward',
    input_forget=0
)

# 그래프 생성
graph = helper.make_graph(
    [node],
    "LSTMExample",
    [X],
    [Y],
    [W, R, B]
)

# 모델 생성
model = helper.make_model(graph, producer_name='onnx-lstm')

# ONNX 모델 저장
onnx.save(model, "simple_lstm.onnx")

# 모델 로드 및 실행을 통해 확인
ort_session = onnxruntime.InferenceSession("simple_lstm.onnx")

# 임의의 입력 데이터로 추론 실행
x_data = np.random.rand(1, 10, 1).astype(np.float32)
ort_inputs = {ort_session.get_inputs()[0].name: x_data}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs)

