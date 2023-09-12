from onnx2pytorch import ConvertModel
import onnx
import torch

onnx_model = onnx.load("lstm_dense.onnx")
pytorch_model = ConvertModel(onnx_model)

dummy_input = torch.randn(1, 10, 1)
torch.onnx.export(pytorch_model, dummy_input, "trained_lstm_dense.onnx", verbose=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX, opset_version=11)
