import torch
import torch.nn as nn

class CustomLSTM_HN(nn.Module):
    def __init__(self):
        super(CustomLSTM_HN, self).__init__()
        # Define LSTM layers with specified input and output features
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.lstm4 = nn.LSTM(input_size=1, hidden_size=1, batch_first=True)

    def forward(self, x):
        # Pass the input through each LSTM layer and use the hidden state for the next layer
        x, (hn1, _) = self.lstm1(x)
        hn1 = hn1.permute(1, 2, 0)
        x, (hn2, _) = self.lstm2(hn1)
        hn2 = hn2.permute(1, 2, 0)
        x, (hn3, _) = self.lstm3(hn2)
        hn3 = hn3.permute(1, 2, 0)
        x, (hn4, _) = self.lstm4(hn3)
        return hn1, hn2, hn3, hn4

# Create an instance of the model
model_hn = CustomLSTM_HN()

dummy_input = torch.randn(1, 128, 1)

onnx_name = "test_lstm.onnx"

output_names = ['hn1', 'hn2', 'hn3', 'hn4']

torch.onnx.export(model_hn, dummy_input, onnx_name, output_names=output_names)
