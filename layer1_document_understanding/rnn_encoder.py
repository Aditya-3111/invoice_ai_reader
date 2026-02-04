import torch
import torch.nn as nn

class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2):
        super(RNNEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, sequences):
        """
        sequences: Tensor [B, T, input_dim]
        """
        outputs, _ = self.lstm(sequences)
        return outputs
