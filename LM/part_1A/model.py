import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DEVICE


class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_len, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1, pretrained_weights=None):
        super(LM_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Embedding layer
        if pretrained_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weights, freeze=False, padding_idx=pad_index)
            # Ensure emb_size matches pretrained embedding dimension
            emb_size = pretrained_weights.shape[1]
        else:
            self.embedding = nn.Embedding(
                vocab_len, emb_size, padding_idx=pad_index)

        # LSTM layer
        self.lstm = nn.LSTM(emb_size, hidden_size,
                            num_layers=n_layers, batch_first=True)

        # Output layer (fully connected)
        self.output = nn.Linear(hidden_size, vocab_len)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _ = self.lstm(emb)
        output = self.output(rnn_out).permute(0, 2, 1)
        return output
