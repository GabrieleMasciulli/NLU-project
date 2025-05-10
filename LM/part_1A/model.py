import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DEVICE


class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_len, pad_index=0, n_layers=1):

        super(LM_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Embedding layer
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


class LSTM_Dropout(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_len, pad_index=0, emb_dropout_rate=0.1,
                 lstm_dropout_rate=0.0, out_dropout_rate=0.1, n_layers=1):
        super(LSTM_Dropout, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_len, emb_size, padding_idx=pad_index)
        self.emb_dropout = nn.Dropout(emb_dropout_rate)

        # LSTM layer
        # Apply dropout between LSTM layers only if n_layers > 1
        lstm_dropout = lstm_dropout_rate if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(emb_size, hidden_size,
                            num_layers=n_layers, batch_first=True, dropout=lstm_dropout)

        self.out_dropout = nn.Dropout(out_dropout_rate)
        # Output layer (fully connected)
        self.output = nn.Linear(hidden_size, vocab_len)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.emb_dropout(emb)
        rnn_out, _ = self.lstm(emb)
        rnn_out = self.out_dropout(rnn_out)
        output = self.output(rnn_out).permute(0, 2, 1)
        return output
