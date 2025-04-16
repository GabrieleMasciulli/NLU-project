import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DEVICE


# LockedDropout Implementation (for Variational Dropout)
class LockedDropout(nn.Module):
    """ LockedDropout applies the same dropout mask to every timestep of a sequence.

    See https://github.com/salesforce/awd-lstm-lm/blob/master/locked_dropout.py
    for original implementation.
    """

    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        # x has shape (batch_size, seq_len, emb_size)
        if not self.training or not self.dropout:
            return x

        # Create a mask for the embedding dimension (features)
        # Same mask applied to all elements in batch and sequence
        # Shape: (1, 1, emb_size)
        mask = x.new_empty(
            1, 1, x.size(-1), requires_grad=False).bernoulli_(1 - self.dropout)
        mask = mask / (1 - self.dropout)  # Scale the mask

        # Expand mask to match input shape and apply
        # Shape: (batch_size, seq_len, emb_size)
        mask = mask.expand_as(x)
        return x * mask


class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_len, pad_index=0, emb_dropout_rate=0.1,
                 lstm_dropout_rate=0.0, out_dropout_rate=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_len, emb_size, padding_idx=pad_index)

        self.emb_dropout = LockedDropout(emb_dropout_rate)

        # LSTM layer
        # Apply dropout between LSTM layers only if n_layers > 1
        lstm_dropout = lstm_dropout_rate if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(emb_size, hidden_size,
                            num_layers=n_layers, batch_first=True, dropout=lstm_dropout)

        self.out_dropout = LockedDropout(out_dropout_rate)
        # Output layer (fully connected)
        self.output = nn.Linear(hidden_size, vocab_len)

        # --- Weight Tying ---
        # Tie the weights of embedding and output layers
        # Requires emb_size == hidden_size, which should be ensured by config
        if emb_size != hidden_size:
            raise ValueError(f"Weight tying requires embedding size ({emb_size}) "
                             f"to be equal to hidden size ({hidden_size})")
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.emb_dropout(emb)
        rnn_out, _ = self.lstm(emb)
        rnn_out = self.out_dropout(rnn_out)
        output = self.output(rnn_out).permute(0, 2, 1)
        return output
