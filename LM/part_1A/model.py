import torch.nn as nn


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
