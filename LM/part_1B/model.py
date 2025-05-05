import torch.nn as nn
from locked_dropout import LockedDropout


class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_len, pad_index=0, emb_dropout_rate=0.1,
                 out_dropout_rate=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.emb_dropout_rate = emb_dropout_rate
        self.out_dropout_rate = out_dropout_rate

        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_len, emb_size, padding_idx=pad_index)

        # LockedDropout layer
        self.locked_dropout = LockedDropout()

        # Create a list of LSTM layers
        self.rnns = nn.ModuleList()
        for i in range(n_layers):
            input_sz = emb_size if i == 0 else hidden_size
            self.rnns.append(nn.LSTM(input_sz, hidden_size,
                             num_layers=1, batch_first=True))

        # Create a list of LockedDropout layers to apply *after* each LSTM layer's output
        self.inter_rnn_dropouts = nn.ModuleList(
            [LockedDropout() for _ in range(n_layers)])

        # Output layer (fully connected)
        self.output = nn.Linear(hidden_size, vocab_len)

        # --- Weight Tying ---
        if emb_size != hidden_size:
            # Weight tying is only possible if emb_size == hidden_size
            print(
                f"Warning: Weight tying not applied because embedding size ({emb_size}) != hidden size ({hidden_size})")
        else:
            # Tie the weights of embedding and output layers
            self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        # hidden should be a list of (h_n, c_n) tuples, one for each layer
        emb = self.embedding(input_sequence)
        # Apply dropout to embeddings
        current_input = self.locked_dropout(emb, self.emb_dropout_rate)

        for i, rnn in enumerate(self.rnns):

            rnn_output, _ = rnn(current_input)

            # Apply LockedDropout to the output of this LSTM layer
            rnn_output = self.inter_rnn_dropouts[i](rnn_output, self.out_dropout_rate)

            # The output of this layer becomes the input for the next
            current_input = rnn_output

        # The final rnn_output (after dropout) is used for the decoder
        output = self.output(rnn_output).permute(0, 2, 1)

        return output
