import torch.nn as nn
from locked_dropout import LockedDropout


class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_len, pad_index=0, emb_dropout_rate=0.1,
                 out_dropout_rate=0.1, n_layers=1, pretrained_embeddings=None, freeze_embeddings=False):
        super(LM_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.emb_dropout_rate = emb_dropout_rate
        self.out_dropout_rate = out_dropout_rate

        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_len, emb_size, padding_idx=pad_index)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = not freeze_embeddings

        # LockedDropout layer
        self.locked_dropout = LockedDropout()

        # Create a list of LSTM layers
        self.rnns = nn.ModuleList()
        for i in range(n_layers):
            current_input_size = emb_size if i == 0 else self.hidden_size
            
            # Determine output size for this LSTM layer
            if i == n_layers - 1:  # If this is the final LSTM layer
                current_output_size = emb_size
            else:  # For all non-final LSTM layers
                current_output_size = self.hidden_size
            
            self.rnns.append(nn.LSTM(current_input_size, current_output_size,
                                     num_layers=1, batch_first=True))

        # Create a list of LockedDropout layers to apply *after* each LSTM layer's output
        self.inter_rnn_dropouts = nn.ModuleList(
            [LockedDropout() for _ in range(n_layers)])

        # Output layer (fully connected)
        # The input to this layer is the output of the final LSTM
        self.output = nn.Linear(emb_size, vocab_len)

        # --- Weight Tying ---
        # Weight tying is done directly, as the output layer's input dimension (emb_size)
        # matches the embedding layer's dimension (emb_size).
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
