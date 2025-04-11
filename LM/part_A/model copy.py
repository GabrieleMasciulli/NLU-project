import torch.nn as nn
import torch.nn.functional as F


class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, compression_dim, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_RNN, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(
            output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers,
                          bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        # Compression Layer
        self.compression_layer = nn.Linear(hidden_size, compression_dim)
        # Linear layer to project the compressed hidden layer to our output space
        self.output = nn.Linear(compression_dim, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _ = self.rnn(emb)
        # paper mentioned using sigmoid instead
        compressed_out = F.relu(self.compression_layer(rnn_out))
        output = self.output(compressed_out).permute(0, 2, 1)
        return output
