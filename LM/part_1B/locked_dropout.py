import torch.nn as nn

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
