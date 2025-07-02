import torch.nn as nn

# LockedDropout Implementation (for Variational Dropout)


class LockedDropout(nn.Module):
    """
    - The difference between this and the standard PyTorch dropout is that regular
    nn.Dropout applies a new, random mask at each timestep which harms recurrent
    memory.
    - Variational Dropout uses the same dropout mask across every timestep of a
    sequence instead, which effectively preserves temporal consistency.
    - DropConnect is a special case of variational dropout where the dropout mask
    is applied to the input weights rather than the activations.

    NOTE:
    - When applying variational dropout, we want the expected value of the
    activations to be the same as the original activations because the network
    would get "confused" at inference time as the average activations would be
    much higher with respect to the "variatonally-dropped" acrtivations during
    training. Therefore, we divide the mask by the probability of the mask being
    applied, which effectively scales the remaining activations such that the
    expected value remains the same. Scaling is also used such that to prevent
    the gradients from vanishing during backpropagation.

    Example:
        - Given some activations x = [10.0, 20.0, 30.0, 40.0] and a dropout rate
          of 0.5, the expected value of the activations after dropout is (10.0 +
          20.0 + 30.0 + 40.0) / 4 = 25.0.

        - If we apply a dropout mask of [0, 1, 0, 1], the resulting activations
          would look like x_dropped = [0.0, 20.0, 0.0, 40.0], and the expected
          value of the activations after dropout would be (0.0 + 20.0 + 0.0 +
          40.0) / 4 = 10.0.

        - To preserve the expected value, we divide the mask by (1 -
          dropout_rate), leading to a mask of [0.5, 1.0, 0.5, 1.0].
          Note that scaling the mask by (1 - dropout_rate) is equivalent to scaling
          the remaining activations by (1 / (1 - dropout_rate)).


    See https://github.com/salesforce/awd-lstm-lm/blob/master/locked_dropout.py
    for original implementation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, dropout_rate=0.5):
        # x has shape (batch_size, seq_len, emb_size)
        if not self.training or not dropout_rate:
            return x

        # Create a mask for the embedding dimension (features)
        # Same mask applied to all elements in batch and sequence
        # Shape: (1, 1, emb_size)
        mask = x.new_empty(
            1, 1, x.size(-1), requires_grad=False).bernoulli_(1 - dropout_rate)
        mask = mask / (1 - dropout_rate)

        # Expand mask to match input shape and apply
        # Shape: (batch_size, seq_len, emb_size)
        mask = mask.expand_as(x)
        return x * mask
