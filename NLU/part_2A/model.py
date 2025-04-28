import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ModelIAS(nn.Module):
    """
    This class instantiates the model for the intent and slot tagging task.
    """

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, lstm_dropout, fc_dropout, n_layers=1, pad_index=0):
        # hid_size: hidden size of the LSTM
        # out_slot: number of output classes for the slot tagging task
        # out_int: number of output classes for the intent classification task
        # emb_size: size of the word embeddings
        # vocab_len: size of the vocabulary
        # lstm_dropout: dropout probability on the hidden state of the LSTM
        # fc_dropout: dropout probability on the output of the LSTM
        # n_layers: number of layers of the LSTM
        # pad_index: index of the padding token
        super(ModelIAS, self).__init__()

        self.embedding = nn.Embedding(
            vocab_len, emb_size, padding_idx=pad_index)
        self.utt_encoder = nn.LSTM(
            emb_size, hid_size, n_layers, batch_first=True, bidirectional=True, dropout=fc_dropout)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.slot_out = nn.Linear(hid_size * 2, out_slot)
        self.intent_out = nn.Linear(hid_size * 2, out_int)

    def forward(self, utterance, seq_lengths):
        """
        Input:
            utterance: tensor of shape (batch_size, seq_len)
            seq_lengths: tensor of shape (batch_size)
        Output:
            slot_logits: tensor of shape (batch_size, seq_len, out_slot)
            intent_logits: tensor of shape (batch_size, out_int)
        """
        utt_emb = self.embedding(utterance)  # (batch_size, seq_len, emb_size)

        # pack_padded_sequence avoids computing the hidden state for padding tokens
        # --> reduces computational cost
        packed_input = pack_padded_sequence(
            utt_emb, seq_lengths.cpu().numpy(), batch_first=True)

        # process the batch
        packed_output, (hidden, cell) = self.utt_encoder(packed_input)

        # unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(
            packed_output, batch_first=True)  # (batch_size, seq_len, hid_size)

        # dropout on the output of the LSTM
        utt_encoded = self.fc_dropout(utt_encoded)

        # (batch_size, hid_size * 2)
        combined_hidden = torch.cat(
            (hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # dropout on the combined hidden state
        combined_hidden = self.fc_dropout(combined_hidden)

        # compute the logits for the slot tagging task
        # (batch_size, seq_len, classes)
        slot_logits = self.slot_out(utt_encoded)

        # compute the logits for the intent classification task
        # (batch_size, out_int)
        intent_logits = self.intent_out(combined_hidden)

        # (batch_size, classes, seq_len)
        slot_logits = slot_logits.permute(0, 2, 1)
        return slot_logits, intent_logits
