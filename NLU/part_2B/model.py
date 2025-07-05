import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from utils import SLOT_PAD_LABEL_ID

# Define CTRAN specific parameters
CNN_KERNEL_SIZES = [1, 2, 3, 5]
CNN_FILTERS = 256
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 2
TRANSFORMER_FF_DIM = 1024


class CTRAN_INSPIRED(BertPreTrainedModel):
    """
    CTRAN_INSPIRED: CNN-Transformer-based network inspired by the C-TRAN model
    for joint intent classification and slot filling. Inherits from
    BertPreTrainedModel.

    Original C-TRAN paper: `https://arxiv.org/abs/2303.10606`
    Original C-TRAN code: `https://github.com/rafiepour/CTran`
    """

    def __init__(self, config, num_intent_labels, num_slot_labels, dropout_prob=0.1):
        """
        Initializes the model.

        Args:
            config: The BERT model configuration object.
            num_intent_labels (int): Number of unique intent labels.
            num_slot_labels (int): Number of unique slot labels.
            dropout_prob (float): Dropout probability.
        """
        super().__init__(config)
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        bert_hidden_size = config.hidden_size

        # Load the pre-trained BERT model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout_prob)

        # --- CTRAN Specific Layers ---
        # 1. CNN Layers
        # Input: (batch_size, seq_len, bert_hidden_size)
        # Output: (batch_size, seq_len, cnn_filters)
        self.conv_layers = nn.ModuleList(modules=[
            nn.Conv1d(in_channels=bert_hidden_size, out_channels=CNN_FILTERS // len(CNN_KERNEL_SIZES),
                      kernel_size=k, padding='same') for k in CNN_KERNEL_SIZES])

        self.cnn_activation = nn.ReLU()
        self.cnn_dropout = nn.Dropout(dropout_prob)

        # 2. Transformer Encoder Layer
        # Input: (batch_size, seq_len, cnn_filters)
        # Output: (batch_size, seq_len, cnn_filters) - Transformer preserves dimensions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=CNN_FILTERS,
            nhead=TRANSFORMER_HEADS,
            dim_feedforward=TRANSFORMER_FF_DIM,
            dropout=dropout_prob,
            activation='relu',
            batch_first=True  # Expect input as (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=TRANSFORMER_LAYERS
        )
        # --- End CTRAN Specific Layers ---

        # Classifier for intent classification (uses the BERT [CLS] token output)
        self.intent_classifier = nn.Linear(
            bert_hidden_size, num_intent_labels)

        # Classifier for slot filling (uses the output of the Transformer Encoder)
        # Input dim is now CNN_FILTERS
        self.slot_classifier = nn.Linear(CNN_FILTERS, num_slot_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        intent_labels=None,
        slot_labels=None,
    ):
        """
        Forward pass of the CTRAN model.
        """

        # Get BERT outputs
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Sequence output from BERT
        # Shape: (batch_size, sequence_length, bert_hidden_size)
        sequence_output = outputs[0]

        # Pooled output from BERT (for intent classification)
        # Shape: (batch_size, bert_hidden_size)
        pooled_output = outputs[1]

        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)  # Dropout for intent path

        # --- CTRAN Processing for Slots ---
        # 1. CNN
        # Permute for Conv1d: (batch, seq_len, hidden) -> (batch, hidden, seq_len)
        cnn_input = sequence_output.permute(0, 2, 1)
        
        # Apply each convolution separately
        convolve1 = self.cnn_activation(self.conv_layers[0](cnn_input))
        convolve2 = self.cnn_activation(self.conv_layers[1](cnn_input))
        convolve3 = self.cnn_activation(self.conv_layers[2](cnn_input))
        convolve4 = self.cnn_activation(self.conv_layers[3](cnn_input))
        
        # Transpose back: (batch, filters, seq_len) -> (batch, seq_len, filters)
        convolve1 = torch.transpose(convolve1, dim0=1, dim1=2)
        convolve2 = torch.transpose(convolve2, dim0=1, dim1=2)
        convolve3 = torch.transpose(convolve3, dim0=1, dim1=2)
        convolve4 = torch.transpose(convolve4, dim0=1, dim1=2)
        
        # Concatenate along feature dimension
        transformer_input = torch.cat((convolve1, convolve2, convolve3, convolve4), dim=2)
        transformer_input = self.cnn_dropout(transformer_input)

        # 2. Transformer Encoder
        # Transformer expects src_key_padding_mask where True indicates padding
        # attention_mask is 1 for real tokens, 0 for padding. Need to invert.
        if attention_mask is not None:
            # Ensure mask has same seq length as transformer input
            transformer_mask = attention_mask[:,
                                              :transformer_input.shape[1]] == 0
        else:
            transformer_mask = None

        transformer_output = self.transformer_encoder(
            transformer_input,
            src_key_padding_mask=transformer_mask
        )
        # Apply dropout after transformer as well
        transformer_output = self.dropout(transformer_output)
        # --- End CTRAN Processing ---

        # Get logits
        # Intent uses original pooled output
        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(
            transformer_output)  # Slot uses CTRAN output

        # Calculate loss if labels are provided
        total_loss = None
        intent_loss = None
        slot_loss = None

        if intent_labels is not None and slot_labels is not None:
            intent_loss_fct = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fct(
                intent_logits.view(-1, self.num_intent_labels), intent_labels.view(-1))

            # Slot Loss (using CrossEntropyLoss, ignoring PAD tokens)
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=SLOT_PAD_LABEL_ID)
            # Only compute loss for active parts of the sequence
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1,
                                                 self.num_slot_labels)[active_loss]
                active_labels = slot_labels.view(-1)[active_loss]
                # Check if there are any active labels to compute loss on
                if active_labels.nelement() > 0:
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    # Handle case where the batch might only contain padding after masking
                    # Or handle as appropriate
                    slot_loss = torch.tensor(0.0, device=slot_logits.device)
            else:
                # No attention mask, compute loss on all tokens
                slot_loss = slot_loss_fct(
                    slot_logits.view(-1, self.num_slot_labels), slot_labels.view(-1))

            total_loss = intent_loss + slot_loss

        # Return a dictionary containing all relevant outputs
        return {
            'loss': total_loss,
            'intent_logits': intent_logits,
            'slot_logits': slot_logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
            'intent_loss': intent_loss,
            'slot_loss': slot_loss,
        }
