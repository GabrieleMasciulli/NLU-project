import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from utils import SLOT_PAD_LABEL_ID

# Define CTRAN specific parameters
CNN_KERNEL_SIZE = 3
CNN_FILTERS = 256
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 2
TRANSFORMER_FF_DIM = 1024


class CTRAN(BertPreTrainedModel):
    """
    CTRAN: CNN-Transformer-based network for joint intent classification and slot filling.
    Inherits from BertPreTrainedModel.
    """

    def __init__(self, config, num_intent_labels, num_slot_labels, dropout_prob=0.1):
        """
        Initializes the CTRAN model.

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
        # 1. CNN Layer
        # Input: (batch_size, seq_len, bert_hidden_size) -> permute to (batch_size, bert_hidden_size, seq_len)
        # Output: (batch_size, cnn_filters, seq_len) -> permute back to (batch_size, seq_len, cnn_filters)
        self.conv1d = nn.Conv1d(
            in_channels=bert_hidden_size,
            out_channels=CNN_FILTERS,
            kernel_size=CNN_KERNEL_SIZE,
            padding=(CNN_KERNEL_SIZE - 1) // 2  # Maintain sequence length
        )
        self.cnn_activation = nn.ReLU()

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
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        intent_labels=None,
        slot_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        loss_alpha=0.5,
    ):
        """
        Forward pass of the CTRAN model.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get BERT outputs
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
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
        cnn_output = self.conv1d(cnn_input)
        cnn_output = self.cnn_activation(cnn_output)
        # Permute back: (batch, filters, seq_len) -> (batch, seq_len, filters)
        transformer_input = cnn_output.permute(0, 2, 1)

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

            # Combine losses
            # Ensure slot_loss is a valid tensor before combining
            if slot_loss is not None:
                total_loss = loss_alpha * intent_loss + \
                    (1 - loss_alpha) * slot_loss
            else:  # Fallback if slot_loss couldn't be computed
                total_loss = loss_alpha * intent_loss

        if not return_dict:
            output = (intent_logits, slot_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

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
