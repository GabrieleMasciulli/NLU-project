import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from torchcrf import CRF


class JointBERT(BertPreTrainedModel):
    """
    BERT-based model for joint intent classification and slot filling.
    Inherits from BertPreTrainedModel for easy loading of pre-trained weights.
    """

    def __init__(self, config, num_intent_labels, num_slot_labels, dropout_prob=0.1):
        """
        Initializes the JointBERT model.

        Args:
            config: The BERT model configuration object.
            num_intent_labels (int): Number of unique intent labels.
            num_slot_labels (int): Number of unique slot labels.
            dropout_prob (float): Dropout probability for the classification layers.
        """
        super().__init__(config)
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels

        # Load the pre-trained BERT model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout_prob)

        # Classifier for intent classification (uses the [CLS] token output)
        # config.hidden_size is the dimension of BERT's output embeddings
        self.intent_classifier = nn.Linear(
            config.hidden_size, num_intent_labels)

        # Classifier for slot filling
        self.slot_classifier = nn.Linear(config.hidden_size, num_slot_labels)

        # CRF layer for slot filling
        self.crf = CRF(num_tags=num_slot_labels, batch_first=True)

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
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Batch of input token IDs.
            attention_mask (torch.Tensor): Batch of attention masks.
            token_type_ids (torch.Tensor, optional): Batch of token type IDs. Defaults to None.
            ... (other standard BERT input arguments)
            intent_labels (torch.Tensor, optional): Batch of intent labels for loss calculation. Defaults to None.
            slot_labels (torch.Tensor, optional): Batch of slot labels for loss calculation. Defaults to None.

        Returns:
            tuple or dict: Depending on `return_dict` and labels provided.
                           Contains intent logits, slot logits, and potentially loss(es).
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

        # Sequence output corresponds to the embeddings of each token in the input sequence
        # Shape: (batch_size, sequence_length, hidden_size)
        sequence_output = outputs[0]

        # Pooled output is typically derived from the [CLS] token's embedding after further processing
        # Shape: (batch_size, hidden_size)
        pooled_output = outputs[1]

        # Apply dropout for regularization
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        # Get logits
        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        # Calculate loss if labels are provided
        total_loss = None
        intent_loss = None
        slot_loss = None

        if intent_labels is not None and slot_labels is not None:
            intent_loss_fct = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fct(
                intent_logits.view(-1, self.num_intent_labels), intent_labels.view(-1))

            # Slot Loss (using CRF which calculates the negative log likelihood loss.)
            if attention_mask is not None:
                # The mask ensures that padded tokens are ignored.
                crf_mask = attention_mask.bool()

                if crf_mask.dim() > 2:
                    # Adjust if mask has extra dims
                    crf_mask = crf_mask[:, :slot_logits.shape[1]]

                # We negate the result because optimizers minimize loss, and CRF returns log-likelihood.
                slot_loss = -self.crf(slot_logits, slot_labels,
                                      mask=crf_mask, reduction='mean')
            else:
                # Calculate loss without mask if attention_mask is None
                slot_loss = -self.crf(slot_logits,
                                      slot_labels, reduction='mean')

            # Combine losses using alpha
            total_loss = loss_alpha * intent_loss + \
                (1 - loss_alpha) * slot_loss

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
