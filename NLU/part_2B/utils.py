import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizerFast

BERT_MODEL_NAME = 'bert-base-uncased'
# Use -100 to ignore tokens in CrossEntropyLoss
SLOT_PAD_LABEL_ID = -100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_TOKEN = 0  # BERT uses 0 for padding input_ids

tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)


class Lang():
    """
    This class will now primarily hold the intent and slot label mappings.
    Word vocabulary is handled by the BERT tokenizer.
    """

    def __init__(self, intents, slots):
        # Intent mapping (no padding needed)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2intent = {v: k for k, v in self.intent2id.items()}
        # Slot mapping (add pad label for sub-tokens/padding)
        # Use a specific pad label if needed for consistency
        self.slot2id = self.lab2id(slots, pad=True, pad_label='[PAD]')
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        # Add the special ignore index label
        self.slot2id['[IGNORE]'] = SLOT_PAD_LABEL_ID

    def lab2id(self, elements, pad=True, pad_label='[PAD]'):
        """
        Creates a dictionary mapping labels to IDs.
        """
        vocab = {}
        if pad:
            vocab[pad_label] = 0
        for elem in elements:
            if elem not in vocab:
                vocab[elem] = len(vocab)
        return vocab


class IntentsAndSlots(data.Dataset):
    """
    Dataset class adapted for BERT.
    Handles tokenization, sub-token alignment, and padding mask generation.
    """

    def __init__(self, dataset, lang: Lang, tokenizer: BertTokenizerFast):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.lang = lang
        self.tokenizer = tokenizer

        for sample in dataset:
            self.utterances.append(sample['utterance'])
            self.intents.append(sample['intent'])
            # Keep slots as list of words for alignment
            self.slots.append(sample['slots'].split())

        self.intent_ids = [self.lang.intent2id[intent]
                           for intent in self.intents]

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utterance = self.utterances[idx]
        intent_id = self.intent_ids[idx]
        slot_labels = self.slots[idx]

        # Tokenize utterance and align labels
        tokenized_inputs = self.tokenizer(
            utterance,
            return_tensors="pt",  # Return PyTorch tensors
            padding='do_not_pad',  # We'll pad in collate_fn
            truncation=True,  # Truncate long sequences
            return_offsets_mapping=True  # Needed for label alignment
        )

        input_ids = tokenized_inputs["input_ids"].squeeze(
            0)  # Remove batch dim
        attention_mask = tokenized_inputs["attention_mask"].squeeze(0)
        offset_mapping = tokenized_inputs["offset_mapping"].squeeze(
            0).tolist()  # For alignment

        # Align slot labels to tokens
        aligned_slot_ids = self._align_labels_with_tokens(
            slot_labels, offset_mapping)
        aligned_slot_ids_tensor = torch.LongTensor(aligned_slot_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'intent_id': torch.tensor(intent_id, dtype=torch.long),
            'slot_labels': aligned_slot_ids_tensor
        }

    def _align_labels_with_tokens(self, word_labels, offset_mapping):
        """
        Aligns word-level slot labels to BERT token-level, assigning SLOT_PAD_LABEL_ID
        to special tokens ([CLS], [SEP]) and subsequent sub-tokens.
        """
        aligned_labels = [SLOT_PAD_LABEL_ID] * len(offset_mapping)
        word_idx = 0
        for i, offset in enumerate(offset_mapping):
            if offset == (0, 0):  # Special token ([CLS], [SEP], [PAD])
                continue

            # If it's the start of a new word (offset[0] is 0)
            if offset[0] == 0 and word_idx < len(word_labels):
                # Assign label to the first sub-token of the word
                label = word_labels[word_idx]
                aligned_labels[i] = self.lang.slot2id.get(
                    label, self.lang.slot2id['O'])  # Default to 'O' if label not found
                word_idx += 1
            # Subsequent sub-tokens of the same word get the ignore index
            # (Handled by initializing aligned_labels with SLOT_PAD_LABEL_ID)

        return aligned_labels


def load_data(path):
    """
    Loads data from JSON file.
    """
    import json
    with open(path, 'r') as f:
        data = json.load(f)

    for item in data:
        if isinstance(item.get('utterance'), list):
            item['utterance'] = " ".join(item['utterance'])
        if isinstance(item.get('slots'), list):
            item['slots'] = " ".join(item['slots'])
    return data


def collate_fn(batch, pad_token_id=tokenizer.pad_token_id, slot_pad_label_id=SLOT_PAD_LABEL_ID):
    """
    Collate function for DataLoader to handle padding.

    Args:
        batch (list): A list of dictionaries, where each dictionary is an output
                      from the IntentsAndSlots dataset's __getitem__.
        pad_token_id (int): The ID used for padding input sequences.
        slot_pad_label_id (int): The ID used for padding slot label sequences (-100).

    Returns:
        dict: A dictionary containing batched and padded tensors for
              'input_ids', 'attention_mask', 'intent_id', 'slot_labels'.
    """
    # Separate components of the batch
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]
    slot_labels_list = [item['slot_labels'] for item in batch]
    # Intents are single values, just stack them
    intent_ids = torch.stack([item['intent_id'] for item in batch])

    # Pad sequences
    padded_input_ids = pad_sequence(
        input_ids_list, batch_first=True, padding_value=pad_token_id
    )
    padded_attention_mask = pad_sequence(
        # Attention mask uses 0 for padding
        attention_mask_list, batch_first=True, padding_value=0
    )
    padded_slot_labels = pad_sequence(
        # Slot labels use -100 for padding/ignored tokens
        slot_labels_list, batch_first=True, padding_value=slot_pad_label_id
    )

    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'intent_id': intent_ids,
        'slot_labels': padded_slot_labels
    }
