from collections import Counter
import os
import json
import torch
from torch.utils import data

# global variables
DEVICE = 'cuda:0'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # used to debug CUDA errors
PAD_TOKEN = 0


def load_data(path):
    """
    Input:
        path: path to the file
    Output:
        dataset: json file
    """
    dataset = []
    with open(path, "r") as f:
        dataset = json.loads(f.read())
    return dataset


class Lang():
    """
    This class computes and stores our vocab
    Word to ids and ids to word
    """

    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def w2id(self, elements, cutoff=None, unk=True):
        """
        This function creates a dictionary of words to ids.

        Input:
            elements: list of elements
            cutoff: frequency cutoff
            unk: whether to add <unk> token
        Output:
            vocab: dictionary of word to id
        """
        vocab = {'pad': PAD_TOKEN}

        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)

        for word, freq in count.items():
            if freq > cutoff:
                vocab[word] = len(vocab)

        return vocab

    def lab2id(self, elements, pad=True):
        """
        This function creates a dictionary of labels to ids.

        Input:
            elements: list of elements
            pad: whether to add <pad> token
        Output:
            vocab: dictionary of label to id
        """
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab


class IntentsAndSlots(data.Dataset):
    """
    This class creates a dataset for the intent and slot tagging task.
    """

    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk="unk"):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk

        for sample in dataset:
            self.utterances.append(sample['utterance'])
            self.intents.append(sample['intent'])
            self.slots.append(sample['slots'])

        self.utterances_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_seq(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utterances_ids[idx])
        slot = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slot, 'intent': intent}
        return sample

    # Auxiliary methods
    def mapping_seq(self, data, mapper):
        """
        This function maps a sequence of elements to their corresponding ids.
        Input:
            data: list of sequences
            mapper: dictionary of element to id
        Output:
            output: list of sequences of ids
        """
        output = []
        for seq in data:
            tmp_seq = []
            for elem in seq.split():
                if elem in mapper:
                    tmp_seq.append(mapper[elem])
                else:
                    tmp_seq.append(mapper[self.unk])
            output.append(tmp_seq)
        return output
