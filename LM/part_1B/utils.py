import torch
import torch.utils.data as data
import numpy as np
import os
from tqdm import tqdm

DEVICE = torch.device("cuda:0")

# Loading the corpus


def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

# Vocab with tokens to ids


def get_vocab(corpus, special_tokens=[]):
    output = {}
    i = 0
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output

# This class computes and stores our vocab
# Word to ids and ids to word


class Lang:
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}


class PennTreeBank (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []

        for sentence in corpus:
            # We get from the first token till the second-last token
            self.source.append(sentence.split()[0:-1])
            # We get from the second token till the last token
            self.target.append(sentence.split()[1:])
            # See example in section 6.2

        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample

    # Auxiliary methods

    # Map sequences of tokens to corresponding computed in Lang class
    def mapping_seq(self, data, lang):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    print('You have to deal with that')
                    break
            res.append(tmp_seq)
        return res
