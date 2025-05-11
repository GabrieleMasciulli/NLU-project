import os
import torch
from utils import DEVICE, PAD_TOKEN, IntentsAndSlots, Lang, load_data
from functions import collate_fn, eval_loop
from model import ModelIAS
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split


def build_lang_and_data():
    train_raw = load_data(os.path.join("dataset", "ATIS", "train.json"))
    test_raw = load_data(os.path.join("dataset", "ATIS", "test.json"))

    portion = 0.10
    intents = [sample['intent'] for sample in train_raw]
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(train_raw[id_y])

    X_train, X_dev, y_train, y_dev = train_test_split(
        inputs, labels, test_size=portion, stratify=labels, shuffle=True, random_state=42)

    X_train.extend(mini_train)
    dev_raw = X_dev

    # Build vocab from train + dev + test, as in main.py
    corpus = X_train + dev_raw + test_raw
    words = sum([sample['utterance'].split() for sample in X_train], [])
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])
    lang = Lang(words, intents, slots, cutoff=0)

    test_dataset = IntentsAndSlots(test_raw, lang)
    test_loader = DataLoader(test_dataset, batch_size=64,
                             shuffle=False, collate_fn=collate_fn)
    return lang, test_loader


def evaluate_model(model_path, hid_size, emb_size, n_layers, fc_dropout, lstm_dropout):
    lang, test_loader = build_lang_and_data()
    out_slots = len(lang.slot2id)
    out_intents = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    model = ModelIAS(
        hid_size=hid_size,
        emb_size=emb_size,
        vocab_len=vocab_len,
        lstm_dropout=lstm_dropout,
        fc_dropout=fc_dropout,
        out_slot=out_slots,
        out_int=out_intents,
        n_layers=n_layers,
        pad_index=PAD_TOKEN
    ).to(DEVICE)

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    import torch.nn as nn
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    results_test, intent_test, _ = eval_loop(
        test_loader, criterion_slots, criterion_intents, model, lang)
    print(f"Results for {model_path}:")
    print('Slot F1:', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])
    print('-' * 40)


if __name__ == "__main__":
    # Model 1: bin/best_model_lstm_bidir.pt
    evaluate_model(
        model_path="bin/best_model_lstm_bidir.pt",
        hid_size=512,
        emb_size=200,
        n_layers=1,
        fc_dropout=0.0,
        lstm_dropout=0.0
    )

    # Model 2: bin/best_model_lstm_drop.pt
    evaluate_model(
        model_path="bin/best_model_lstm_drop.pt",
        hid_size=500,
        emb_size=300,
        n_layers=2,
        fc_dropout=0.0,
        lstm_dropout=0.0
    )
