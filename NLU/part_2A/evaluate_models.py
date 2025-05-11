import os
import torch
import pickle
from utils import DEVICE, PAD_TOKEN, IntentsAndSlots, Lang, load_data
from functions import collate_fn, eval_loop
from model import ModelIAS
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split


def build_lang_and_data(lang_path):
    test_raw = load_data(os.path.join("dataset", "ATIS", "test.json"))
    # Load the specified Lang object
    with open(lang_path, 'rb') as f:
        lang = pickle.load(f)
    test_dataset = IntentsAndSlots(test_raw, lang)
    test_loader = DataLoader(test_dataset, batch_size=64,
                             shuffle=False, collate_fn=collate_fn)
    return lang, test_loader

def evaluate_model(model_path, lang_path, hid_size, emb_size, n_layers, fc_dropout, lstm_dropout):
    lang, test_loader = build_lang_and_data(lang_path)
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
        lang_path="bin/lang_bidir.pkl",
        hid_size=512,
        emb_size=200,
        n_layers=1,
        fc_dropout=0.0,
        lstm_dropout=0.0
    )

    # Model 2: bin/best_model_lstm_dropout.pt
    evaluate_model(
        model_path="bin/best_model_lstm_dropout.pt",
        lang_path="bin/lang_dropout.pkl",
        hid_size=500,
        emb_size=300,
        n_layers=2,
        fc_dropout=0.3,
        lstm_dropout=0.2
    )
