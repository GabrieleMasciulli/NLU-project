import os
import torch
import pickle
from utils import DEVICE, IntentsAndSlots, load_data, collate_fn
from functions import eval_loop
from model import CTRAN_INSPIRED
from torch.utils.data import DataLoader
from transformers import BertConfig


def build_lang_and_data(lang_path):
    test_raw = load_data(os.path.join("dataset", "ATIS", "test.json"))
    # Load the specified Lang object
    with open(lang_path, 'rb') as f:
        lang = pickle.load(f)
    test_dataset = IntentsAndSlots(test_raw, lang)
    test_loader = DataLoader(test_dataset, batch_size=64,
                             shuffle=False, collate_fn=collate_fn)
    return lang, test_loader


def evaluate_model(model_path, lang_path, bert_model_name, dropout_prob):
    lang, test_loader = build_lang_and_data(lang_path)
    num_intent_labels = len(lang.intent2id)
    num_slot_labels = len(lang.slot2id)

    config = BertConfig.from_pretrained(
        bert_model_name, num_labels=num_intent_labels)
    model = CTRAN_INSPIRED.from_pretrained(
        bert_model_name,
        config=config,
        num_intent_labels=num_intent_labels,
        num_slot_labels=num_slot_labels,
        dropout_prob=dropout_prob
    ).to(DEVICE)

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    results_test = eval_loop(model, test_loader, lang, is_test=True)
    print(f"Results for {model_path}:")
    print('Slot F1 (Macro):', results_test['slot_f1_macro'])
    print('Slot F1 (Micro):', results_test['slot_f1_micro'])
    print('Intent Accuracy:', results_test['intent_acc'])
    print('-' * 40)


if __name__ == "__main__":
    evaluate_model(
        model_path="bin/best_model_CTRAN_multiple_kernels.pt",
        lang_path="bin/lang_CTRAN_multiple_kernels.pkl",
        bert_model_name="bert-base-uncased",
        dropout_prob=0.1
    )
