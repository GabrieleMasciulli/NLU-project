import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from functools import partial
from utils import PennTreeBank, Lang, DEVICE, read_file
from model import LM_RNN
from functions import ensemble_eval_loop, collate_fn


def evaluate_rnn_ensemble():
    """
    Evaluates a pre-trained ensemble of RNN language models on the Penn TreeBank test set.
    Models are loaded from the './bin/ensemble_models' directory relative to this script's location
    and use specific hyperparameters for instantiation.
    """
    # --- Configuration ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_directory = os.path.join(base_dir, "bin", "ensemble_models")

    # Ensemble model filenames
    ensemble_model_filenames = [
        "best_model_member_1_seed42.pt",
        "best_model_member_2_seed43.pt",
        "best_model_member_3_seed44.pt",
        "best_model_member_4_seed45.pt"
    ]

    # Model Hyperparameters (common for all ensemble members)
    emb_size = 300
    hidden_size = 256
    compression_dim = 100
    n_layers = 1

    batch_size_eval = 16

    # Data paths
    train_file_path = os.path.join(
        base_dir, "dataset", "PennTreeBank", "ptb.train.txt")
    test_file_path = os.path.join(
        base_dir, "dataset", "PennTreeBank", "ptb.test.txt")

    # --- Load Vocabulary and Data ---
    print("Loading vocabulary and data...")
    if not os.path.exists(train_file_path):
        print(f"Error: Training data file not found at '{train_file_path}'.")
        return
    if not os.path.exists(test_file_path):
        print(f"Error: Test data file not found at '{test_file_path}'.")
        return

    train_raw = read_file(train_file_path)
    test_raw = read_file(test_file_path)

    # Lang initialization
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)
    pad_index = lang.word2id["<pad>"]

    test_dataset = PennTreeBank(test_raw, lang)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_eval,
                             collate_fn=partial(
                                 collate_fn, pad_token=pad_index),
                             shuffle=False)

    print(f"Vocabulary size: {vocab_len}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Using device: {DEVICE}")

    # --- Load Ensemble Models ---
    print("\n--- Loading Ensemble Models ---")
    ensemble_models = []
    all_models_found = True
    for model_filename in ensemble_model_filenames:
        full_model_path = os.path.join(model_directory, model_filename)
        if not os.path.exists(full_model_path):
            print(
                f"  Error: Model file not found at '{full_model_path}'. Aborting ensemble evaluation.")
            all_models_found = False
            break

        print(f"  Loading model: {model_filename}")
        model = LM_RNN(
            emb_size=emb_size,
            hidden_size=hidden_size,
            output_size=vocab_len,
            compression_dim=compression_dim,
            pad_index=pad_index,
            n_layers=n_layers
        ).to(DEVICE)

        try:
            model.load_state_dict(torch.load(
                full_model_path, map_location=DEVICE))
            model.eval()  # Set to evaluation mode
            ensemble_models.append(model)
            print(f"    Successfully loaded {model_filename}")
        except Exception as e:
            print(f"    Error loading state_dict for {model_filename}: {e}")
            all_models_found = False  # Treat as if model not found for safety
            break

    if not all_models_found or not ensemble_models:
        print("Could not load all models for the ensemble. Evaluation cannot proceed.")
        return

    if len(ensemble_models) != len(ensemble_model_filenames):
        print("Mismatch between expected and loaded models. Evaluation cannot proceed.")
        return

    # --- Evaluate Ensemble ---
    print("\n--- Evaluating Ensemble ---")
    print(
        f"  Using params: emb_size={emb_size}, hid_size={hidden_size}, comp_dim={compression_dim}, n_layers={n_layers}")

    eval_criterion = nn.CrossEntropyLoss(
        ignore_index=pad_index, reduction='sum')

    perplexity, avg_loss = ensemble_eval_loop(
        test_loader, eval_criterion, ensemble_models)

    print("\n--- Ensemble Evaluation Results ---")
    print(f"  Test Loss: {avg_loss:.4f}")
    print(f"  Test Perplexity: {perplexity:.2f}")
    print("---------------------------------")


if __name__ == "__main__":
    evaluate_rnn_ensemble()
