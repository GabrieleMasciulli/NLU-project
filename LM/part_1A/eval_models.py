import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
import os
from utils import Lang, read_file, PennTreeBank, DEVICE
from model import LM_LSTM, LSTM_Dropout
from functions import collate_fn, eval_loop


def evaluate_models():
    """
    Evaluates pre-trained LSTM language models on the Penn TreeBank test set.
    Models are loaded from the './bin' directory relative to this script's location
    and use specific hyperparameters for instantiation.
    """
    # --- Configuration ---
    # Default/common parameters
    batch_size_eval = 32

    # Data paths
    data_base_path = "dataset/PennTreeBank"
    train_file_path = os.path.join(data_base_path, "ptb.train.txt")
    test_file_path = os.path.join(data_base_path, "ptb.test.txt")

    # Models to evaluate
    model_filenames = [
        "best_model_LSTM.pt",
        "best_model_dropout.pt",
        "best_model_AdamW.pt",
    ]
    model_directory = "bin"

    # --- Validate Data Paths ---
    if not os.path.exists(train_file_path):
        print(f"Error: Training data file not found at '{train_file_path}'.")
        print(
            f"Please ensure the dataset is correctly placed. Expected in './{data_base_path}'.")
        return
    if not os.path.exists(test_file_path):
        print(f"Error: Test data file not found at '{test_file_path}'.")
        return

    # --- Load Data and Vocabulary ---
    print("Loading data and building vocabulary...")
    train_raw = read_file(train_file_path)
    test_raw = read_file(test_file_path)

    lang = Lang(train_raw, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)
    pad_index = lang.word2id["<pad>"]

    test_dataset = PennTreeBank(test_raw, lang)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_eval,
                             collate_fn=partial(collate_fn, pad_token=pad_index))
    print(f"Vocabulary size: {vocab_len}")
    print(f"Using device: {DEVICE}")

    # --- Evaluation Criterion ---
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=pad_index, reduction='sum')

    print("\n--- Starting Model Evaluation ---")

    for model_filename in model_filenames:
        print(f"\nEvaluating model: {model_filename}")

        full_model_path = os.path.join(model_directory, model_filename)

        if not os.path.exists(full_model_path):
            print(
                f"  Error: Model file not found at '{full_model_path}'. Skipping.")
            continue

        # --- Set Model-Specific Hyperparameters ---
        if model_filename == "best_model_LSTM.pt":
            emb_size = 400
            hid_size = 650
            n_layers = 1
            print(
                f"  Using params: emb_size={emb_size}, hid_size={hid_size}, n_layers={n_layers}")
            model = LM_LSTM(
                emb_size=emb_size,
                hidden_size=hid_size,
                vocab_len=vocab_len,
                pad_index=pad_index,
                n_layers=n_layers
            ).to(DEVICE)
        elif model_filename in ["best_model_dropout.pt", "best_model_AdamW.pt"]:
            emb_size = 650
            hid_size = 650
            n_layers = 2
            print(
                f"  Using params: emb_size={emb_size}, hid_size={hid_size}, n_layers={n_layers}")
            model = LSTM_Dropout(
                emb_size=emb_size,
                hidden_size=hid_size,
                vocab_len=vocab_len,
                pad_index=pad_index,
                n_layers=n_layers
            ).to(DEVICE)
        else:
            print(
                f"  Error: Unknown model type or unconfigured for {model_filename}. Skipping.")
            continue

        try:
            # Load the saved model weights into the instantiated structure
            model.load_state_dict(torch.load(
                full_model_path, map_location=DEVICE))
            model.eval()  # Set model to evaluation mode (disables dropout, etc.)

            # Perform evaluation
            # eval_loop from part_1A/functions.py returns (perplexity, loss)
            perplexity, loss = eval_loop(test_loader, criterion_eval, model)
            print(f"  Test Perplexity: {perplexity:.2f}")
            print(f"  Test Loss: {loss:.4f}")

        except Exception as e:
            print(f"  Error during evaluation of {model_filename}: {e}")
            print(
                f"  This could be due to a mismatch in model architecture hyperparameters ")
            print(
                f"  (e.g., emb_size, hid_size, n_layers used for instantiation vs. saved model) ")
            print(f"  or a corrupted model file.")

    print("\n--- Evaluation Finished ---")


if __name__ == "__main__":
    evaluate_models()
