import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
import os
from utils import Lang, read_file, PennTreeBank, DEVICE, load_glove_embeddings, download_and_extract_glove
from functions import collate_fn, eval_loop
from model import LSTM_WT, LSTM_VarDrop

BATCH_SIZE_EVAL = 10


def evaluate_models():
    # --- Data Loading ---
    print("Loading data...")
    try:
        train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
        test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")
    except FileNotFoundError:
        print("Error: Penn Treebank dataset files not found in 'dataset/PennTreeBank/' directory.")
        print("Please ensure ptb.train.txt and ptb.test.txt are present.")
        exit()

    lang = Lang(train_raw, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)
    pad_index = lang.word2id["<pad>"]

    test_dataset = PennTreeBank(test_raw, lang)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_EVAL, collate_fn=partial(
        collate_fn, pad_token=pad_index), shuffle=False)

    # --- Criterion for Evaluation ---
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=pad_index, reduction='sum')

    # --- Model Configurations ---
    model_configs = [
        {
            "name": "LSTM_WT (650emb, 650hid, 3L)",
            "path": "bin/best_model_wt.pt",
            "type": "LSTM_WT",
            "emb_size": 650,
            "hidden_size": 650,
            "n_layers": 3,
            "use_glove": False,
        },
        {
            "name": "LSTM_VarDrop (650emb, 650hid, 3L)",
            "path": "bin/best_model_var_drop.pt",
            "type": "LSTM_VarDrop",
            "emb_size": 650,
            "hidden_size": 650,
            "n_layers": 3,
            "use_glove": False,
        },
        {
            "name": "LSTM_VarDrop_GloVe (300emb, 1150hid, 3L)",
            "path": "bin/best_model_glove.pt",
            "type": "LSTM_VarDrop",
            "emb_size": 300,
            "hidden_size": 1150,
            "n_layers": 3,
            "use_glove": True,
        },
    ]

    print(f"\nEvaluating models on {DEVICE}...")
    # --- Evaluation Loop ---
    for config in model_configs:
        print(f"\n--- Evaluating: {config['name']} ---")

        model_path = config["path"]
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            continue

        emb_size = config["emb_size"]
        glove_embeddings_tensor = None

        if config["use_glove"]:
            print(f"Loading GloVe embeddings (dim: {emb_size})...")
            glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"
            glove_dir = "glove"
            glove_filename = f"glove.6B.{emb_size}d.txt"

            # Ensure the glove directory for extraction exists
            os.makedirs(glove_dir, exist_ok=True)

            glove_txt_path = download_and_extract_glove(
                glove_url, glove_dir, glove_filename)
            if glove_txt_path:
                glove_embeddings_tensor = load_glove_embeddings(
                    glove_txt_path, lang.word2id, emb_size)
                if glove_embeddings_tensor is None:
                    print(
                        f"Warning: Failed to load GloVe embeddings for {config['name']}. Proceeding without them.")
            else:
                print(
                    f"Warning: GloVe file {glove_filename} not found or couldn't be downloaded/extracted. Proceeding without them for {config['name']}.")

        # Instantiate model
        if config["type"] == "LSTM_WT":
            model = LSTM_WT(
                emb_size=config["emb_size"],
                hidden_size=config["hidden_size"],
                vocab_len=vocab_len,
                pad_index=pad_index,
                n_layers=config["n_layers"]
            )
        elif config["type"] == "LSTM_VarDrop":
            model = LSTM_VarDrop(
                emb_size=config["emb_size"],
                hidden_size=config["hidden_size"],
                vocab_len=vocab_len,
                pad_index=pad_index,
                n_layers=config["n_layers"],
                pretrained_embeddings=glove_embeddings_tensor,
                freeze_embeddings=True
            )
        else:
            print(
                f"Error: Unknown model type '{config['type']}' for {config['name']}")
            continue

        try:
            print(f"Loading model state from {model_path}...")
            # Load the saved state_dict
            # Ensure the model is loaded to the correct device if saved on a different one
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        except Exception as e:
            print(f"Error loading model state for {config['name']}: {e}")
            continue

        model.to(DEVICE)
        model.eval()  # Set model to evaluation mode (disables dropout, etc.)

        print("Starting evaluation...")
        perplexity, _ = eval_loop(test_loader, criterion_eval, model)
        print(f"Test Perplexity for {config['name']}: {perplexity:.2f}")

    print("\nAll evaluations complete.")


if __name__ == "__main__":
    evaluate_models()
