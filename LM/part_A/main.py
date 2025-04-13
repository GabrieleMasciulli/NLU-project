from functools import partial
import math
import torch.optim as optim
import torch.nn as nn
from utils import Lang, read_file
from functions import collate_fn, init_weights, train_loop, eval_loop
from model import LM_LSTM
from utils import DEVICE, PennTreeBank, load_glove_embeddings
import wandb
from tqdm import tqdm
import copy
import numpy as np
from torch.utils.data import DataLoader
import torch
import os
import requests
import zipfile

GLOVE_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
DATA_DIR = "data"
EXPECTED_GLOVE_FILE = "glove.6B.300d.txt"


def download_and_extract_glove(url, download_dir, expected_filename):
    """Downloads and extracts the specified GloVe file if it doesn't exist."""
    os.makedirs(download_dir, exist_ok=True)
    glove_txt_path = os.path.join(download_dir, expected_filename)

    if os.path.exists(glove_txt_path):
        print(
            f"GloVe file '{expected_filename}' already exists in '{download_dir}'. Skipping download.")
        return glove_txt_path

    zip_filename = url.split('/')[-1]
    zip_filepath = os.path.join(download_dir, zip_filename)

    print(f"Downloading GloVe embeddings from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 8192  # 8KB
            progress_bar = tqdm(total=total_size, unit='iB',
                                unit_scale=True, desc=f"Downloading {zip_filename}")
            with open(zip_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
            if total_size != 0 and progress_bar.n != total_size:
                print("ERROR, something went wrong during download")
                return None

        print(f"Extracting '{expected_filename}' from {zip_filename}...")
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            # Ensure the specific file exists in the archive before extracting
            if expected_filename in zip_ref.namelist():
                zip_ref.extract(expected_filename, path=download_dir)
                print(f"Successfully extracted '{expected_filename}'.")
            else:
                print(
                    f"Error: '{expected_filename}' not found in the downloaded zip file.")
                return None

    except requests.exceptions.RequestException as e:
        print(f"Error downloading GloVe embeddings: {e}")
        return None
    except zipfile.BadZipFile:
        print(
            f"Error: Downloaded file '{zip_filename}' is not a valid zip file.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    finally:
        # Clean up the zip file
        if os.path.exists(zip_filepath):
            print(f"Removing downloaded zip file: {zip_filepath}")
            os.remove(zip_filepath)

    return glove_txt_path


if __name__ == "__main__":
    wandb.login()

    # Ensure GloVe embeddings are downloaded and extracted
    glove_path = download_and_extract_glove(
        GLOVE_URL, DATA_DIR, EXPECTED_GLOVE_FILE)
    if not glove_path:
        print("Could not obtain GloVe embeddings. Exiting.")
        exit()

    # Load the data
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    # Create the vocabulary
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    # Create the datasets
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # Dataloader instantiation
    # You can reduce the batch_size if the GPU memory is not enough
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(
        collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(
        collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(
        collate_fn, pad_token=lang.word2id["<pad>"]))

    # Define the model parameters
    hid_size = 256
    emb_size = 300  # Set embedding size to GloVe dimension (e.g., 300)
    n_layers = 1  # Number of LSTM layers
    lr = 1
    # glove_path is now set by download_and_extract_glove
    # glove_path = "data/glove.6B.300d.txt" # <-- ADJUST THIS PATH IF NEEDED

    # Load GloVe embeddings
    pretrained_weights = load_glove_embeddings(
        glove_path, lang.word2id, emb_size)
    if pretrained_weights is None:
        # Handle error: Exit or fallback to random initialization
        print("Failed to load GloVe embeddings. Exiting.")
        exit()  # Or potentially fallback: pretrained_weights = None

    # Get the vocabulary length
    vocab_len = len(lang.word2id)

    # Instantiate the model with pre-trained embeddings
    model = LM_LSTM(emb_size, hid_size, vocab_len,
                    pad_index=lang.word2id["<pad>"],
                    n_layers=n_layers,
                    pretrained_weights=pretrained_weights).to(DEVICE)

    # Define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=lang.word2id["<pad>"], reduction='sum')

    # Define the training parameters
    clip = 5  # Clip the gradient
    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1, n_epochs))

    # group_name = "lstm_num_layers" # You might want to change the group name
    group_name = "lstm_glove_embeddings"
    run_name = f"n_layers_{n_layers}_emb_{emb_size}_glove"

    # Initialize W&B
    run = wandb.init(
        project="NLU-project",
        group=group_name,
        name=run_name,
        config={
            "learning_rate": lr,
            "batch_size": 64,
            "hidden_size": hid_size,
            "embedding_size": emb_size,  # Updated emb_size
            "optimizer": "SGD",
            "epochs": n_epochs,
            "patience": patience,
            "n_layers": n_layers,
            "embeddings": "GloVe 6B 300d"  # Log embedding type
        }
    )

    for epoch in pbar:
        loss = train_loop(train_loader, optimizer,
                          criterion_train, model, clip)

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            # Ensure loss is correctly handled (e.g., converting tensor to float if needed)
            current_train_loss = loss.item() if isinstance(loss, torch.Tensor) else loss
            losses_train.append(current_train_loss)
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            # Ensure loss_dev is correctly handled
            current_dev_loss = loss_dev.item() if isinstance(
                loss_dev, torch.Tensor) else loss_dev
            losses_dev.append(current_dev_loss)
            pbar.set_description("PPL: %f" % ppl_dev)

            # Log values to W&B
            run.log({
                "epoch": epoch,
                "train_loss": current_train_loss,  # Log mean train loss for the epoch
                "dev_loss": current_dev_loss,     # Log mean dev loss for the epoch
                "dev_perplexity": ppl_dev
            })

            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3  # Reset patience
            else:
                patience -= 1

            # Potentially adjust learning rate if using a scheduler or based on patience
            # Example: if patience == 1: optimizer.param_groups[0]['lr'] *= 0.5

            if patience <= 0:  # Early stopping with patience
                print(f"Early stopping triggered at epoch {epoch}!")
                break

    if best_model is not None:
        best_model.to(device=DEVICE)
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
        print('Test ppl: ', final_ppl)
        # Save the best model
        # Consider adding experiment details to the filename
        model_save_path = f'bin/best_model_glove_l{n_layers}_h{hid_size}.pt'
        torch.save(best_model.state_dict(), model_save_path)
        print(f"Best model saved to {model_save_path}")
        # Log final test perplexity to W&B
        run.log({"test_perplexity": final_ppl})
    else:
        print(
            'No best model found - training did not improve significantly or stopped early.')

    wandb.finish()
