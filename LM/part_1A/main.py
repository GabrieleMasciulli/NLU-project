from functools import partial
import math
import torch.optim as optim
import torch.nn as nn
from utils import Lang, read_file
from functions import collate_fn, init_weights, train_loop, eval_loop
from model import LM_LSTM
from utils import DEVICE, PennTreeBank, load_glove_embeddings, download_and_extract_glove
import wandb
from tqdm import tqdm
import copy
import numpy as np
from torch.utils.data import DataLoader
import torch
import os


GLOVE_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
DATA_DIR = "data"
EXPECTED_GLOVE_FILE = "glove.6B.300d.txt"


def main(hid_size, emb_size, n_layers, lr, emb_dropout_rate, out_dropout_rate,
         batch_size_train, batch_size_eval, epochs, patience, clip,
         glove_path, use_glove, wandb_project, wandb_group_prefix):
    """Main function to train and evaluate the LSTM Language Model."""

    # --- Data Loading and Preprocessing ---
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    lang = Lang(train_raw, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)
    pad_index = lang.word2id["<pad>"]

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, collate_fn=partial(
        collate_fn, pad_token=pad_index), shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size_eval, collate_fn=partial(
        collate_fn, pad_token=pad_index))
    test_loader = DataLoader(test_dataset, batch_size=batch_size_eval, collate_fn=partial(
        collate_fn, pad_token=pad_index))

    # --- GloVe Embeddings --- (Load if specified)
    pretrained_weights = None
    embedding_type = "Random Initialization"
    if use_glove:
        pretrained_weights = load_glove_embeddings(
            glove_path, lang.word2id, emb_size)
        if pretrained_weights is None:
            print(
                "Failed to load GloVe embeddings specified, falling back to random initialization.")
            # Optionally exit or force random init by keeping pretrained_weights=None
            use_glove = False  # Ensure config reflects the fallback
        else:
            embedding_type = f"GloVe {emb_size}d"

    # --- Model Initialization ---
    model = LM_LSTM(emb_size, hid_size, vocab_len,
                    pad_index=pad_index,
                    n_layers=n_layers,
                    pretrained_weights=pretrained_weights,
                    emb_dropout_rate=emb_dropout_rate,
                    out_dropout_rate=out_dropout_rate).to(DEVICE)

    # --- Optimizer and Loss --- #
    # Consider using AdamW or Adam for potentially faster/better convergence
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_index)
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=pad_index, reduction='sum')

    # --- Training Setup --- #
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    current_patience = patience
    pbar = tqdm(range(1, epochs + 1))  # Correct range to include final epoch

    # --- W&B Initialization --- #
    run_name_parts = [
        f"lstm",
        f"l{n_layers}",
        f"h{hid_size}",
        f"emb{emb_size}",
        f"lr{lr}",
        f"do_emb{emb_dropout_rate}",
        f"do_out{out_dropout_rate}",
        f"glove{use_glove}"
    ]
    run_name = "_".join(run_name_parts)
    group_name = f"{wandb_group_prefix}_h{hid_size}_l{n_layers}"

    run = wandb.init(
        project=wandb_project,
        group=group_name,
        name=run_name,
        config={
            "learning_rate": lr,
            "batch_size_train": batch_size_train,
            "batch_size_eval": batch_size_eval,
            "hidden_size": hid_size,
            "embedding_size": emb_size,
            "optimizer": type(optimizer).__name__,
            "epochs": epochs,
            "patience": patience,
            "clip_gradient": clip,
            "n_layers": n_layers,
            "embeddings": embedding_type,
            "emb_dropout": emb_dropout_rate,
            "out_dropout": out_dropout_rate
        }
    )

    # --- Training Loop --- #
    print(f"Starting training for run: {run_name}")
    for epoch in pbar:
        # Train one epoch
        model.train()
        epoch_train_loss = train_loop(
            train_loader, optimizer, criterion_train, model, clip)

        # Evaluate on development set
        model.eval()
        ppl_dev, epoch_dev_loss = eval_loop(dev_loader, criterion_eval, model)

        # Process and log metrics
        avg_train_loss = epoch_train_loss.item() if isinstance(
            epoch_train_loss, torch.Tensor) else epoch_train_loss
        avg_dev_loss = epoch_dev_loss.item() if isinstance(
            epoch_dev_loss, torch.Tensor) else epoch_dev_loss
        losses_train.append(avg_train_loss)
        losses_dev.append(avg_dev_loss)
        sampled_epochs.append(epoch)

        pbar.set_description(
            f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Dev PPL: {ppl_dev:.2f}")

        run.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "dev_loss": avg_dev_loss,
            "dev_perplexity": ppl_dev
        })

        # --- Early Stopping and Best Model Saving --- #
        if ppl_dev < best_ppl:
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            current_patience = patience  # Reset patience
            print(f"  New best model found! Dev PPL: {best_ppl:.2f}")
            # Save checkpoint of best model immediately (optional)
            # best_model_path = f'bin/best_model_checkpoint_epoch{epoch}.pt'
            # torch.save(best_model.state_dict(), best_model_path)
            # print(f"  Best model checkpoint saved to {best_model_path}")
        else:
            current_patience -= 1
            print(f"  No improvement. Patience left: {current_patience}")

        # Optional: Learning rate scheduling based on patience or validation loss
        # if current_patience == patience // 2: optimizer.param_groups[0]['lr'] *= 0.5

        if current_patience <= 0:
            print(f"Early stopping triggered at epoch {epoch}!")
            break

    # --- Final Evaluation on Test Set --- #
    print("\nTraining finished.")
    if best_model is not None:
        print(
            f"Evaluating best model (from epoch {sampled_epochs[np.argmin(losses_dev)] if losses_dev else 'N/A'}, PPL: {best_ppl:.2f}) on test set...")
        best_model.to(device=DEVICE)
        best_model.eval()  # Ensure model is in eval mode
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        print(f'Final Test Perplexity: {final_ppl:.2f}')

        # Save the final best model
        os.makedirs('bin', exist_ok=True)  # Ensure bin directory exists
        model_save_path = f'bin/best_model_{run_name}.pt'
        torch.save(best_model.state_dict(), model_save_path)
        print(f"Best model saved to {model_save_path}")

        # Log final test perplexity to W&B
        run.log({"test_perplexity": final_ppl})
    else:
        print('No best model found - training might have diverged or stopped very early.')

    wandb.finish()
    print("Run finished.")


if __name__ == "__main__":
    # --- Hardcoded Hyperparameters ---
    hid_size = 256
    # Default embedding size (will be overwritten by GloVe if used)
    emb_size = 300
    n_layers = 1
    lr = 1.0
    emb_dropout_rate = 0.25
    out_dropout_rate = 0.5
    batch_size_train = 64
    batch_size_eval = 128
    epochs = 100
    patience = 3
    clip = 5.0
    use_glove = True  # Set to False to train without GloVe
    wandb_project = "NLU-project"
    wandb_group_prefix = "lstm_glove"

    # --- GloVe Setup ---
    glove_path = None
    if use_glove:
        # Determine GloVe path based on hardcoded dimension
        expected_glove_file = "glove.6B.300d.txt"
        print(f"Attempting to download/use GloVe 300d...")
        glove_path = download_and_extract_glove(
            GLOVE_URL, DATA_DIR, expected_glove_file)
        if not glove_path:
            print(
                f"Could not obtain {expected_glove_file}. You might need to download {GLOVE_URL} manually.")
            print("Continuing without GloVe embeddings.")
            use_glove = False  # Override if download failed
        else:
            # Ensure emb_size matches the GloVe dimension if using GloVe
            emb_size = 300
    else:
        print("Not using GloVe embeddings.")

    # --- Login to W&B --- #
    try:
        wandb.login()
    except Exception as e:
        print(f"Could not login to WandB: {e}. Proceeding without logging.")
        # wandb.init(mode="disabled") # Uncomment to explicitly disable

    # --- Run Main Function --- #
    main(
        hid_size=hid_size,
        emb_size=emb_size,
        n_layers=n_layers,
        lr=lr,
        emb_dropout_rate=emb_dropout_rate,
        out_dropout_rate=out_dropout_rate,
        batch_size_train=batch_size_train,
        batch_size_eval=batch_size_eval,
        epochs=epochs,
        patience=patience,
        clip=clip,
        glove_path=glove_path,
        use_glove=use_glove,
        wandb_project=wandb_project,
        wandb_group_prefix=wandb_group_prefix
    )
