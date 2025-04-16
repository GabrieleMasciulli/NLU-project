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


def main(hid_size, emb_size, n_layers, lr, emb_dropout_rate, lstm_dropout_rate, out_dropout_rate,
         batch_size_train, batch_size_eval, epochs, clip,
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

    # --- GloVe Embeddings --- (Explicitly disabled for Zaremba config)
    pretrained_weights = None
    embedding_type = "Random Initialization U[-0.05, 0.05]"
    # Ensure use_glove is false if we are following Zaremba
    if use_glove:
        print("Warning: Zaremba config typically uses random init. Overriding use_glove to False.")
        use_glove = False

    # --- Model Initialization ---
    model = LM_LSTM(emb_size, hid_size, vocab_len,
                    pad_index=pad_index,
                    n_layers=n_layers,
                    pretrained_weights=pretrained_weights,  # Will be None
                    emb_dropout_rate=emb_dropout_rate,
                    lstm_dropout_rate=lstm_dropout_rate,
                    out_dropout_rate=out_dropout_rate).to(DEVICE)  # Use single dropout rate

    # Apply Zaremba weight initialization
    init_weights(model)

    # --- Optimizer and Loss --- #
    optimizer = optim.SGD(model.parameters(), lr=lr)  # Changed to SGD
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_index)
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=pad_index, reduction='sum')

    # --- Training Setup --- #
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    # Removed current_patience
    lr_decay_factor = 0.5  # Zaremba LR decay
    lr_decay_epoch_threshold = 6  # Zaremba LR decay threshold

    pbar = tqdm(range(1, epochs + 1))

    # --- W&B Initialization --- #
    run_name_parts = [
        f"zaremba_medium",
        f"lstm",
        f"l{n_layers}",
        f"h{hid_size}",  # emb_size is same as hid_size
        f"lr{lr}",
        f"emb_dropout{emb_dropout_rate}",
        f"lstm_dropout{lstm_dropout_rate}",
        f"out_dropout{out_dropout_rate}",
        f"clip{clip}",
        f"bptt_N/A"  # Indicate BPTT sequence length not implemented
    ]
    run_name = "_".join(run_name_parts)
    group_name = f"{wandb_group_prefix}_zaremba_medium"

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
            # Removed patience
            "clip_gradient": clip,
            "n_layers": n_layers,
            "embeddings": embedding_type,
            "emb_dropout": emb_dropout_rate,
            "lstm_dropout": lstm_dropout_rate,
            "out_dropout": out_dropout_rate,
            # Removed separate dropout entries
            "lr_decay_factor": lr_decay_factor,
            "lr_decay_epoch_threshold": lr_decay_epoch_threshold
        }
    )

    # --- Training Loop --- #
    print(f"Starting training for run: {run_name}")
    last_saved_epoch = -1
    try:
        for epoch in pbar:
            # Train one epoch
            model.train()
            epoch_train_loss = train_loop(
                train_loader, optimizer, criterion_train, model, clip)

            # Evaluate on development set
            model.eval()
            ppl_dev, epoch_dev_loss = eval_loop(
                dev_loader, criterion_eval, model)

            # Process and log metrics
            avg_train_loss = epoch_train_loss.item() if isinstance(
                epoch_train_loss, torch.Tensor) else epoch_train_loss
            avg_dev_loss = epoch_dev_loss.item() if isinstance(
                epoch_dev_loss, torch.Tensor) else epoch_dev_loss
            losses_train.append(avg_train_loss)
            losses_dev.append(avg_dev_loss)
            sampled_epochs.append(epoch)

            # Get current LR for logging
            current_lr = optimizer.param_groups[0]['lr']

            pbar.set_description(
                f"Epoch {epoch} | LR: {current_lr:.4f} | Train Loss: {avg_train_loss:.4f} | Dev PPL: {ppl_dev:.2f}")

            run.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "dev_loss": avg_dev_loss,
                "dev_perplexity": ppl_dev,
                "learning_rate": current_lr
            })

            # --- Zaremba LR Schedule & Best Model Saving --- #
            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                last_saved_epoch = epoch
                print(
                    f"  New best model found! Dev PPL: {best_ppl:.2f}. Saving model.")
            elif epoch >= lr_decay_epoch_threshold:
                # Decay LR if validation perplexity hasn't improved
                print(
                    f"  No improvement in Dev PPL ({ppl_dev:.2f} vs best {best_ppl:.2f}). Decaying learning rate.")
                current_lr *= lr_decay_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            # Removed patience-based early stopping check

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    # --- Final Evaluation on Test Set --- #
    print("\nTraining finished.")
    if best_model is not None:
        print(
            f"Evaluating best model (from epoch {last_saved_epoch}, PPL: {best_ppl:.2f}) on test set...")
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
    # --- Zaremba Medium LSTM Hyperparameters --- #
    hid_size = 650
    emb_size = 650  # Zaremba uses tied emb/hid size
    n_layers = 2
    lr = 1.0  # Initial LR for SGD
    emb_dropout_rate = 0.1
    lstm_dropout_rate = 0.0
    out_dropout_rate = 0.1
    batch_size_train = 20
    batch_size_eval = 20
    epochs = 39  # Zaremba paper ran for 39 epochs
    clip = 5.0  # Gradient clipping max norm
    use_glove = False  # Zaremba used random init
    wandb_project = "NLU-project"
    wandb_group_prefix = "zaremba_medium_SGD"  # Updated group prefix

    # --- GloVe Setup (Should be skipped if use_glove is False) --- #
    glove_path = None
    if use_glove:
        # This block should not run with Zaremba config, but kept for completeness
        expected_glove_file = "glove.6B.300d.txt"
        print(f"Attempting to download/use GloVe 300d...")
        glove_path = download_and_extract_glove(
            GLOVE_URL, DATA_DIR, expected_glove_file)
        if not glove_path:
            print(
                f"Could not obtain {expected_glove_file}. You might need to download {GLOVE_URL} manually.")
            print("Continuing without GloVe embeddings.")
            # use_glove = False # Override if download failed - already False
        else:
            # This would override Zaremba's emb_size if GloVe were used
            # emb_size = 300
            pass
    else:
        print("Using random weight initialization U[-0.05, 0.05], not GloVe.")

    # --- Login to W&B --- #
    try:
        wandb.login()
    except Exception as e:
        print(f"Could not login to WandB: {e}. Proceeding without logging.")

    # --- Run Main Function --- #
    main(
        hid_size=hid_size,
        emb_size=emb_size,
        n_layers=n_layers,
        lr=lr,
        emb_dropout_rate=emb_dropout_rate,
        lstm_dropout_rate=lstm_dropout_rate,
        out_dropout_rate=out_dropout_rate,
        batch_size_train=batch_size_train,
        batch_size_eval=batch_size_eval,
        epochs=epochs,
        clip=clip,
        glove_path=glove_path,
        use_glove=use_glove,
        wandb_project=wandb_project,
        wandb_group_prefix=wandb_group_prefix
    )
