from functools import partial
import math
import torch.optim as optim
import torch.nn as nn
from utils import Lang, read_file
from functions import collate_fn, init_weights, train_loop, eval_loop
from model import LM_LSTM
from utils import DEVICE, PennTreeBank
import wandb
from tqdm import tqdm
import copy
import numpy as np
from torch.utils.data import DataLoader
import torch
import os


def main(hid_size, emb_size, n_layers, lr, emb_dropout_rate, lstm_dropout_rate, out_dropout_rate,
         batch_size_train, batch_size_eval, epochs, clip, weight_decay, patience, asgd_trigger_epochs, wandb_project, wandb_group_prefix):
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

    # --- Model Initialization ---
    # Initialize with random weights U[-0.05, 0.05] (handled by init_weights)
    embedding_type = "Random Initialization U[-0.05, 0.05]"
    model = LM_LSTM(emb_size, hid_size, vocab_len,
                    pad_index=pad_index,
                    n_layers=n_layers,
                    emb_dropout_rate=emb_dropout_rate,
                    lstm_dropout_rate=lstm_dropout_rate,
                    out_dropout_rate=out_dropout_rate).to(DEVICE)  # Use single dropout rate

    # Apply Zaremba weight initialization
    init_weights(model)

    # --- Optimizer and Loss --- #
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_index)
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=pad_index, reduction='sum')

    # --- Training Setup --- #
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    current_patience = 0
    asgd_switched = False

    pbar = tqdm(range(1, epochs + 1))

    # --- W&B Initialization --- #
    run_name_parts = [
        f"lstm",
        f"l{n_layers}",
        f"h{hid_size}",
        f"emb_dropout{emb_dropout_rate}",
        f"out_dropout{out_dropout_rate}",
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
            "weight_decay": weight_decay,
            "clip_gradient": clip,
            "patience": patience,
            "n_layers": n_layers,
            "asgd_trigger_epochs": asgd_trigger_epochs,
            "emb_dropout": emb_dropout_rate,
            "lstm_dropout": lstm_dropout_rate,
            "out_dropout": out_dropout_rate,
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

            # --- Early Stopping & Best Model Saving --- #
            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                last_saved_epoch = epoch
                current_patience = 0  # Reset patience
                print(
                    f"  New best model found! Dev PPL: {best_ppl:.2f}. Saving model.")
            else:
                if not asgd_switched:
                    current_patience += 1
                    print(
                        f"  No improvement in Dev PPL ({ppl_dev:.2f} vs best {best_ppl:.2f}) using SGD. Patience: {current_patience}/{max(patience, asgd_trigger_epochs)}")

                    # --- NT-ASGD Trigger --- #
                    if current_patience >= asgd_trigger_epochs:
                        print(
                            f"--- Switching to ASGD --- (Triggered after {asgd_trigger_epochs} epochs without improvement)")
                        optimizer = optim.ASGD(
                            model.parameters(), lr=current_lr, t0=0, lambd=0., weight_decay=weight_decay)
                        asgd_switched = True
                        current_patience = 0  # Reset patience after switching optimizer

                else:  # Already switched to ASGD
                    current_patience += 1
                    print(
                        f"  No improvement in Dev PPL ({ppl_dev:.2f} vs best {best_ppl:.2f}) using ASGD. Patience: {current_patience}/{patience}")

            if current_patience >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement.")
                break  # Exit the training loop

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
    # --- AWD-LSTM Medium PTB Hyperparameters (Approximation) --- #
    hid_size = 650
    emb_size = 650
    n_layers = 3
    lr = 30.0
    emb_dropout_rate = 0.4  # Input dropout (dropouti in paper)
    lstm_dropout_rate = 0.3  # Hidden layer dropout (dropouth in paper)
    out_dropout_rate = 0.4  # Output dropout (dropouto in paper)
    batch_size_train = 20
    batch_size_eval = 10
    epochs = 100  # Train longer, rely on ASGD + early stopping
    clip = 0.25  # Lower gradient clipping
    weight_decay = 1.2e-6  # L2 penalty
    patience = 10
    asgd_trigger_epochs = 5  # Epochs without improvement before switching to ASGD
    wandb_project = "NLU-project-part-1"
    wandb_group_prefix = "AWD-LSTM-Medium-Approx"

    print("Using random weight initialization U[-0.05, 0.05].")
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
        weight_decay=weight_decay,
        patience=patience,
        asgd_trigger_epochs=asgd_trigger_epochs,
        wandb_project=wandb_project,
        wandb_group_prefix=wandb_group_prefix
    )
