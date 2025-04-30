from functools import partial
import math
import torch.nn as nn
from utils import Lang, read_file
from functions import collate_fn, init_weights, train_loop, eval_loop
from model import LM_LSTM
from utils import DEVICE, PennTreeBank
from nt_asgd import NTAvSGD
import wandb
from tqdm import tqdm
import copy
import os
from torch.utils.data import DataLoader
import torch


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
    model = LM_LSTM(emb_size, hid_size, vocab_len,
                    pad_index=pad_index,
                    n_layers=n_layers,
                    emb_dropout_rate=emb_dropout_rate,
                    lstm_dropout_rate=lstm_dropout_rate,
                    out_dropout_rate=out_dropout_rate).to(DEVICE)

    # Apply Zaremba weight initialization
    init_weights(model)

    # --- Optimizer and Loss --- #
    optimizer = NTAvSGD(model.parameters(), lr=lr,
                        weight_decay=weight_decay, n=asgd_trigger_epochs)
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_index)
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=pad_index, reduction='sum')

    # --- Training Setup --- #
    losses_train = []
    losses_dev = []
    dev_ppl_logs = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model_state = None
    current_patience = 0

    pbar = tqdm(range(1, epochs + 1))

    # --- W&B Initialization --- #
    run_name_parts = [
        f"lstm",
        f"l{n_layers}",
        f"h{hid_size}",
        f"emb_dropout{emb_dropout_rate}",
        f"out_dropout{out_dropout_rate}",
        "NT-AvSDG"
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
        t = 0  # Counter for validation checks
        for epoch in pbar:
            # Train one epoch
            model.train()

            epoch_train_loss = train_loop(
                train_loader, optimizer, criterion_train, model, clip)

            # --- Evaluate on development set ---
            model.eval()
            original_params = None # Initialize original_params to None
            if optimizer.is_averaging():
                original_params = optimizer.swap_parameters()  # Swap to averaged parameters and store original ones

            # Run evaluation
            ppl_dev, epoch_dev_loss = eval_loop(
                dev_loader, criterion_eval, model)

            if optimizer.is_averaging() and original_params is not None:
                optimizer.load_original_params(original_params)  # Swap back using the stored original parameters
            # --- End Evaluation --- #

            # Process and log metrics
            avg_train_loss = epoch_train_loss.item() if isinstance(
                epoch_train_loss, torch.Tensor) else epoch_train_loss
            avg_dev_loss = epoch_dev_loss.item() if isinstance(
                epoch_dev_loss, torch.Tensor) else epoch_dev_loss
            losses_train.append(avg_train_loss)
            losses_dev.append(avg_dev_loss)
            sampled_epochs.append(epoch)

            # Log perplexity for NT-ASGD check
            dev_ppl_logs.append(ppl_dev)
            t += 1  # Increment validation check counter

            # Get current LR for logging
            current_lr = optimizer.param_groups[0]['lr']

            pbar.set_description(
                f"Epoch {epoch} | LR: {current_lr:.4f} | Train Loss: {avg_train_loss:.4f} | Dev PPL: {ppl_dev:.2f} | Avg: {optimizer.is_averaging()}")

            run.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "dev_loss": avg_dev_loss,
                "dev_perplexity": ppl_dev,
                "learning_rate": current_lr,
                "is_averaging": optimizer.is_averaging()
            })

            # --- Early Stopping & Best Model Saving --- #
            if ppl_dev < best_ppl:
                best_ppl = ppl_dev

                # 1. Swap the *original* model to its averaged parameters
                original_params_for_saving = optimizer.swap_parameters()

                # Check if the swap actually happened before saving
                if original_params_for_saving:
                    # 2. Get the state dict of the now-averaged original model
                    best_model_state = copy.deepcopy(model.state_dict())
                    # 3. Swap the original model back to its non-averaged state for continued training
                    optimizer.load_original_params(original_params_for_saving)
                    print(f"  New best model found! Dev PPL: {best_ppl:.2f}. Saving averaged model state (Epoch {epoch}).")
                else:
                    # This case should ideally not happen if averaging is active,
                    print(f"  Improved Dev PPL ({best_ppl:.2f}), but failed to swap to averaged parameters for saving. Saving current non-averaged state instead (Epoch {epoch}).")
                    # Save the current (non-averaged) state as a fallback
                    best_model_state = copy.deepcopy(model.state_dict())

                last_saved_epoch = epoch
                current_patience = 0  # Reset patience only if we improve

            else:
                # No improvement
                # Since averaging started immediately, we just increment patience
                current_patience += 1
                print(
                    f"  No improvement in Dev PPL ({ppl_dev:.2f} vs best {best_ppl:.2f}) using NTAvSGD (Averaging). Patience: {current_patience}/{patience}")

            # Check for final patience-based early stopping
            if current_patience >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement during averaging phase.")
                break  # Exit the training loop

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # --- Final Evaluation on Test Set --- #
    print("\nTraining finished.")
    if best_model_state is not None:
        print(
            f"Loading best model state (from epoch {last_saved_epoch}, Dev PPL: {best_ppl:.2f}) for final evaluation...")
        # Reload the best state into the model structure
        final_model = LM_LSTM(emb_size, hid_size, vocab_len,
                              pad_index=pad_index, n_layers=n_layers,
                              emb_dropout_rate=emb_dropout_rate,
                              lstm_dropout_rate=lstm_dropout_rate,
                              out_dropout_rate=out_dropout_rate)
        final_model.load_state_dict(best_model_state)
        final_model.to(DEVICE)
        final_model.eval()

        print("Evaluating on test set...")
        # The best_model_state already contains the averaged weights if averaging was active when saved.
        final_ppl, _ = eval_loop(test_loader, criterion_eval, final_model)
        print(f'Final Test Perplexity: {final_ppl:.2f}')

        # Save the best model state dict
        os.makedirs('bin', exist_ok=True)  # Ensure the directory exists
        model_save_path = f'bin/best_model_{run_name}.pt'
        torch.save(best_model_state, model_save_path)
        print(f"Best model saved to {model_save_path}")

        run.log({"test_perplexity": final_ppl})
    else:
        print('No best model state found.')

    wandb.finish()
    print("Run finished.")


if __name__ == "__main__":
    # --- Hyperparameters --- #
    hid_size = 650
    emb_size = 650
    n_layers = 3
    lr = 30.0
    emb_dropout_rate = 0.4  # Input dropout (dropouti in paper)
    lstm_dropout_rate = 0.3  # Hidden layer dropout (dropouth in paper)
    out_dropout_rate = 0.4  # Output dropout (dropouto in paper)
    batch_size_train = 20
    batch_size_eval = 10
    epochs = 100  # Train longer, rely on NT-AvSGD + early stopping
    clip = 0.25
    weight_decay = 1.2e-6  # L2 penalty
    patience = 10  # Patience *after* ASGD switch (or if ASGD never triggers)
    asgd_trigger_epochs = 2  # n in paper
    wandb_project = "NLU-project-part-1"
    wandb_group_prefix = "NT-ASGD-AWD-LSTM-Medium-Approx"

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
