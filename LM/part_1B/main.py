from functools import partial
import math
import torch.nn as nn
from utils import Lang, read_file
from functions import collate_fn, init_weights, train_loop, eval_loop
from model import LM_LSTM
from utils import DEVICE, PennTreeBank
import torch.optim as optim
import wandb
from tqdm import tqdm
import copy
import os
from torch.utils.data import DataLoader
import torch


def main(hid_size, emb_size, n_layers, lr, emb_dropout_rate, out_dropout_rate,
         batch_size_train, batch_size_eval, epochs, clip, weight_decay, patience, wandb_project, wandb_group_prefix):
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
                    out_dropout_rate=out_dropout_rate).to(DEVICE)

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
    best_model_state = None
    current_patience = 0

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=patience // 2, verbose=True)

    pbar = tqdm(range(1, epochs + 1))

    # --- W&B Initialization --- #
    run_name_parts = [
        f"lstm",
        f"l{n_layers}",
        f"h{hid_size}",
        f"emb_dropout{emb_dropout_rate}",
        f"out_dropout{out_dropout_rate}",
        "SGD",
        "VarDrop"
    ]
    run_name = "_".join(run_name_parts)
    group_name = f"{wandb_group_prefix}_VarDrop"

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
            "emb_dropout": emb_dropout_rate,
            "out_dropout": out_dropout_rate,
            "lr_scheduler": type(lr_scheduler).__name__
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

            # --- Evaluate on development set ---
            model.eval()
            ppl_dev, epoch_dev_loss = eval_loop(
                dev_loader, criterion_eval, model)
            # --- End Evaluation --- #

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

            lr_scheduler.step(avg_dev_loss)

            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model_state = copy.deepcopy(model.state_dict())
                last_saved_epoch = epoch
                current_patience = 0  # Reset patience on improvement
                print(
                    f"  New best model found! Dev PPL: {best_ppl:.2f}. Saving model state (Epoch {epoch}).")

            else:
                # No improvement
                current_patience += 1
                print(
                    f"  No improvement in Dev PPL ({ppl_dev:.2f} vs best {best_ppl:.2f}). Patience: {current_patience}/{patience}")

            # Check for early stopping
            if current_patience >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement.")
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
                              out_dropout_rate=out_dropout_rate)
        final_model.load_state_dict(best_model_state)
        final_model.to(DEVICE)
        final_model.eval()

        print("Evaluating on test set...")
        # Evaluate the loaded best model state
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
    lr = 10.0
    emb_dropout_rate = 0.4
    out_dropout_rate = 0.4
    batch_size_train = 64
    batch_size_eval = 128
    epochs = 100
    clip = 0.25
    weight_decay = 1.2e-6
    patience = 10
    wandb_project = "NLU-project-part-1B"
    wandb_group_prefix = "SGD-LSTM-weight-tying-VarDrop"

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
        out_dropout_rate=out_dropout_rate,
        batch_size_train=batch_size_train,
        batch_size_eval=batch_size_eval,
        epochs=epochs,
        clip=clip,
        weight_decay=weight_decay,
        patience=patience,
        wandb_project=wandb_project,
        wandb_group_prefix=wandb_group_prefix
    )
