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
from torch.utils.data import DataLoader
import torch
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau


def main(hid_size, emb_size, n_layers, lr,
         batch_size_train, batch_size_eval, epochs, clip,
         wandb_project, wandb_group_prefix,
         emb_dropout_rate=0.5, lstm_dropout_rate=0.5, out_dropout_rate=0.5,
         patience=3):
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

    init_weights(model)

    # --- Optimizer and Loss --- #
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_index)
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=pad_index, reduction='sum')

    # --- Learning Rate Scheduler --- #
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr=1e-4)

    # --- Training Setup --- #
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    lr_decay_factor = 0.5
    lr_decay_epoch_threshold = 6

    pbar = tqdm(range(1, epochs + 1))

    # --- Early Stopping Setup --- #
    patience_counter = 0

    # --- W&B Initialization --- #
    run_name_parts = [
        f"lstm",
        f"l{n_layers}",
        f"h{hid_size}",
        f"emb{emb_size}",
        f"drop{emb_dropout_rate}",
        f"lr{lr}",
    ]
    run_name = "_".join(run_name_parts)
    group_name = f"{wandb_group_prefix}"

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
            "clip_gradient": clip,
            "n_layers": n_layers,
            "lr_decay_factor": lr_decay_factor,
            "lr_decay_epoch_threshold": lr_decay_epoch_threshold,
            "emb_dropout_rate": emb_dropout_rate,
            "lstm_dropout_rate": lstm_dropout_rate,
            "out_dropout_rate": out_dropout_rate
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

            # --- LR Schedule & Best Model Saving --- #
            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                last_saved_epoch = epoch
                patience_counter = 0
                print(
                    f"  New best model found! Dev PPL: {best_ppl:.2f}. Saving model.")
            else:
                patience_counter += 1
                print(
                    f"  No improvement in Dev PPL ({ppl_dev:.2f} vs best {best_ppl:.2f}). Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(
                        f"  Early stopping triggered after {patience} epochs without improvement.")
                    break

            # --- LR Scheduler Step --- #
            scheduler.step(ppl_dev)

            # --- Early Stopping --- #
            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                last_saved_epoch = epoch
                patience_counter = 0
                print(
                    f"  New best model found! Dev PPL: {best_ppl:.2f}. Saving model.")
            else:
                patience_counter += 1
                print(
                    f"  No improvement in Dev PPL ({ppl_dev:.2f} vs best {best_ppl:.2f}). Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(
                        f"  Early stopping triggered after {patience} epochs without improvement.")
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    # --- Final Evaluation on Test Set --- #
    print("\nTraining finished.")
    if best_model is not None:
        print(
            f"Evaluating best model (from epoch {last_saved_epoch}, PPL: {best_ppl:.2f}) on test set...")
        best_model.to(device=DEVICE)
        best_model.eval()
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        print(f'Final Test Perplexity: {final_ppl:.2f}')
        print(f'Dev PPL: {best_ppl:.2f}')

        os.makedirs('bin', exist_ok=True)
        model_save_path = f'bin/best_model_{run_name}.pt'
        torch.save(best_model.state_dict(), model_save_path)
        print(f"Best model saved to {model_save_path}")

        run.log({"test_perplexity": final_ppl})
        wandb.finish()
        print("Run finished.")
        return final_ppl
    else:
        print('No best model found - training might have diverged or stopped very early.')
        wandb.finish()
        print("Run finished.")
        return float('inf')  # <-- Return inf if no model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LSTM Language Model")
    parser.add_argument("--hid_size", type=int, default=650)
    parser.add_argument("--emb_size", type=int, default=650)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--batch_size_train", type=int, default=32)
    parser.add_argument("--batch_size_eval", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--wandb_project", type=str,
                        default="NLU-project-part1A")
    parser.add_argument("--wandb_group_prefix", type=str,
                        default="zaremba-medium")
    parser.add_argument("--emb_dropout_rate", type=float, default=0.5)
    parser.add_argument("--lstm_dropout_rate", type=float, default=0.5)
    parser.add_argument("--out_dropout_rate", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=3)

    args = parser.parse_args()

    try:
        wandb.login()
    except Exception as e:
        print(f"Could not login to WandB: {e}. Proceeding without logging.")

    main(
        hid_size=args.hid_size,
        emb_size=args.emb_size,
        n_layers=args.n_layers,
        lr=args.lr,
        batch_size_train=args.batch_size_train,
        batch_size_eval=args.batch_size_eval,
        epochs=args.epochs,
        clip=args.clip,
        wandb_project=args.wandb_project,
        wandb_group_prefix=args.wandb_group_prefix,
        emb_dropout_rate=args.emb_dropout_rate,
        lstm_dropout_rate=args.lstm_dropout_rate,
        out_dropout_rate=args.out_dropout_rate,
        patience=args.patience
    )
