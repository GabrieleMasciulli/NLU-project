from functools import partial
import math
import torch.nn as nn
from utils import Lang, read_file
from functions import collate_fn, init_weights, train_loop, eval_loop
from model import LM_LSTM
from utils import DEVICE, PennTreeBank, load_glove_embeddings, download_and_extract_glove
import torch.optim as optim
import wandb
from tqdm import tqdm
import copy
import os
from torch.utils.data import DataLoader
import torch
from nt_asgd import NTAvSGD


def main(hid_size, emb_size, n_layers, lr, emb_dropout_rate, out_dropout_rate,
         batch_size_train, batch_size_eval, epochs, clip, weight_decay, patience, nonmono,
         wandb_project, wandb_group_prefix):
    """Main function to train and evaluate the LSTM Language Model."""

    # --- Data Loading and Preprocessing ---
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    lang = Lang(train_raw, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)
    pad_index = lang.word2id["<pad>"]

    # --- GloVe Embedding Loading ---
    glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"
    glove_dir = "glove"
    glove_filename = f"glove.6B.{emb_size}d.txt"
    glove_txt_path = download_and_extract_glove(glove_url, glove_dir, glove_filename)
    if glove_txt_path is not None:
        glove_embeddings = load_glove_embeddings(
            glove_txt_path, lang.word2id, emb_size)
    else:
        print(
            "GloVe embeddings could not be loaded. Proceeding with random initialization.")
        glove_embeddings = None

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
                    out_dropout_rate=out_dropout_rate,
                    pretrained_embeddings=glove_embeddings
                    ).to(DEVICE)

    # Apply Zaremba weight initialization
    init_weights(model)

    # --- Optimizer and Loss --- #
    # (Initial optimizer is SGD)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    asgd_optimizer_params = {'lr': lr, 't0': 0, 'weight_decay': weight_decay}
    asgd_triggered = False

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

    pbar = tqdm(range(1, epochs + 1))

    # --- W&B Initialization --- #
    run_name_parts = [
        f"lstm",
        f"l{n_layers}",
        f"h{hid_size}",
        f"emb_dropout{emb_dropout_rate}",
        f"out_dropout{out_dropout_rate}",
        "SGD_then_NTAvSGD",
        "VarDrop",
        "GloVe" if glove_embeddings is not None else ""
    ]
    run_name = "_".join(run_name_parts)
    # Updated group name
    group_name = f"{wandb_group_prefix}_NTASGD"

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
            "optimizer_initial": "SGD",
            "optimizer_final_type": "NT_ASGD",
            "nonmono_trigger": nonmono,
            "epochs": epochs,
            "weight_decay": weight_decay,
            "clip_gradient": clip,
            "patience": patience,
            "n_layers": n_layers,
            "emb_dropout": emb_dropout_rate,
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

            # --- Evaluate on development set ---
            model.eval()

            original_params_dev = None  # To store original params when swapped for dev eval
            if asgd_triggered and optimizer.is_averaging():  # Check if ASGD is active and averaging
                print(
                    f"  Epoch {epoch}: Swapping to averaged weights for dev evaluation.")
                original_params_dev = optimizer.swap_parameters(model)

            ppl_dev, epoch_dev_loss = eval_loop(
                dev_loader, criterion_eval, model)

            if original_params_dev is not None:  # Swap back if params were changed for dev eval
                print(
                    f"  Epoch {epoch}: Swapping back to original weights after dev evaluation.")
                optimizer.load_original_params(original_params_dev, model)
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

            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                # Saving logic modified to handle averaged weights
                if asgd_triggered and optimizer.is_averaging():
                    print(
                        f"  Epoch {epoch}: New best model found with averaged weights. Dev PPL: {best_ppl:.2f}.")
                    original_params_for_saving = optimizer.swap_parameters(
                        model)
                    best_model_state = copy.deepcopy(
                        model.state_dict())  # Save averaged weights
                    # Swap back for continued training
                    optimizer.load_original_params(
                        original_params_for_saving, model)
                else:
                    print(
                        f"  Epoch {epoch}: New best model found with SGD weights. Dev PPL: {best_ppl:.2f}.")
                    best_model_state = copy.deepcopy(
                        model.state_dict())  # Save current SGD weights

                last_saved_epoch = epoch
                current_patience = 0  # Reset patience on improvement
                # General print moved after specific saving logic
                print(
                    f"  Saving model state (Epoch {epoch}).")

            else:
                # No improvement
                current_patience += 1
                print(
                    f"  No improvement in Dev PPL ({ppl_dev:.2f} vs best {best_ppl:.2f}). Patience: {current_patience}/{patience}")

                if not asgd_triggered and current_patience == nonmono:
                    for param_group in optimizer.param_groups:
                        old_lr = param_group['lr']
                        param_group['lr'] *= 0.5
                        new_lr = param_group['lr']
                        print(
                            f"  Halving learning rate from {old_lr:.4f} to {new_lr:.4f} as nonmono patience ({current_patience}/{nonmono}) reached before ASGD switch.")

            if not asgd_triggered and current_patience >= nonmono:
                print(
                    f"Switching to NT-ASGD optimizer at epoch {epoch} after {nonmono} epochs of no improvement on dev PPL.")
                current_optimizer_lr = optimizer.param_groups[0]['lr']
                asgd_optimizer_params['lr'] = current_optimizer_lr
                optimizer = NTAvSGD(model.parameters(), **
                                    asgd_optimizer_params)
                optimizer.start_averaging()  # Explicitly start the averaging process
                asgd_triggered = True
                print(
                    f"  Optimizer switched to NTAvSGD with LR: {current_optimizer_lr:.4f} and averaging started.")

                run.log({"optimizer_switched_to_ASGD_epoch": epoch,
                        "asgd_start_lr": current_optimizer_lr})
                run.config.update(
                    {"optimizer_active": "NT_ASGD"}, allow_val_change=True)

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
    n_layers = 3
    lr = 30.0
    hid_size = 1150
    emb_size = 300
    emb_dropout_rate = 0.4
    out_dropout_rate = 0.4
    batch_size_train = 64
    batch_size_eval = 128
    epochs = 100
    clip = 0.25
    weight_decay = 1.2e-6
    patience = 10
    nonmono = 5
    wandb_project = "NLU-project-part-1B"
    wandb_group_prefix = "LSTM-weight-tying-VarDrop-NT-AvSGD-"

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
        nonmono=nonmono,
        wandb_project=wandb_project,
        wandb_group_prefix=wandb_group_prefix
    )
