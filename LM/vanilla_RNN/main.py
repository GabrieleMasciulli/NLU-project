from functools import partial
import math
import torch.optim as optim
import torch.nn as nn
from utils import Lang, read_file
from functions import collate_fn, init_weights, train_loop, eval_loop, ensemble_eval_loop
from model import LM_RNN
from utils import DEVICE, PennTreeBank
import wandb
from tqdm import tqdm
import copy
import numpy as np
from torch.utils.data import DataLoader
import torch
import os
import random

# --- Seeding Function ---


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    wandb.login()

    # --- Configuration ---
    N_ENSEMBLE = 4
    BASE_SEED = 42
    MODEL_SAVE_DIR = "bin/ensemble_models"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Load the data (outside the loop)
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    # Create the vocabulary
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)
    pad_idx = lang.word2id["<pad>"]

    # Create the datasets
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # Dataloader instantiation
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(
        collate_fn, pad_token=pad_idx),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(
        collate_fn, pad_token=pad_idx))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(
        collate_fn, pad_token=pad_idx))

    # Define Fixed Hyperparameters
    hid_size = 256
    compression_dim = 100
    emb_size = 300
    lr = 1.0
    clip = 5
    n_epochs = 100
    base_patience = 3
    n_layers = 1

    # --- Loop for Training Ensemble Members ---
    trained_model_paths = []
    for i in range(N_ENSEMBLE):
        current_seed = BASE_SEED + i
        print(
            f"--- Training Ensemble Member {i+1}/{N_ENSEMBLE} (Seed: {current_seed}) ---")
        seed_everything(current_seed)  # Set the seed for this run

        model = LM_RNN(emb_size, hid_size, vocab_len, compression_dim,
                       pad_index=pad_idx, n_layers=n_layers).to(DEVICE)

        # Initialize the model weights

        model.apply(init_weights)

        # Define the optimizer
        optimizer = optim.SGD(model.parameters(), lr=lr)

        # Define loss functions
        criterion_train = nn.CrossEntropyLoss(ignore_index=pad_idx)
        criterion_eval = nn.CrossEntropyLoss(
            ignore_index=pad_idx, reduction='sum')
        best_ppl = math.inf
        best_model_state = None
        patience = base_patience

        # W&B Initialization
        group_name = "ensemble_training"
        run_name = f"ensemble_member_{i+1}_seed{current_seed}"

        run = wandb.init(
            project="rnn-hyperparam-tuning",
            group=group_name,
            name=run_name,
            config={
                "model_type": "RNN",
                "ensemble_member": i+1,
                "seed": current_seed,
                "n_layers": n_layers,
                "learning_rate": lr,
                "batch_size": 64,
                "hidden_size": hid_size,
                "embedding_size": emb_size,
                "compression_dim": compression_dim,
                "optimizer": "SGD",
                "clip": clip,
                "max_epochs": n_epochs,
                "patience": base_patience
            },
            reinit=True
        )

        # Training Loop
        pbar = tqdm(range(1, n_epochs + 1), desc=f"M{i+1} E{0} PPL: N/A")
        for epoch in pbar:
            model.train()
            loss = train_loop(train_loader, optimizer,
                              criterion_train, model, clip)

            # Evaluation
            model.eval()
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)

            # Log values to W&B
            run.log({
                "epoch": epoch,
                "train_loss": loss,
                "dev_loss": loss_dev,
                "dev_perplexity": ppl_dev
            })

            pbar.set_description(f"M{i+1} E{epoch} PPL: {ppl_dev:.2f}")

            # Check for improvement and update best model
            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model_state = copy.deepcopy(
                    model.state_dict())  # Save state_dict
                patience = base_patience
                print(f"Epoch {epoch}: New best PPL: {best_ppl:.2f}")
            else:
                patience -= 1

            if patience <= 0:
                print(f"Epoch {epoch}: Early stopping triggered.")
                break

        # Save the best model state for this run
        print(f"--- Training Finished: Member {i+1} ---")
        print(f"Best Dev PPL: {best_ppl:.2f}")
        run.summary["best_dev_perplexity"] = best_ppl

        if best_model_state is not None:
            model_save_path = os.path.join(
                MODEL_SAVE_DIR, f"best_model_member_{i+1}_seed{current_seed}.pt")
            torch.save(best_model_state, model_save_path)
            trained_model_paths.append(model_save_path)
            print(
                f"Saved best model state for member {i+1} to {model_save_path}")
        else:
            print(
                f'No best model found for member {i+1} - training did not improve.')

        # Finish W&B run for this member
        wandb.finish()

    print(f"--- Finished Training All {N_ENSEMBLE} Ensemble Members ---")

    # --- Ensemble Evaluation ---
    if len(trained_model_paths) == N_ENSEMBLE:
        print("\n--- Starting Ensemble Evaluation ---")
        ensemble_models = []
        print("Loading trained ensemble members...")

        for i, path in enumerate(trained_model_paths):
            print(f"Loading model {i+1}: {path}")
            # instantiating the model with the same parameters used during training
            model = LM_RNN(emb_size, hid_size, vocab_len, compression_dim,
                           pad_index=pad_idx, n_layers=n_layers).to(DEVICE)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.eval()
            ensemble_models.append(model)

        # Need the ensemble eval criterion (sum reduction is important for correct PPL calculation)
        criterion_eval_ensemble = nn.CrossEntropyLoss(
            ignore_index=pad_idx, reduction='sum')

        print("Calculating ensemble perplexity on test set...")
        final_ensemble_ppl, _ = ensemble_eval_loop(
            test_loader, criterion_eval_ensemble, ensemble_models)
        print(f'\n>>> Final Ensemble Test PPL: {final_ensemble_ppl:.2f} <<<')

        # Log ensemble result (optional, creates a new W&B run)
        print("Logging final ensemble result to W&B...")
        wandb.init(project="rnn-ensemble-ptb",
                   name="ensemble_final_eval", reinit=True)
        wandb.config.update({
            "n_ensemble_members": N_ENSEMBLE,
            "model_type": "LSTM" if isinstance(ensemble_models[0].rnn, nn.LSTM) else "RNN",
            "n_layers": n_layers,
            "hidden_size": hid_size,
            "embedding_size": emb_size,
            "compression_dim": compression_dim,
        })
        wandb.summary["final_ensemble_test_perplexity"] = final_ensemble_ppl
        wandb.finish()
    else:
        print("Ensemble evaluation skipped: Not all members trained successfully.")
