import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
import wandb
import os
import copy
from collections import Counter
from utils import DEVICE, IntentsAndSlots, Lang, load_data, collate_fn, tokenizer, BERT_MODEL_NAME
from model import CTRAN
from functions import train_loop, eval_loop


def main(
    # --- BERT/Model Hyperparameters ---
    bert_model_name: str = BERT_MODEL_NAME,
    dropout_prob: float = 0.1,

    # --- Training Hyperparameters ---
    lr: float = 5e-5,
    n_epochs: int = 10,
    patience: int = 3,
    warmup_steps: int = 0,
    batch_size_train: int = 16,
    batch_size_eval: int = 32,

    # --- W&B Config ---
    wandb_project: str = "NLU-project-part-2B",
    wandb_group_prefix: str = "joint-bert",

    # --- Data Paths ---
    data_dir: str = os.path.join("dataset", "ATIS")
):
    """
    Main function to train and evaluate the Joint BERT model.
    """
    print(f"Using device: {DEVICE}")
    print(f"Using BERT model: {bert_model_name}")

    # --- Load Data ---
    train_path = os.path.join(data_dir, "train.json")
    test_path = os.path.join(data_dir, "test.json")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(
            f"Error: Data files not found in {data_dir}. Please ensure train.json and test.json exist.")
        return

    tmp_train_raw = load_data(train_path)
    test_raw = load_data(test_path)

    # --- Split train into train and dev ---
    portion = 0.10
    intents_full = [sample['intent'] for sample in tmp_train_raw]
    count_y = Counter(intents_full)
    labels_stratify = []
    inputs_stratify = []
    mini_train = []

    for i, intent in enumerate(intents_full):
        # Ensure stratification is possible only for intents with >1 sample
        if count_y[intent] > 1:
            inputs_stratify.append(tmp_train_raw[i])
            labels_stratify.append(intent)
        else:
            # Keep single-sample intents in training
            mini_train.append(tmp_train_raw[i])

    # Stratified split only on samples where stratification is possible
    if len(inputs_stratify) > 0 and len(set(labels_stratify)) > 1:
        X_train_strat, X_dev_strat, _, _ = train_test_split(
            inputs_stratify, labels_stratify, test_size=portion, stratify=labels_stratify, shuffle=True, random_state=42
        )
        # Combine stratified part with single samples
        train_raw = X_train_strat + mini_train
        dev_raw = X_dev_strat
    else:  # Handle cases with insufficient data for stratification
        print("Warning: Not enough data or distinct labels for stratified split. Using random split or direct assignment.")
        # Fallback: simple split or assign all to train/dev if needed
        split_idx = int(len(tmp_train_raw) * (1 - portion))
        train_raw = tmp_train_raw[:split_idx]
        dev_raw = tmp_train_raw[split_idx:]

    print(f"Train set size: {len(train_raw)}")
    print(f"Dev set size: {len(dev_raw)}")
    print(f"Test set size: {len(test_raw)}")

    # --- Create Lang object (Simpler for BERT) ---
    corpus = train_raw + dev_raw + test_raw
    slots_unique = set(sum([line['slots'].split() for line in corpus], []))
    intents_unique = set([line['intent'] for line in corpus])
    # No need for words list or cutoff with BERT tokenizer
    lang = Lang(intents_unique, slots_unique)
    print(f"Intents: {len(lang.intent2id)}, Slots: {len(lang.slot2id)}")

    # --- Create Dataset objects (Pass tokenizer) ---
    # Tokenizer is already initialized in utils.py
    train_dataset = IntentsAndSlots(train_raw, lang, tokenizer)
    dev_dataset = IntentsAndSlots(dev_raw, lang, tokenizer)
    test_dataset = IntentsAndSlots(test_raw, lang, tokenizer)

    # --- Create DataLoader objects (Use new collate_fn) ---
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size_eval, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size_eval, shuffle=False, collate_fn=collate_fn)

    # --- Setting up BERT model, optimizer and scheduler ---
    num_intent_labels = len(lang.intent2id)
    num_slot_labels = len(lang.slot2id)

    # Can set num_labels to intent or slot
    config = BertConfig.from_pretrained(
        bert_model_name, num_labels=num_intent_labels)
    model = CTRAN.from_pretrained(
        bert_model_name,
        config=config,
        num_intent_labels=num_intent_labels,
        num_slot_labels=num_slot_labels,
        dropout_prob=dropout_prob
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * n_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # --- W&B Initialization ---
    run_name = f"{wandb_group_prefix}_lr{lr}_bs{batch_size_train}_do{dropout_prob}"
    run = wandb.init(
        project=wandb_project,
        group=wandb_group_prefix,
        name=run_name,
        config={
            "bert_model": bert_model_name,
            "learning_rate": lr,
            "batch_size_train": batch_size_train,
            "batch_size_eval": batch_size_eval,
            "dropout_prob": dropout_prob,
            "optimizer": "AdamW",
            "scheduler": "LinearWarmup",
            "epochs": n_epochs,
            "patience": patience,
            "warmup_steps": warmup_steps,
            "num_intent_labels": num_intent_labels,
            "num_slot_labels": num_slot_labels,
        }
    )
    print(f"--- Starting Training: {run_name} ---")

    # --- Training Loop ---
    best_dev_metric = -1.0
    epochs_no_improve = 0
    best_model_state = None
    last_saved_epoch = -1

    try:
        for epoch in range(1, n_epochs + 1):
            print(f"\n--- Epoch {epoch}/{n_epochs} ---")

            # Training
            avg_train_loss = train_loop(
                model, train_loader, optimizer, scheduler)

            # Evaluation
            dev_metrics = eval_loop(model, dev_loader, lang)
            dev_intent_acc = dev_metrics["intent_acc"]
            dev_slot_f1 = dev_metrics["slot_f1_macro"]
            # Example combined metric:
            current_dev_metric = (dev_intent_acc + dev_slot_f1) / 2

            print(f"Epoch {epoch} Summary:")
            print(f"  Avg Train Loss: {avg_train_loss:.4f}")
            print(f"  Dev Intent Acc: {dev_intent_acc:.4f}")
            print(f"  Dev Slot F1 (Macro): {dev_slot_f1:.4f}")
            print(f"  Dev Combined Metric: {current_dev_metric:.4f}")

            # Log metrics to W&B
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "dev_intent_accuracy": dev_intent_acc,
                "dev_slot_f1_macro": dev_slot_f1,
                "dev_slot_f1_micro": dev_metrics["slot_f1_micro"],
                "dev_combined_metric": current_dev_metric,
                "learning_rate": scheduler.get_last_lr()[0]
            })

            # Early stopping and best model saving
            if current_dev_metric > best_dev_metric:
                best_dev_metric = current_dev_metric
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
                last_saved_epoch = epoch
                print(
                    f"  New best model found! Metric: {best_dev_metric:.4f}. Saving...")
            else:
                epochs_no_improve += 1
                print(
                    f"  No improvement. Patience: {epochs_no_improve}/{patience}")

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # --- Final Test Evaluation ---
        if best_model_state:
            print("\n--- Evaluating best model on Test Set ---")
            print(
                f"Loading best model from epoch {last_saved_epoch} (Dev Metric: {best_dev_metric:.4f})")
            model.load_state_dict(best_model_state)

            test_metrics = eval_loop(
                model, test_loader, lang, is_test=True)  # Pass is_test=True

            print("\nTest Set Performance:")
            print(f"  Intent Accuracy: {test_metrics['intent_acc']:.4f}")
            print(f"  Slot F1 (Macro): {test_metrics['slot_f1_macro']:.4f}")
            print(f"  Slot F1 (Micro): {test_metrics['slot_f1_micro']:.4f}")

            # Log final test metrics to W&B summary
            run.summary["best_dev_metric"] = best_dev_metric
            run.summary["best_dev_epoch"] = last_saved_epoch
            run.summary["test_intent_accuracy"] = test_metrics['intent_acc']
            run.summary["test_slot_f1_macro"] = test_metrics['slot_f1_macro']
            run.summary["test_slot_f1_micro"] = test_metrics['slot_f1_micro']

            model_save_path = f'bin/best_model_{run_name}.pt'
            os.makedirs('bin', exist_ok=True)
            torch.save(best_model_state, model_save_path)
            print(f"Best model state saved to {model_save_path}")

        else:
            print("No best model was saved during training.")

        # Finish W&B run
        if run:
            run.finish()
        print("--- Run Finished ---")


if __name__ == "__main__":
    main(
        bert_model_name=BERT_MODEL_NAME,
        dropout_prob=0.15,
        lr=5e-5,
        n_epochs=15,
        patience=3,
        warmup_steps=0,
        batch_size_train=32,
        batch_size_eval=64,
        wandb_project="NLU-project-part-2B",
        wandb_group_prefix="joint-bert-atis-CTRAN",
        data_dir=os.path.join("dataset", "ATIS")
    )
