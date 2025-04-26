from collections import Counter
import copy
import os
from torch import optim
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm 
import wandb
from utils import DEVICE, PAD_TOKEN, IntentsAndSlots, Lang, load_data
from functions import collate_fn, eval_loop, init_weights, train_loop
from model import ModelIAS
import numpy as np
from sklearn.model_selection import train_test_split


def main(
    hid_size: int,
    emb_size: int,
    lr: float,
    clip: float,
    n_epochs: int,
    patience: int,
    batch_size_train: int,
    batch_size_eval: int,
    wandb_project: str,
    wandb_group_prefix: str
):
    tmp_train_raw = load_data(os.path.join("dataset", "ATIS", "train.json"))
    test_raw = load_data(os.path.join("dataset", "ATIS", "test.json"))

    # --- Split train into train and dev doing stratified sampling over intents --- #
    portion = 0.10

    intents = [sample['intent'] for sample in tmp_train_raw]
    count_y = Counter(intents)

    labels = []  # dev set labels
    inputs = []  # dev set inputs
    mini_train = []  # train set inputs

    for id_y, y in enumerate(intents):
        if count_y[y] > 1:  # if the intent is present more than once, we add it to the dev set
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:  # otherwise, we add it to the train set
            mini_train.append(tmp_train_raw[id_y])

    # random stratified sampling
    X_train, X_dev, y_train, y_dev = train_test_split(
        inputs, labels, test_size=portion, stratify=labels, shuffle=True, random_state=42)

    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    print(f"Train set size: {len(train_raw)}")
    print(f"Dev set size: {len(dev_raw)}")
    print(f"Test set size: {len(test_raw)}")

    # --- Create Lang object --- #
    words = sum([sample['utterance'].split() for sample in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])
    lang = Lang(words, intents, slots, cutoff=0)

    # --- Create Dataset objects --- #
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # --- Create DataLoader objects --- #
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size_eval, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size_eval, shuffle=False, collate_fn=collate_fn)

    # --- Setting up model, optimizer and loss functions --- #

    out_slots = len(lang.slot2id)
    out_intents = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    model = ModelIAS(
        hid_size=hid_size,
        emb_size=emb_size,
        vocab_len=vocab_len,
        out_slot=out_slots,
        out_int=out_intents,
        pad_index=PAD_TOKEN
    ).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    # --- Training --- #
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_model = None

    pbar = tqdm(range(1, n_epochs))

    # --- W&B Initialization --- #
    run_name_parts = [
        f"lstm",
        f"l{1}",
        f"h{hid_size}",
        f"emb{emb_size}",
    ]
    run_name = "_".join(run_name_parts)
    group_name = wandb_group_prefix

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
            "epochs": n_epochs,
            "clip_gradient": clip,
            "patience": patience,
            "n_layers": 1,
        }
    )

    # --- Training Loop --- #
    print(f"Starting training for run: {run_name}")
    try:
        for epoch in pbar:
            loss = train_loop(train_loader, optimizer,
                              criterion_slots, criterion_intents, model, clip)

            # Log training loss every epoch
            run.log({
                "epoch": epoch,
                "train_loss": np.asarray(loss).mean()
            })

            if epoch % 5 == 0:  # checking the performance of the model every 5 epochs
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(
                    dev_loader, criterion_slots, criterion_intents, model, lang)
                losses_dev.append(np.asarray(loss_dev).mean())

                run.log({
                    "dev_loss": losses_dev[-1],
                    "dev_slot_f1": results_dev['total']['f'],
                    "dev_intent_accuracy": intent_res['accuracy']
                })
                print(
                    f"Epoch {epoch} - Train loss: {losses_train[-1]} - Dev loss: {losses_dev[-1]}")

                f1 = results_dev['total']['f']
                # @todo: for decreasing the patience, we could use the average btw slot f1 and intent accuracy
                if f1 > best_f1:
                    best_f1 = f1
                    patience = 3
                    best_model = copy.deepcopy(model).to('cpu')
                    print(
                        f"  New best model found! F1: {best_f1:.2f}. Saving model.")
                else:
                    patience -= 1
                if patience <= 0:
                    print("Early stopping.")
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    # --- Final Evaluation on Test Set --- #
    print("\nTraining finished.")
    if best_model is not None:
        results_test, intent_test, _ = eval_loop(
            test_loader, criterion_slots, criterion_intents, best_model, lang)
        print('Slot F1: ', results_test['total']['f'])
        print('Intent Accuracy:', intent_test['accuracy'])

        # Log the results to W&B
        wandb.log({
            "test_slot_f1": results_test['total']['f'],
            "test_intent_accuracy": intent_test['accuracy']
        })
    else:
        print('No best model found - training might have diverged or stopped very early.')

    wandb.finish()
    print("Run finished.")


if __name__ == "__main__":
    hid_size = 200
    emb_size = 300
    lr = 0.0001
    clip = 5.0
    n_epochs = 200
    patience = 3
    batch_size_train = 128
    batch_size_eval = 64
    wandb_project_name = "NLU_project_part_2A"
    wandb_group_prefix = "before_changes"

    # --- Login to W&B --- #
    try:
        wandb.login()
    except Exception as e:
        print(f"Could not login to WandB: {e}. Proceeding without logging.")

    # --- Starts training --- #
    main(
        hid_size=hid_size,
        emb_size=emb_size,
        lr=lr,
        clip=clip,
        n_epochs=n_epochs,
        patience=patience,
        batch_size_train=batch_size_train,
        batch_size_eval=batch_size_eval,
        wandb_project=wandb_project_name,
        wandb_group_prefix=wandb_group_prefix
    )
