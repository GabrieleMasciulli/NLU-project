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

if __name__ == "__main__":
    wandb.login()

    # Load the data
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    # Create the vocabulary
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    # Create the datasets
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # Dataloader instantiation
    # You can reduce the batch_size if the GPU memory is not enough
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(
        collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(
        collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(
        collate_fn, pad_token=lang.word2id["<pad>"]))

    # Define the model parameters
    hid_size = 256
    emb_size = 512
    lr = 0.1

    # Get the vocabulary length
    vocab_len = len(lang.word2id)

    # Instantiate the model
    model = LM_LSTM(emb_size, hid_size, vocab_len,
                    pad_index=lang.word2id["<pad>"]).to(DEVICE)

    # loading pre-trained model
    # path = 'model_name.pt'
    # model.load_state_dict(torch.load(path))

    # Initialize the model weights
    model.apply(init_weights)  # call when training from scratch

    # Define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=lang.word2id["<pad>"], reduction='sum')

    # Define the training parameters
    clip = 5  # Clip the gradient
    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1, n_epochs))

    group_name = "lstm_emb_size"
    run_name = f"emb_{emb_size}"

    # Initialize W&B
    run = wandb.init(
        project="NLU-project",
        group=group_name,
        name=run_name,
        config={
            "learning_rate": lr,
            "batch_size": 64,
            "hidden_size": hid_size,
            "embedding_size": emb_size,
            "optimizer": "SGD",
            "epochs": n_epochs,
            "patience": patience
        }
    )

    for epoch in pbar:
        loss = train_loop(train_loader, optimizer,
                          criterion_train, model, clip)

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)

            # Log values to W&B
            run.log({
                "epoch": epoch,
                "train_loss": loss,
                "dev_loss": loss_dev,
                "dev_perplexity": ppl_dev
            })

            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1

            if patience <= 0:  # Early stopping with patience
                break

    if best_model is not None:
        best_model.to(device=DEVICE)
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
        print('Test ppl: ', final_ppl)
        # Save the best model
        torch.save(best_model.state_dict(), 'bin/best_model.pt')
    else:
        print('No best model found - training did not improve')
    wandb.finish()
