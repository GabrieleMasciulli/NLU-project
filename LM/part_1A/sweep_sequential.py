import sys
sys.path.append(".")  # Ensure current directory is in path

from main import main

# Define your search spaces
hid_sizes = [256, 512, 650]
emb_sizes = [200, 400, 650]
n_layers_list = [1, 2, 3]

# Fixed parameters
lr = 1.0
batch_size_train = 64
batch_size_eval = 128
epochs = 30
clip = 5.0
wandb_project = "NLU-project-part1A"
wandb_group_prefix = "sequential_sweep"

def run_main(hid_size, emb_size, n_layers):
    ppl = main(
        hid_size=hid_size,
        emb_size=emb_size,
        n_layers=n_layers,
        lr=lr,
        batch_size_train=batch_size_train,
        batch_size_eval=batch_size_eval,
        epochs=epochs,
        clip=clip,
        wandb_project=wandb_project,
        wandb_group_prefix=wandb_group_prefix
    )
    return ppl

# 1. Tune hid_size
best_hid_size = None
best_ppl = float('inf')
for hid_size in hid_sizes:
    ppl = run_main(hid_size, emb_sizes[0], n_layers_list[0])
    if ppl < best_ppl:
        best_ppl = ppl
        best_hid_size = hid_size

# 2. Tune emb_size
best_emb_size = None
best_ppl = float('inf')
for emb_size in emb_sizes:
    ppl = run_main(best_hid_size, emb_size, n_layers_list[0])
    if ppl < best_ppl:
        best_ppl = ppl
        best_emb_size = emb_size

# 3. Tune n_layers
best_n_layers = None
best_ppl = float('inf')
for n_layers in n_layers_list:
    ppl = run_main(best_hid_size, best_emb_size, n_layers)
    if ppl < best_ppl:
        best_ppl = ppl
        best_n_layers = n_layers

print(f"Best configuration: hid_size={best_hid_size}, emb_size={best_emb_size}, n_layers={best_n_layers}, dev_ppl={best_ppl}")