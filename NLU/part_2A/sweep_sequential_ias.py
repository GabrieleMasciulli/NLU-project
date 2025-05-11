from main import main
import sys
import os
import wandb

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


# Search spaces
hid_sizes = [256, 512, 650]
emb_sizes = [200, 300, 400]
n_layers_list = [1, 2, 3]

# Fixed parameters for ModelIAS training
lr = 0.0001
clip = 5.0
fc_dropout = 0.0
lstm_dropout = 0.0
n_epochs = 100
patience = 10
batch_size_train = 128
batch_size_eval = 64
wandb_project = "NLU-project-part-2A"
wandb_group_prefix = "sequential_sweep_ias_bidir"


# Cache to store results of (hid_size, emb_size, n_layers) -> f1_score
results_cache = {}


def run_training_for_sweep(hid_size, emb_size, n_layers, cache):
    """
    Wrapper function to call the main training function and return the metric to optimize.
    For ModelIAS, we optimize for the F1 score on the development set.
    Uses a cache to avoid re-running identical configurations.
    """
    config = (hid_size, emb_size, n_layers)
    if config in cache:
        print(
            f"\n--- Using cached result for: hid_size={hid_size}, emb_size={emb_size}, n_layers={n_layers} ---")
        return cache[config]

    print(
        f"\n--- Sweeping with: hid_size={hid_size}, emb_size={emb_size}, n_layers={n_layers} ---")

    # Ensure wandb login
    try:
        # Allow anonymous login if not configured
        wandb.login(anonymous="allow")
    except Exception as e:
        print(
            f"Could not login to WandB: {e}. Proceeding without logging for this run, but sweep might fail.")

    dev_f1 = main(
        hid_size=hid_size,
        emb_size=emb_size,
        lr=lr,
        clip=clip,
        fc_dropout=fc_dropout,
        lstm_dropout=lstm_dropout,
        n_epochs=n_epochs,
        patience=patience,
        batch_size_train=batch_size_train,
        batch_size_eval=batch_size_eval,
        wandb_project=wandb_project,
        wandb_group_prefix=f"{wandb_group_prefix}_h{hid_size}_emb{emb_size}_l{n_layers}",
        n_layers=n_layers
    )
    cache[config] = dev_f1  # Store result in cache
    return dev_f1

# Sequential Tuning Logic (Higher F1 is better)


# 1. Tune hid_size
print("\n--- Tuning Hidden Size (hid_size) ---")
best_hid_size = None
best_f1_for_hid = -1.0  # Initialize with a value lower than any possible F1
# Use default/first values for other hyperparameters during this stage
default_emb_size = emb_sizes[0]
default_n_layers = n_layers_list[0]
for hid_size_candidate in hid_sizes:
    current_f1 = run_training_for_sweep(
        hid_size_candidate, default_emb_size, default_n_layers, results_cache)
    print(
        f"  hid_size={hid_size_candidate}, emb_size={default_emb_size}, n_layers={default_n_layers} -> Dev F1: {current_f1:.4f}")
    if current_f1 > best_f1_for_hid:
        best_f1_for_hid = current_f1
        best_hid_size = hid_size_candidate
print(f"Best hid_size: {best_hid_size} with Dev F1: {best_f1_for_hid:.4f}")

# 2. Tune emb_size (using the best_hid_size found)
print("\n--- Tuning Embedding Size (emb_size) ---")
best_emb_size = None
best_f1_for_emb = -1.0
# Use default/first value for n_layers during this stage
for emb_size_candidate in emb_sizes:
    current_f1 = run_training_for_sweep(
        best_hid_size, emb_size_candidate, default_n_layers, results_cache)
    print(
        f"  hid_size={best_hid_size}, emb_size={emb_size_candidate}, n_layers={default_n_layers} -> Dev F1: {current_f1:.4f}")
    if current_f1 > best_f1_for_emb:
        best_f1_for_emb = current_f1
        best_emb_size = emb_size_candidate
print(
    f"Best emb_size: {best_emb_size} with Dev F1: {best_f1_for_emb:.4f} (using hid_size={best_hid_size})")

# 3. Tune n_layers (using best_hid_size and best_emb_size found)
print("\n--- Tuning Number of Layers (n_layers) ---")
best_n_layers = None
final_best_f1 = -1.0
for n_layers_candidate in n_layers_list:
    current_f1 = run_training_for_sweep(
        best_hid_size, best_emb_size, n_layers_candidate, results_cache)
    print(
        f"  hid_size={best_hid_size}, emb_size={best_emb_size}, n_layers={n_layers_candidate} -> Dev F1: {current_f1:.4f}")
    if current_f1 > final_best_f1:
        final_best_f1 = current_f1
        best_n_layers = n_layers_candidate
print(f"Best n_layers: {best_n_layers} with Dev F1: {final_best_f1:.4f} (using hid_size={best_hid_size}, emb_size={best_emb_size})")

print("\n--- Sequential Sweep Complete ---")
print(f"Best configuration found:")
print(f"  Hidden Size (hid_size): {best_hid_size}")
print(f"  Embedding Size (emb_size): {best_emb_size}")
print(f"  Number of Layers (n_layers): {best_n_layers}")
print(f"  Corresponding Best Dev F1-score: {final_best_f1:.4f}")
print(
    f"  Fixed parameters during sweep: lr={lr}, clip={clip}, fc_dropout={fc_dropout}, lstm_dropout={lstm_dropout}, epochs={n_epochs}, patience={patience}")

if __name__ == "__main__":
    print("Starting sequential hyperparameter sweep for ModelIAS...")
