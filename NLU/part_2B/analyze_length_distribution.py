import os
import matplotlib.pyplot as plt
from utils import load_data, tokenizer


def analyze_length_distribution(data_dir="dataset/ATIS"):
    """
    Analyzes and visualizes the tokenized utterance length distribution for the
    ATIS dataset.
    - This information will be used to choose the maximum sequence length for the
    BERT model.
    - The idea is to avoid padding too much, which can lead to a loss in
    performance, but also to avoid truncating too much, which can lead to losing
    important information.
    """
    print("Loading data...")
    train_raw = load_data(os.path.join(data_dir, "train.json"))
    test_raw = load_data(os.path.join(data_dir, "test.json"))

    corpus = train_raw + test_raw

    print("Tokenizing utterances and calculating lengths...")
    tokenized_lengths = []
    for item in corpus:
        tokenized_output = tokenizer(item['utterance'], return_tensors='pt')
        # Get the length of the input_ids tensor
        length = tokenized_output['input_ids'].shape[1]
        tokenized_lengths.append(length)

    # --- Find and print the maximum length ---
    max_len = max(tokenized_lengths)
    print(f"Maximum tokenized utterance length: {max_len}")

    # --- Visualization ---
    print("Creating visualization...")
    plt.figure(figsize=(12, 6))
    plt.hist(tokenized_lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Tokenized Utterance Lengths (ATIS Dataset)')
    plt.xlabel('Tokenized Length (BERT)')
    plt.ylabel('Frequency')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save the plot
    output_path = "bin/utterance_length_distribution.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Distribution plot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    analyze_length_distribution()
