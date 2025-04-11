# Vanilla RNN Language Model

This is the implementation of a vanilla RNN language model for the NLU course project.

## Project Structure

- `main.py`: Main script to run the training and evaluation
- `model.py`: Contains the RNN model architecture
- `utils.py`: Dataset preprocessing and loading utilities
- `functions.py`: Training and evaluation functions
- `dataset/`: Contains the Penn TreeBank dataset
- `bin/`: Directory for storing trained models

## Usage

To train the model:

```bash
python main.py
```

The best model will be saved in the `bin/` directory.

## Model Architecture

- Embedding size: 300
- Hidden size: 256
- Optimizer: SGD
- Learning rate: 1.0
- Gradient clipping: 5.0
