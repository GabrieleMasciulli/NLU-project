# Natural Language Understanding Project

This project explores various Natural Language Understanding (NLU) and Language Modeling (LM) tasks. It is divided into two main parts: Language Modeling (LM) and Natural Language Understanding (NLU), each with its own sub-projects and experiments.

## Project Structure

The repository is organized as follows:

```
├── LM/
│   ├── part_1A/            # Part 1A: Basic LSTM for Language Modeling
│   ├── part_1B/            # Part 1B: Advanced LSTM techniques (Weight Tying, Variational Dropout, etc.)
│   └── vanilla_RNN/        # Initial experiments with vanilla RNNs
├── NLU/
│   ├── part_2A/            # Part 2A: Joint Intent Recognition & Slot Filling via LSTMs
│   └── part_2B/            # Part 2B: Joint Intent Recognition & Slot Filling via BERT fine-tuning
├── .gitignore
└── README.md
```

## Language Modeling (LM)

This part of the project focuses on the progressive refinement of a language model based on LSTM architectures on the Penn Treebank (PTB) dataset. The work was conducted in two stages: (1.A) a sequence of baseline enhancements and (1.B) integration of advanced regularization techniques. The goal was to minimize test perplexity (PPL), with emphasis on the regularization phase, where improvements were guided by techniques from the AWD-LSTM framework.

### Part 1.A: Baseline Enhancements

- **Objective**: Substitute the initial RNN-based architecture with a multi-layer LSTM and conduct extensive hyperparameter tuning.
- **Enhancements**:
  - Introduced two dropout layers (one after embedding, one before final projection) to reduce overfitting, following Zaremba et al. (2014).
  - Replaced SGD with AdamW, which accelerated convergence but resulted in slightly worse final test perplexity.

### Part 1.B: Advanced Regularization

- **Objective**: Improve the baseline LSTM by integrating advanced regularization techniques from Merity et al. (2017).
- **Techniques**:
  - **Weight Tying**: The embedding and final projection leyer weights are shared.
  - **Variational Dropout**: Applied `LockedDropout` to word embeddings and between LSTM layers, using the same dropout mask across timesteps for temporal consistency.
  - **Non-monotonically Triggered Averaged SGD (NT-ASGD)**: Implemented a custom optimizer that switches from SGD to Averaged SGD based on a non-monotonic validation perplexity trigger, improving training dynamics.
  - **GloVe Embeddings**: Experimented with pre-trained GloVe embeddings to initialize the embedding layer.

### Results

The main goal was to achieve the lowest possible perplexity on the test set. Each enhancement led to a consistent improvement in model performance, with NT-ASGD providing the largest single improvement.

| Run Description            | Test PPL |
| -------------------------- | -------- |
| A1 Base LSTM + tuning      | 140.90   |
| A2 + Zaremba-style dropout | 104.28   |
| A3 + AdamW optimizer       | 111.39   |
| B1 Weight Tying only       | 92.52    |
| B2 + Variational Dropout   | 89.92    |
| B3 + NT-ASGD (final)       | 77.34    |
| (Opt) + GloVe embeddings   | 75.59    |

_Note: For exact numbers, please run the evaluation scripts._

## Natural Language Understanding (NLU)

This project addresses the joint tasks of intent classification and slot filling using the ATIS dataset. Two approaches are explored:

- **Part 2.A**: A baseline LSTM model is enhanced with bidirectionality and dropout.
- **Part 2.B**: A pre-trained BERT model is fine-tuned using a multi-task architecture inspired by the CTRAN framework, with attention to handling sub-tokenization and enriching BERT’s representations using CNN layers.

### Part 2.A – LSTM Enhancements

- **Objective**: Incrementally modify the baseline LSTM model.
- **Enhancements**:
  - **Bidirectional LSTMs**: Enabled each token’s representation to benefit from both past and future context, improving both slot filling and intent classification.
  - **Dropout**: Introduced a dropout layer between the LSTM and output layers to regularize the model.
  - **Weighted Loss**: Experimented with a weighted cross-entropy loss to automatically balance the two tasks, but the standard summed loss performed better.

### Part 2.B – BERT Fine-Tuning with CNN

- **Objective**: Fine-tune a BERT model for the joint tasks, inspired by the CTRAN architecture.
- **Architecture**: Combines three components:
  1.  **BERT**: Provides rich contextual embeddings.
  2.  **Multi-scale CNNs**: Extract local n-gram patterns (kernel sizes 1, 2, 3, 5).
  3.  **Transformer Encoder**: Models global dependencies between CNN features.
- **Task-Specific Branches**:
  - **Slot Filling**: BERT's sequence output is passed through the CNN-Transformer pipeline. A selective loss masking strategy is used to handle sub-tokenization, ensuring only the first sub-token of each original word contributes to the loss.
  - **Intent Classification**: BERT’s pooled output ([CLS] token) is passed through a classification head.
- The model jointly optimizes a summed cross-entropy loss for both tasks.

### Results

Evaluation on the ATIS test set.

| Model        | Intent Acc. (%) | Slot F1 (%) |
| ------------ | --------------- | ----------- |
| LSTM + BiDir | 94.44           | 92.51       |
| + Dropout    | 94.96           | 93.05       |
| BERT + CNN   | 97.76           | 95.06       |

_Note: For exact numbers, please run the evaluation scripts._

## How to Run

Each sub-project (`part_1A`, `part_1B`, etc.) contains its own set of scripts to train and evaluate the models.

1.  **Navigate to a sub-project directory**:
    ```bash
    cd LM/part_1A
    ```
2.  **Install dependencies** (it is recommended to use a virtual environment):
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download the data**: The scripts will typically handle data download if it's not present.
4.  **Train a model**:
    ```bash
    python main.py
    ```
5.  **Evaluate models**:
    ```bash
    python eval_models.py
    ```

Refer to the `main.py` and `eval_models.py` scripts in each directory for
specific configurations and hyperparameters.
