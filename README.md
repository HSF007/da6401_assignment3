# DA6401 Assignment 3

## Overview

The implementation uses a sequence-to-sequence model with attention mechanism and supports various RNN cell types (RNN, GRU, LSTM). This code includes features for:
- Training models on different languages
- Hyperparameter optimization using Weights & Biases (W&B) sweeps
- Comprehensive evaluation with visualizations
- Error analysis and reporting

## Dataset
Dateset used is [Dakshina dataset](https://github.com/google-research-datasets/dakshina) created by Google Research, which includes:


## Structure

```
.
├── dakshina_dataset_v1.0/     # dataset
├── predictions
   ├── predictions_attention/ # Predition made by attention model
   ├── predictions_vanilla/   # Preditions made by vanilla model
├── config.py                 # Configuration management
├── dataloader.py             # Data loading and preprocessing
├── evaluate.py               # Model evaluation and visualization on test dataset
├── evaluation_attention.py   # attention model evalution and visualization on test dataset
├── main.py                   # Main entry point to run all train, sweep and evaluate files
├── main_attention.py         # To run all train, sweep and evaluate files
├── seq2seq.py                # Sequence-to-sequence model implementation
├── seq2seq_attention.py      # Sequence-to-sequence model with attention implementation
├── sweep.py                  # Hyperparameter optimization with W&B
├── sweep_attention.py        # Hyperparameter optimization for attention with W&B
├── train.py                  # Model training
├── train_attention.py        # attention Model training
└── utils.py                  # Utility functions
```

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd dakshina-transliteration
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install torch numpy pandas matplotlib seaborn tqdm wandb
   ```

4. Download the Dakshina dataset:
   ```bash
   # Download from https://github.com/google-research-datasets/dakshina
   # Extract to a directory named 'dakshina_dataset_v1.0' in the project root
   # or update the data_dir in config.py to point to your dataset location
   ```

5. Set up Weights & Biases:
   ```bash
   wandb login
   ```

6. Check config.py
   Changes to make if necessary
   * change base_dir if dataset is not in the same folder as code please point where the dataset is not the exact location of dataset
   ```bash
   base_dir = os.path.dirname(os.path.abspath(__file__))
   ```
   * You can change model parameters in this file also

   * Pleas change W&B parameters (**Must change**)
   ```bash
   # W&B parameters
   wandb_project = "dakshina_transliteration"
   wandb_entity = "da24m008-iit-madras"  # Set your W&B username
   ```

   * 




## Usage for vanilla models

### Training a Model

To train a model for Hindi (default):

```bash
python main.py --mode train
```

To train for a different language:

```bash
python main.py --mode train --language ta  # Tamil
```

Customize training parameters:

```bash
python main.py --mode train --language hi --embed_size 128 --hidden_size 256 --cell_type LSTM --dropout 0.3 --batch_size 64 --learning_rate 0.001 --epochs 30
```

### Running Hyperparameter Sweeps

Run a hyperparameter sweep to find the optimal model configuration:

```bash
python main.py --mode sweep --language hi --count 50
```

Continue an existing sweep:

```bash
python main.py --mode sweep --language hi --sweep_id <sweep-id> --count 20
```

### Evaluating Models on test dataset from sweep

Evaluate a trained model with default language:

**While evaluting model you must provide sweep_id**

```bash
python main.py --mode evaluate --sweep_id <sweep-id>
```

Evaluate a specific model from a sweep:

```bash
python main.py --mode evaluate --language ta --sweep_id <sweep-id>
```

## Usage for attention based models

### Training a Model

To train a model for Hindi (default):

```bash
python main_attention.py --mode train
```

To train for a different language:

```bash
python main_attention.py --mode train --language ta  # Tamil
```

Customize training parameters:

```bash
python main_attention.py --mode train --language hi --embed_size 128 --hidden_size 256 --cell_type LSTM --dropout 0.3 --batch_size 64 --learning_rate 0.001 --epochs 30
```

### Running Hyperparameter Sweeps

Run a hyperparameter sweep to find the optimal model configuration:

```bash
python main_attention.py --mode sweep --count 50
```

Continue an existing sweep:

```bash
python main_attention.py --mode sweep --language hi --sweep_id <sweep-id> --count 20
```

### Evaluating Models on test dataset from sweep

Evaluate a trained model with default language:

**While evaluting model you must provide sweep_id**

```bash
python main_attention.py --mode evaluate --sweep_id <sweep-id>
```

Evaluate a model with different language (not default) from a sweep:

```bash
python main_attention.py --mode evaluate --language ta --sweep_id <sweep-id>
```

## Model Architecture

The transliteration model is a sequence-to-sequence architecture with:

- Character-level tokenization
- Embedding layers for source and target
- Encoder-decoder architecture with RNN/GRU/LSTM cells
- Teacher forcing during training
- Greedy/beam search decoding

The model supports varying:
- Embedding dimensions
- Hidden layer sizes
- Number of encoder/decoder layers
- RNN cell types (RNN, GRU, LSTM)
- Dropout rates

## Configuration

The project uses a centralized configuration in `config.py` which manages:

- Data paths and file locations
- Model architecture parameters
- Training hyperparameters
- Evaluation settings
- W&B integration settings

**Important**
Configuration can be overridden via command-line arguments in `main.py` and `main_attetion.py`.

## Visualization and Analysis

The evaluation pipeline generates:

- Accuracy metrics on test data
- Sample visualizations of correct and incorrect predictions
- Error analysis by word length
- Character-level error analysis
- Error patterns in incorrect predictions

All visualizations are saved to the `predictions/predictions_vanilla/` and ``predictions/predictions_attention/` directory and also logged to W&B if enabled.

## Weights & Biases Integration

The project integrates with [Weights & Biases](https://wandb.ai/) for:

- Experiment tracking
- Hyperparameter optimization
- Model versioning
- Visualization
- Collaboration

W&B features used:
- Logging metrics during training
- Saving and versioning models as artifacts
- Hyperparameter sweeps
- Tables for prediction visualization
- Image logging for error analysis