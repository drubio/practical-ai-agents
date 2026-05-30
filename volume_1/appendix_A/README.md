# Language Model evolution

This appendix provides three different implementations of language models to compare their effectiveness on text prediction tasks:

- **N-gram Model** (`ngram_model.py`)
- **LSTM Model** (`lstm_model.py`)
- **Transformer Model** (`transformer_model.py`)

By default all models run on a nursery rhyme corpus or can also use a shakespeare corpus (using the `--shakespeare` flag), to generate word predictions based on a given prompt.

## Setup

Install the required dependencies in this folder:

```bash
pip install -r requirements.txt
```

NOTE: This appendix does **not** use the repository-level shared libraries in `shared/`

## Usage

All scripts are command-line runnable and use default values for every parameter. Run the script with `--help` to see all the available parameters that can be overriden with flags

Below are example commands for each language model:

### N-gram

```bash
python ngram_model.py 'Twinkle, Twinkle, Little'
python ngram_model.py 'A rose by any' --shakespeare --ngram 4 --topk 5
```

### LSTM

```bash
python lstm_model.py 'Twinkle, Twinkle, Little' --context 15 --maxwords 5
python lstm_model.py 'A rose by any' --shakespeare --context 15 --epochs 10 --maxwords 5
```

### Transformer

```bash
python transformer_model.py 'Twinkle, Twinkle, Little' --context 15 --maxwords 5
python transformer_model.py 'A rose by any' --shakespeare --context 15 --epochs 5 --maxwords 5
```

## Notes

- Models and vocabularies are cached to disk after training.
- The nursery corpus is downloaded from Project Gutenberg and Poetry Foundation if not cached.
- The shakespeare corpus is a tiny shakespeare compilation downloaded from github if not cached.
- Transformer and LSTM models require training unless previously saved models are available.
- Use `--force_train` to retrain even if a cached model exists.

## Directory Structure

- `utils.py`: Shared functions for tokenization, vocab building, and corpus loading.
- `corpus_cache/`: Stores downloaded and cleaned corpus files.
- `vocab_cache/`: Caches vocabularies built from the corpus.
- `lstm_models/`: Stores LSTM trained model weights.
- `transformer_models/`: Stores transformer trained model weights.
