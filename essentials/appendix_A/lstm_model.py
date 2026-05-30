"""
LSTM-based language model implementation.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from pathlib import Path

import utils

# ---------- Configuration ----------
MODEL_CACHE_DIR = "lstm_models"


class TextDataset(Dataset):
    def __init__(self, data, context=3):
        self.data = data
        self.context = context

    def __len__(self):
        return len(self.data) - self.context

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx+self.context]),
            torch.tensor(self.data[idx+self.context]),
        )


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_size=64, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])


def train_model(model, loader, epochs=2, device=None):
    """Train the LSTM model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
            
        avg_loss = total_loss / total_batches
        print(f"[INFO] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    return model


def predict_next(model, vocab, idx_to_word, prompt, context, top_k=5, temperature=0.7, max_words=1):
    """Generate text predictions using the trained LSTM model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    tokens = utils.basic_tokenize(prompt)[-context:]
    generated = tokens.copy()

    # Store all predictions for each step
    all_predictions = []

    for _ in range(max_words):
        if len(generated) < context:
            # Pad with start tokens if needed
            padding = ['<pad>'] * (context - len(generated))
            context_tokens = padding + generated
        else:
            context_tokens = generated[-context:]
        
        # Convert tokens to tensor
        input_ids = torch.tensor([
            [vocab.get(tok, vocab['<unk>']) for tok in context_tokens]
        ], dtype=torch.long).to(device)

        with torch.no_grad():
            output = model(input_ids)

            # Handle temperature
            if temperature == 0:
                # For temperature=0, just use argmax (no randomness)
                top_indices = torch.topk(output, min(top_k, output.size(-1))).indices.squeeze()
                # Set probability to 1.0 for the top item for display purposes
                top_probs = [1.0] + [0.0] * (min(top_k, output.size(-1)) - 1)
            else:
                # Normal case with temperature > 0
                probs = torch.softmax(output / temperature, dim=-1).squeeze()
                topk_results = torch.topk(probs, min(top_k, len(probs)))
                top_indices = topk_results.indices.tolist()
                top_probs = topk_results.values.tolist()

            # Convert indices to words
            if isinstance(top_indices, torch.Tensor):
                # Handle the case where top_indices is a tensor (temperature=0 case)
                options = [(idx_to_word.get(i.item(), '<unk>'), p) 
                           for i, p in zip(top_indices, top_probs)]
            else:
                # Handle the case where top_indices is already a list
                options = [(idx_to_word.get(i, '<unk>'), p) 
                           for i, p in zip(top_indices, top_probs)]

        # Store the predictions for this step
        all_predictions.append(options)
        
        # Add the top prediction to the generated text
        generated.append(options[0][0])

    # Return both the generated text and all predictions
    return " ".join(generated[len(tokens):]), all_predictions



def get_cache_paths(corpus_type, context_size):
    """Generate appropriate cache paths for model files."""
    # Create necessary directories
    os.makedirs(utils.VOCAB_CACHE_DIR, exist_ok=True)
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    
    # Generate filenames with clear suffixes
    vocab_file = Path(utils.VOCAB_CACHE_DIR) / f"vocab_{corpus_type}_ctx{context_size}.json"
    model_file = Path(MODEL_CACHE_DIR) / f"model_{corpus_type}_ctx{context_size}.pt"
    
    return vocab_file, model_file


def main():
    """Entry point when script is run directly."""
    parser = argparse.ArgumentParser(description="LSTM Language Model")
    parser.add_argument("prompt", help="Input prompt to complete")
    parser.add_argument("--shakespeare", action="store_true", help="Use shakespeare corpus")
    parser.add_argument("--context", type=int, default=15, help="Context window size")
    parser.add_argument("--topk", type=int, default=5, help="Show top-k predictions")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--maxwords", type=int, default=5, help="Max number of predicted words")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--force_train", action="store_true", help="Force retraining even if model exists")
    args = parser.parse_args()

    # Process arguments
    corpus_type = "shakespeare" if args.shakespeare else "nursery"
    
    # Load corpus and tokenize
    corpus_text = utils.load_corpus(corpus_type)
    tokens = utils.basic_tokenize(corpus_text)
    print(f"[INFO] Loaded corpus with {len(tokens)} tokens")
    
    # Get cache paths for model and vocabulary
    vocab_file, model_file = get_cache_paths(corpus_type, args.context)
    
    # Build or load vocabulary
    if vocab_file.exists() and not args.force_train:
        vocab = utils.load_vocab(vocab_file)
    else:
        tokens = ['<bos>'] + tokens
        vocab = utils.build_vocab(tokens, min_freq=2)
        utils.save_vocab(vocab, vocab_file)

    # Create mapping from indices to words
    idx_to_word = {i: w for w, i in vocab.items()}
    
    # Encode the entire corpus
    encoded = utils.encode(tokens, vocab)
    
    # Create dataset and dataloader
    dataset = TextDataset(encoded, context=args.context)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create model
    model = LSTMModel(vocab_size=len(vocab))

    # Load or train the model
    if model_file.exists() and not args.force_train:
        try:
            print(f"[INFO] Loading cached model from {model_file}")
            model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        except Exception as e:
            print(f"[WARNING] Error loading model: {e}. Will train a new model.")
            model = train_model(model, loader, epochs=args.epochs)
            torch.save(model.state_dict(), model_file)
            print(f"[INFO] Saved model to {model_file}")
    else:
        print(f"[INFO] Training model with {args.epochs} epochs...")
        model = train_model(model, loader, epochs=args.epochs)
        torch.save(model.state_dict(), model_file)
        print(f"[INFO] Saved model to {model_file}")

    # Generate prediction
    prediction, all_predictions = predict_next(
        model, vocab, idx_to_word,
        args.prompt,
        context=args.context,
        top_k=args.topk,
        temperature=args.temperature,
        max_words=args.maxwords
    )
    
    # Print context info
    print(f"\nUsing LSTM model with context window size: {args.context} tokens")
    
    # Print the complete prediction first
    generated_words = prediction.split()
    print(f"\nLSTM Complete Prediction for '{args.prompt}': {prediction}")
    
    # Print predictions for each word
    for i, predictions in enumerate(all_predictions):
        current_word = generated_words[i] if i < len(generated_words) else "?"
        
        # Get the updated prompt for this step
        if i == 0:
            step_prompt = args.prompt
        else:
            step_prompt = f"{args.prompt} {' '.join(generated_words[:i])}"
            
        print(f"\nStep {i+1}: Top {args.topk} predictions for '{step_prompt}':")
        for j, (word, prob) in enumerate(predictions, 1):
            marker = "→" if word == current_word else " "
            print(f"  {marker} {j}. {word} (probability: {prob:.4f})")
            

if __name__ == "__main__":
    main()
