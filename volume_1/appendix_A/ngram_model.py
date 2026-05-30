"""
N-gram based language model implementation.
"""
from collections import Counter
import argparse

import utils


def tokenize_text(text):
    from utils import basic_tokenize
    return basic_tokenize(text)

def build_ngram_model(tokens, n):
    """Build n-gram language model from tokens."""
    model = {}
    for i in range(len(tokens) - n):
        context = tuple(tokens[i:i+n-1])
        next_word = tokens[i+n-1]
        model.setdefault(context, Counter())[next_word] += 1
    return model


def predict_next(model, prompt, n, top_k):
    """Predict next word based on prompt using n-gram model."""
    tokens = tokenize_text(prompt)
    print(f"[DEBUG] Prompt tokens: {tokens}")
    
    if len(tokens) < n - 1:
        return '<too-short>'
        
    context = tuple(tokens[-(n-1):])
    print(f"[DEBUG] Context used for prediction: {context}")
    
    candidates = model.get(context, {})
    if not candidates:
        return '<unk>'

    # Get top k candidates with their counts
    top_candidates = candidates.most_common(top_k)
    
    # Calculate total frequency for this context to get probabilities
    total_count = sum(candidates.values())
    
    # Return candidates with their counts and probabilities
    return [(word, count, count/total_count) for word, count in top_candidates]


def main():
    """Entry point when script is run directly."""
    parser = argparse.ArgumentParser(description="N-gram Language Model")
    parser.add_argument("prompt", help="Input prompt to complete")
    parser.add_argument("--shakespeare", action="store_true", help="Use shakespeare corpus")
    parser.add_argument("--ngram", type=int, default=4, help="Size of n-gram (default=4)")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to show (defauklt=5)")
    args = parser.parse_args()
    
    # Process arguments
    corpus_type = "shakespeare" if args.shakespeare else "nursery"
    
    # Load corpus
    corpus_text = utils.load_corpus(corpus_type)
    
    # Use improved tokenization approach
    tokens = tokenize_text(corpus_text)
    print(f"[INFO] Loaded corpus with {len(tokens)} tokens")
    
    # Build model
    model = build_ngram_model(tokens, args.ngram)
    
    # Generate prediction
    prediction = predict_next(model, args.prompt, args.ngram, args.topk)

    # Print context info
    print(f"\nUsing {args.ngram}-gram model (context window: {args.ngram-1} tokens)")
    
    # Print results with frequencies and probabilities
    if isinstance(prediction, list):
        print(f"\nTop {args.topk} predictions for '{args.prompt}':")
        for i, (word, count, prob) in enumerate(prediction, 1):
            print(f"  {i}. {word} (frequency: {count}, probability: {prob:.4f})")
    else:
        print(f"Prediction for '{args.prompt}': {prediction}")

if __name__ == "__main__":
    main()
