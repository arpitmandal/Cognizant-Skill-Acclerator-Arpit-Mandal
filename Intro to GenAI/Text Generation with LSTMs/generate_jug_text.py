import os
import sys
import numpy as np
import tensorflow as tf
from lstm_text_generation import TextGenerator

def main():
    """Generate text using a previously trained LSTM model."""
    
    if len(sys.argv) < 2:
        print("Usage: python generate_jug_text.py \"Your seed text here\" [temperature] [length]")
        sys.exit(1)
    
    # Get parameters from command line
    seed_text = sys.argv[1]
    temperature = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    length = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    
    output_dir = "jug_lstm_output"
    model_path = os.path.join(output_dir, "final_model.h5")
    input_file = os.path.join(output_dir, "input_text.txt")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        print("Please run the training script first.")
        sys.exit(1)
    
    # Initialize text generator with the same parameters
    text_gen = TextGenerator(
        input_file=input_file,
        sequence_length=30,     # Must match training parameter
        embedding_dim=128,
        lstm_units=256,
        temperature=temperature
    )
    
    # Load the trained model
    text_gen.load_model(model_path)
    
    # Ensure seed text is long enough
    if len(seed_text) < text_gen.sequence_length:
        print(f"Seed text must be at least {text_gen.sequence_length} characters.")
        print(f"Current seed text is only {len(seed_text)} characters.")
        
        # Get a random sequence from the text as seed
        start_idx = np.random.randint(0, len(text_gen.text) - text_gen.sequence_length)
        seed_text = text_gen.text[start_idx:start_idx + text_gen.sequence_length]
        print(f"Using random seed text: '{seed_text}'")
    
    # Generate text
    print(f"\nGenerating {length} characters with temperature {temperature}...")
    generated_text = text_gen.generate_text(seed_text, length=length, embedding=True)
    
    print("\nGenerated Text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)
    
    # Save the generated text
    output_file = f"generated_jug_text_t{temperature}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(generated_text)
    
    print(f"\nText saved to {output_file}")


if __name__ == "__main__":
    main()