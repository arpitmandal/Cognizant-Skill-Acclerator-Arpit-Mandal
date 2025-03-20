import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from lstm_text_generation import TextGenerator

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def main():
    """Train an LSTM model on The Old Worcester Jug text."""
    
    # Create output directories
    output_dir = "jug_lstm_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save input text to a file
    input_file = os.path.join(output_dir, "input_text.txt")
    with open(input_file, 'w', encoding='utf-8') as f:
        f.write(open("THE OLD WORCESTER JUG.txt", 'r', encoding='utf-8').read())
    
    print("Starting training on 'The Old Worcester Jug' text...")
    
    # Initialize text generator with smaller model parameters for faster training
    text_gen = TextGenerator(
        input_file=input_file,
        sequence_length=30,     # Shorter sequences for faster training
        batch_size=32,          # Smaller batch size
        embedding_dim=128,      # Smaller embedding dimension
        lstm_units=256,         # Fewer LSTM units
        dropout_rate=0.2,
        learning_rate=0.001,
        temperature=0.7         # Lower temperature for more focused text
    )
    
    # Train model with embedding for better performance on smaller datasets
    print("\nTraining LSTM model...")
    text_gen.train(epochs=30, embedding=True, save_dir=output_dir)
    
    # Generate text samples with different temperatures
    print("\nGenerating text samples with different temperatures...")
    
    # Get a seed text from the original text
    seed_start = 500  # Start somewhere in the text, not the beginning
    seed_text = text_gen.text[seed_start:seed_start + text_gen.sequence_length]
    
    # Generate with different temperatures
    temps = [0.5, 0.7, 1.0]
    
    for temp in temps:
        text_gen.temperature = temp
        gen_text = text_gen.generate_text(seed_text, length=300, embedding=True)
        
        print(f"\n--- Temperature: {temp} ---")
        print(gen_text)
        
        # Save generated text
        with open(os.path.join(output_dir, f"gen_text_temp_{temp}.txt"), 'w', encoding='utf-8') as f:
            f.write(gen_text)
    
    print("\nTraining and generation complete.")
    print(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    main()