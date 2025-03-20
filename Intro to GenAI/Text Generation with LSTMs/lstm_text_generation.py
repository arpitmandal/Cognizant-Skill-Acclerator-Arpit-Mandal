import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os
import re
import random
import time
import argparse

class TextGenerator:
    """
    A class to train and generate text using LSTM networks.
    """
    
    def __init__(self, input_file=None, sequence_length=40, batch_size=64, 
                 embedding_dim=256, lstm_units=512, dropout_rate=0.2, 
                 learning_rate=0.001, temperature=1.0):
        """
        Initialize the TextGenerator with the specified parameters.
        
        Args:
            input_file (str): Path to the text file for training
            sequence_length (int): Length of input sequences
            batch_size (int): Batch size for training
            embedding_dim (int): Dimensionality of embedding layer
            lstm_units (int): Number of units in LSTM layer
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for Adam optimizer
            temperature (float): Controls randomness in generation (higher = more random)
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.temperature = temperature
        
        # Initialize empty attributes
        self.model = None
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.text = ""
        
        # Load data if a file is provided
        if input_file:
            self.load_data(input_file)
            
    def load_data(self, input_file):
        """
        Load and preprocess text data from a file.
        
        Args:
            input_file (str): Path to the text file
        """
        try:
            # Read the text file
            with open(input_file, 'r', encoding='utf-8') as f:
                self.text = f.read().lower()
                
            # Clean text - remove extra whitespace and unusual characters if needed
            self.text = re.sub(r'\s+', ' ', self.text)
            
            print(f"Loaded text file: {input_file}")
            print(f"Text length: {len(self.text)} characters")
            
            # Create vocabulary (unique characters)
            chars = sorted(list(set(self.text)))
            self.vocab_size = len(chars)
            
            # Create mapping dictionaries
            self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
            self.idx_to_char = {idx: char for idx, char in enumerate(chars)}
            
            print(f"Vocabulary size: {self.vocab_size} unique characters")
            
        except Exception as e:
            print(f"Error loading file: {e}")
            raise
    
    def create_sequences(self):
        """
        Create input-output sequence pairs from the text data.
        
        Returns:
            tuple: (X, y) where X is the input sequences and y is the expected next characters
        """
        # Create character sequences
        sequences = []
        next_chars = []
        
        for i in range(0, len(self.text) - self.sequence_length):
            sequences.append(self.text[i:i + self.sequence_length])
            next_chars.append(self.text[i + self.sequence_length])
        
        print(f"Created {len(sequences)} sequences")
        
        # Vectorize sequences
        X = np.zeros((len(sequences), self.sequence_length, self.vocab_size), dtype=np.bool_)
        y = np.zeros((len(sequences), self.vocab_size), dtype=np.bool_)
        
        for i, sequence in enumerate(sequences):
            for t, char in enumerate(sequence):
                X[i, t, self.char_to_idx[char]] = 1
            y[i, self.char_to_idx[next_chars[i]]] = 1
        
        return X, y
    
    def build_model(self):
        """
        Build and compile the LSTM model.
        """
        # Create model
        self.model = Sequential([
            LSTM(self.lstm_units, input_shape=(self.sequence_length, self.vocab_size), 
                 return_sequences=True),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units),
            Dropout(self.dropout_rate),
            Dense(self.vocab_size, activation='softmax')
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        
        self.model.summary()
    
    def build_model_with_embedding(self):
        """
        Build and compile the LSTM model with an embedding layer.
        """
        # Create model with embedding layer
        self.model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.sequence_length),
            LSTM(self.lstm_units, return_sequences=True),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units),
            Dropout(self.dropout_rate),
            Dense(self.vocab_size, activation='softmax')
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        
        self.model.summary()
    
    def train(self, epochs=50, embedding=False, save_dir='models'):
        """
        Train the LSTM model.
        
        Args:
            epochs (int): Number of training epochs
            embedding (bool): Whether to use embedding layer
            save_dir (str): Directory to save model checkpoints
        
        Returns:
            keras.callbacks.History: Training history
        """
        # Create sequences
        if embedding:
            # For embedding layer, we need integer sequences
            sequences = []
            next_chars = []
            
            for i in range(0, len(self.text) - self.sequence_length):
                sequences.append([self.char_to_idx[char] for char in self.text[i:i + self.sequence_length]])
                next_chars.append(self.char_to_idx[self.text[i + self.sequence_length]])
            
            X = np.array(sequences)
            y = tf.keras.utils.to_categorical(next_chars, num_classes=self.vocab_size)
            
            # Build model with embedding
            self.build_model_with_embedding()
        else:
            # For one-hot encoding, use the create_sequences method
            X, y = self.create_sequences()
            
            # Build standard model
            self.build_model()
        
        # Create directory for model checkpoints
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup callbacks
        checkpoint_path = os.path.join(save_dir, 'model_{epoch:02d}_{loss:.4f}.h5')
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', save_best_only=True, 
                                     verbose=1, mode='min')
        early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1, 
                                       restore_best_weights=True)
        
        # Train model
        start_time = time.time()
        history = self.model.fit(X, y, batch_size=self.batch_size, epochs=epochs, 
                                verbose=1, callbacks=[checkpoint, early_stopping])
        
        # Print training time
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save the final model
        final_model_path = os.path.join(save_dir, 'final_model.h5')
        self.model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(save_dir, 'training_loss.png'))
        plt.close()
        
        return history
    
    def generate_text(self, seed_text, length=200, embedding=False):
        """
        Generate text with the trained model.
        
        Args:
            seed_text (str): Starting text for generation
            length (int): Number of characters to generate
            embedding (bool): Whether model uses embedding
        
        Returns:
            str: Generated text
        """
        if not self.model:
            raise ValueError("Model is not trained. Train the model first.")
        
        if len(seed_text) < self.sequence_length:
            raise ValueError(f"Seed text must be at least {self.sequence_length} characters long")
        
        # Use the last sequence_length characters as seed
        current_text = seed_text[-self.sequence_length:]
        generated_text = current_text
        
        # Generate characters
        for _ in range(length):
            if embedding:
                # For models with embedding layer
                x_pred = np.array([[self.char_to_idx[char] for char in current_text]])
            else:
                # One-hot encode the input sequence
                x_pred = np.zeros((1, self.sequence_length, self.vocab_size))
                for t, char in enumerate(current_text):
                    x_pred[0, t, self.char_to_idx[char]] = 1
            
            # Get model predictions
            preds = self.model.predict(x_pred, verbose=0)[0]
            
            # Apply temperature to adjust prediction distribution
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / self.temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            
            # Sample next character based on the predictions
            next_index = np.random.choice(len(preds), p=preds)
            next_char = self.idx_to_char[next_index]
            
            # Append to generated text
            generated_text += next_char
            current_text = current_text[1:] + next_char
        
        return generated_text
    
    def save_model(self, filepath):
        """Save the trained model."""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Train the model first.")
    
    def load_model(self, filepath):
        """Load a previously saved model."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


def main():
    """Main function to handle command-line execution."""
    parser = argparse.ArgumentParser(description='LSTM Text Generation')
    
    # File and mode arguments
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to the input text file')
    parser.add_argument('--mode', type=str, choices=['train', 'generate', 'both'],
                        default='both', help='Mode of operation')
    
    # Model and training parameters
    parser.add_argument('--sequence-length', type=int, default=40,
                        help='Length of input sequences')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--embedding-dim', type=int, default=256,
                        help='Dimension of embedding layer')
    parser.add_argument('--lstm-units', type=int, default=512,
                        help='Number of units in LSTM layer')
    parser.add_argument('--dropout-rate', type=float, default=0.2,
                        help='Dropout rate for regularization')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate for Adam optimizer')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--use-embedding', action='store_true',
                        help='Use embedding layer instead of one-hot encoding')
    
    # Generation parameters
    parser.add_argument('--seed-text', type=str,
                        help='Seed text for generation (required if mode is generate)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (higher = more random)')
    parser.add_argument('--length', type=int, default=200,
                        help='Length of generated text')
    
    # Model loading/saving
    parser.add_argument('--model-path', type=str, default='models/final_model.h5',
                        help='Path to save/load model')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to save models and visualizations')
    
    args = parser.parse_args()
    
    # Initialize the text generator
    text_gen = TextGenerator(
        input_file=args.input_file,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        temperature=args.temperature
    )
    
    # Train the model
    if args.mode in ['train', 'both']:
        print("Training LSTM text generation model...")
        text_gen.train(epochs=args.epochs, embedding=args.use_embedding, 
                       save_dir=args.output_dir)
    
    # Load model for generation if needed
    if args.mode == 'generate' and os.path.exists(args.model_path):
        text_gen.load_model(args.model_path)
    
    # Generate text
    if args.mode in ['generate', 'both']:
        if not args.seed_text and args.mode == 'generate':
            print("Error: Seed text is required for text generation.")
            return
        
        # If no seed text provided but in 'both' mode, use a sample from the input text
        seed_text = args.seed_text
        if not seed_text and args.mode == 'both':
            start_idx = random.randint(0, len(text_gen.text) - text_gen.sequence_length - 1)
            seed_text = text_gen.text[start_idx:start_idx + text_gen.sequence_length]
            print(f"Using random seed text: '{seed_text}'")
        
        # Generate text
        generated_text = text_gen.generate_text(
            seed_text=seed_text,
            length=args.length,
            embedding=args.use_embedding
        )
        
        print("\nGenerated Text:")
        print(generated_text)
        
        # Save the generated text
        output_file = os.path.join(args.output_dir, 'generated_text.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        print(f"Generated text saved to {output_file}")


if __name__ == "__main__":
    main()
