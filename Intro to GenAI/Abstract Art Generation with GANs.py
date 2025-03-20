import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
from PIL import Image
import time
import argparse
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class AbstractArtGAN:
    """
    A GAN implementation for generating abstract art images.
    """
    
    def __init__(self, img_shape=(64, 64, 3), latent_dim=100):
        """
        Initialize the GAN with the given parameters.
        
        Parameters:
        img_shape (tuple): The shape of the images (height, width, channels)
        latent_dim (int): The dimension of the latent space
        """
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            metrics=['accuracy']
        )
        
        # Build the generator
        self.generator = self.build_generator()
        
        # The generator takes noise as input and generates images
        z = layers.Input(shape=(self.latent_dim,))
        img = self.generator(z)
        
        # For the combined model, we only train the generator
        self.discriminator.trainable = False
        
        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)
        
        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = models.Model(z, validity)
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        )
        
        print("GAN model initialized with:")
        print(f"  Image shape: {img_shape}")
        print(f"  Latent dimension: {latent_dim}")
    
    def build_generator(self):
        """
        Build the generator network.
        
        Returns:
        Model: The generator model
        """
        noise_shape = (self.latent_dim,)
        
        model = models.Sequential(name="Generator")
        
        # Foundation for 16x16 image
        model.add(layers.Dense(128 * 16 * 16, input_shape=noise_shape))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Reshape((16, 16, 128)))
        
        # Upsample to 32x32
        model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization(momentum=0.8))
        
        # Upsample to 64x64
        model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization(momentum=0.8))
        
        # Final output layer - RGB image with pixel values in [-1, 1]
        model.add(layers.Conv2D(self.img_shape[2], (3, 3), padding='same', activation='tanh'))
        
        print("Generator Model Summary:")
        model.summary()
        
        # Noise input
        noise = layers.Input(shape=noise_shape)
        # Generate image from noise
        img = model(noise)
        
        return models.Model(noise, img)
    
    def build_discriminator(self):
        """
        Build the discriminator network.
        
        Returns:
        Model: The discriminator model
        """
        model = models.Sequential(name="Discriminator")
        
        # First convolutional layer
        model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', 
                               input_shape=self.img_shape))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        
        # Second convolutional layer
        model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(layers.BatchNormalization(momentum=0.8))
        
        # Third convolutional layer
        model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        model.add(layers.BatchNormalization(momentum=0.8))
        
        # Fourth convolutional layer
        model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.25))
        
        # Flatten and output layer
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        
        print("Discriminator Model Summary:")
        model.summary()
        
        # Image input
        img = layers.Input(shape=self.img_shape)
        # Validity output
        validity = model(img)
        
        return models.Model(img, validity)
    
    def train(self, dataset, epochs, batch_size=32, save_interval=50, save_dir='generated_images'):
        """
        Train the GAN using the provided dataset.
        
        Parameters:
        dataset (numpy.ndarray): Array of training images
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        save_interval (int): Interval at which to save generated images
        save_dir (str): Directory to save generated images
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Ground truths for adversarial loss
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        # Store loss values for plotting
        g_losses = []
        d_losses = []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Select a random batch of images
            idx = np.random.randint(0, dataset.shape[0], batch_size)
            real_imgs = dataset[idx]
            
            # Generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise, verbose=0)
            
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            # Generate new noise for generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid)
            
            # Store loss values
            g_losses.append(g_loss)
            d_losses.append(d_loss[0])
            
            # Print the progress
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}] - {elapsed_time:.2f} sec")
            
            # If at save interval, save generated image samples
            if (epoch + 1) % save_interval == 0:
                self.save_imgs(epoch + 1, save_dir)
                
                # Also save the model
                self.save_models(save_dir, epoch + 1)
        
        # Plot the loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(g_losses, label='Generator Loss')
        plt.plot(d_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('GAN Training Loss')
        plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
        plt.close()
        
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        print(f"Final discriminator accuracy: {100*d_loss[1]:.2f}%")
    
    def save_imgs(self, epoch, save_dir):
        """
        Save generated images at a specific epoch.
        
        Parameters:
        epoch (int): Current epoch number
        save_dir (str): Directory to save the images
        """
        r, c = 4, 4  # rows, columns
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise, verbose=0)
        
        # Rescale images from [-1, 1] to [0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        fig, axs = plt.subplots(r, c, figsize=(12, 12))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].axis('off')
                cnt += 1
        
        fig.suptitle(f"Abstract Art (Epoch {epoch})", fontsize=16)
        fig.savefig(os.path.join(save_dir, f"abstract_art_epoch_{epoch}.png"))
        plt.close()
        
        # Save individual images as well
        for i, img in enumerate(gen_imgs):
            # Convert from [0,1] to [0,255] and to uint8
            img_array = (img * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_array)
            img_pil.save(os.path.join(save_dir, f"epoch_{epoch}_img_{i}.png"))
    
    def save_models(self, save_dir, epoch):
        """
        Save the generator and discriminator models.
        
        Parameters:
        save_dir (str): Directory to save the models
        epoch (int): Current epoch number
        """
        model_dir = os.path.join(save_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the generator
        self.generator.save(os.path.join(model_dir, f'generator_epoch_{epoch}.h5'))
        
        # Save the discriminator
        self.discriminator.save(os.path.join(model_dir, f'discriminator_epoch_{epoch}.h5'))
    
    def generate_images(self, num_images, save_dir='generated_art'):
        """
        Generate and save a specified number of images.
        
        Parameters:
        num_images (int): Number of images to generate
        save_dir (str): Directory to save the generated images
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Generating {num_images} abstract art images...")
        
        # Generate noise for image generation
        noise = np.random.normal(0, 1, (num_images, self.latent_dim))
        
        # Generate images
        gen_imgs = self.generator.predict(noise, verbose=1)
        
        # Rescale images from [-1, 1] to [0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        # Save the images
        for i, img in enumerate(gen_imgs):
            # Convert from [0,1] to [0,255] and to uint8
            img_array = (img * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_array)
            img_pil.save(os.path.join(save_dir, f"abstract_art_{i+1}.png"))
        
        print(f"Successfully generated {num_images} images in '{save_dir}'")
    
    def load_models(self, generator_path, discriminator_path=None):
        """
        Load saved models.
        
        Parameters:
        generator_path (str): Path to the saved generator model
        discriminator_path (str): Path to the saved discriminator model (optional)
        """
        print(f"Loading generator from {generator_path}")
        self.generator = models.load_model(generator_path)
        
        if discriminator_path:
            print(f"Loading discriminator from {discriminator_path}")
            self.discriminator = models.load_model(discriminator_path)
            
            # Recreate the combined model
            z = layers.Input(shape=(self.latent_dim,))
            img = self.generator(z)
            self.discriminator.trainable = False
            validity = self.discriminator(img)
            self.combined = models.Model(z, validity)
            self.combined.compile(
                loss='binary_crossentropy',
                optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
            )
        
        print("Models loaded successfully")

def load_and_preprocess_images(image_dir, img_shape):
    """
    Load and preprocess images from a directory.
    
    Parameters:
    image_dir (str): Directory containing image files
    img_shape (tuple): Target shape for the images (height, width, channels)
    
    Returns:
    numpy.ndarray: Array of preprocessed images
    """
    print(f"Loading images from {image_dir}...")
    
    # Get all image files from the directory
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if os.path.splitext(f.lower())[1] in valid_extensions
    ]
    
    if not image_files:
        raise ValueError(f"No valid image files found in {image_dir}")
    
    print(f"Found {len(image_files)} image files")
    
    # Load and preprocess images
    images = []
    for img_path in image_files:
        try:
            # Load image
            img = Image.open(img_path)
            
            # Resize to target shape
            img = img.resize((img_shape[1], img_shape[0]), Image.LANCZOS)
            
            # Convert to RGB if it's not
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Normalize to [-1, 1]
            img_array = (img_array.astype(np.float32) - 127.5) / 127.5
            
            images.append(img_array)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Convert to numpy array
    images = np.array(images)
    
    print(f"Loaded and preprocessed {len(images)} images")
    print(f"Image array shape: {images.shape}")
    
    return images

def main():
    """
    Main function to run the Abstract Art GAN.
    """
    parser = argparse.ArgumentParser(description='Abstract Art Generation with GANs')
    
    parser.add_argument('--mode', choices=['train', 'generate'], default='train',
                       help='Mode: train a new model or generate images from a trained model')
    
    parser.add_argument('--image-dir', type=str, default='Abstract_gallery',
                       help='Directory containing training images (for train mode)')
    
    parser.add_argument('--img-height', type=int, default=64,
                       help='Height of the images')
    
    parser.add_argument('--img-width', type=int, default=64,
                       help='Width of the images')
    
    parser.add_argument('--channels', type=int, default=3,
                       help='Number of image channels (3 for RGB, 1 for grayscale)')
    
    parser.add_argument('--latent-dim', type=int, default=100,
                       help='Dimension of the latent space')
    
    parser.add_argument('--epochs', type=int, default=10000,
                       help='Number of training epochs')
    
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Interval at which to save generated images during training')
    
    parser.add_argument('--output-dir', type=str, default='abstract_art_output',
                       help='Directory to save outputs')
    
    parser.add_argument('--generator-model', type=str, default=None,
                       help='Path to saved generator model (for generate mode)')
    
    parser.add_argument('--num-images', type=int, default=16,
                       help='Number of images to generate (for generate mode)')
    
    args = parser.parse_args()
    
    # Set image shape
    img_shape = (args.img_height, args.img_width, args.channels)
    
    # Create the GAN
    gan = AbstractArtGAN(img_shape=img_shape, latent_dim=args.latent_dim)
    
    if args.mode == 'train':
        # Load and preprocess images
        images = load_and_preprocess_images(args.image_dir, img_shape)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Train the GAN
        gan.train(
            dataset=images,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            save_dir=args.output_dir
        )
        
    elif args.mode == 'generate':
        if args.generator_model is None:
            # Try to find the latest generator model in the output directory
            model_dir = os.path.join(args.output_dir, 'models')
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.startswith('generator_')]
                if model_files:
                    # Get the latest model
                    latest_model = sorted(model_files)[-1]
                    args.generator_model = os.path.join(model_dir, latest_model)
                    print(f"Using latest generator model: {args.generator_model}")
            
            if args.generator_model is None:
                raise ValueError("No generator model specified and could not find one in the output directory.")
        
        # Load the saved model
        gan.load_models(args.generator_model)
        
        # Generate images
        gan.generate_images(
            num_images=args.num_images,
            save_dir=os.path.join(args.output_dir, 'final_generated_art')
        )

if __name__ == "__main__":
    main()