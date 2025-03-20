import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Generator
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(256, input_dim=100, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(64 * 64, activation='tanh'),
        layers.Reshape((64, 64, 1))
    ])
    return model

# Discriminator
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(64, 64, 1)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Load and preprocess image data (assuming grayscale for simplicity)
def load_images():
    images = []  # Load your preprocessed images here
    for filename in os.listdir('data/artwork_processed/'):
        img = Image.open(f'data/artwork_processed/{filename}').convert('L')
        images.append(np.array(img) / 127.5 - 1)  # Normalize to [-1, 1]
    return np.array(images)

# Training loop (simplified)
generator = build_generator()
discriminator = build_discriminator()
gan = tf.keras.Sequential([generator, discriminator])
discriminator.compile(optimizer='adam', loss='binary_crossentropy')
gan.compile(optimizer='adam', loss='binary_crossentropy')

images = load_images()
for epoch in range(1000):  # Adjust epochs as needed
    noise = np.random.normal(0, 1, (32, 100))  # Batch size 32
    fake_images = generator.predict(noise)
    real_images = images[np.random.randint(0, images.shape[0], 32)]
    X = np.concatenate([real_images, fake_images])
    y = np.concatenate([np.ones((32, 1)), np.zeros((32, 1))])
    discriminator.train_on_batch(X, y)
    noise = np.random.normal(0, 1, (32, 100))
    gan.train_on_batch(noise, np.ones((32, 1)))

# Save models
generator.save('models/generator.h5')
