from transformers import BertModel
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model = BertModel.from_pretrained('bert-base-uncased')
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
embeddings = outputs.last_hidden_state[0].detach().numpy()  # Token embeddings

# Reduce dimensions with t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
for i, token in enumerate(tokens[:10]):  # Plot first 10 tokens
    plt.annotate(token, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
plt.show()