import torch
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE

normal_embed_weights = torch.load('./embed/exhird_h_normal_word_embed_cpu.pt')

normal_embed_weights = normal_embed_weights.numpy()
t_sne = TSNE(n_components=2, verbose=2)
tsne_normal_embed = t_sne.fit_transform(normal_embed_weights)

plt.scatter(tsne_normal_embed[:0], tsne_normal_embed[:1])
plt.savefig('normal_embed_figure.png')
