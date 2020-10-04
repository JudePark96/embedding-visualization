import torch
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE

gcn_embed_weights = torch.load('./embed/exhird_h_normal_gcn_word_embed_cpu.pt')

gcn_embed_weights = gcn_embed_weights.numpy()
t_sne = TSNE(n_components=2, verbose=2)
tsne_gcn_embed = t_sne.fit_transform(gcn_embed_weights)

plt.scatter(tsne_gcn_embed[:0], tsne_gcn_embed[:1])
plt.savefig('gcn_embed_figure.png')
