from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np
    import sys
    np.set_printoptions(threshold=sys.maxsize)

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    
    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    
    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    
    from sklearn.manifold import TSNE
    from IPython import embed
except ImportError as e:
    print(e)
    raise ImportError

def draw(enc, labels, figname, n_c=10, points=None):
    tsne = TSNE(n_components=2, verbose=1, init='pca', random_state=0)
    # Cluster with TSNE
    if points is None:
        tsne_enc = tsne.fit_transform(enc)
    else:
        tsne_enc = points

    # Convert to numpy for indexing purposes
    # labels = labels.cpu().data.numpy()

    # Color and marker for each true class
    colors = cm.rainbow(np.linspace(0, 1, n_c))
    markers = matplotlib.markers.MarkerStyle.filled_markers

    # Save TSNE figure to file
    fig, ax = plt.subplots(figsize=(16,10))
    for iclass in range(n_c):
        # Get indices for each class
        idxs = labels==iclass
        # Scatter those points in tsne dims
        ax.scatter(tsne_enc[idxs, 0],
                   tsne_enc[idxs, 1],
                   marker=markers[iclass],
                   c=colors[iclass],
                   edgecolor=None,
                   label=r'$%i$'%iclass)
        center = tsne_enc[idxs].mean(0)
        plt.text(center[0], center[1]-3, str(iclass), fontsize=30)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    fig.savefig(figname)
    return tsne_enc

