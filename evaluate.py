from sklearn import metrics
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from IPython import embed
import random
import numpy as np
from tqdm import tqdm


def cluster(vectors, init='k-means++'):
    kmeans = KMeans(n_clusters=10, random_state=0, init=init).fit(vectors)
    return kmeans.labels_

def get_centers(vectors):
    kmeans = KMeans(n_clusters=10, random_state=0).fit(vectors)
    return kmeans.cluster_centers_

def calacc(labels_true, labels_pred):
    assert len(labels_true)==len(labels_pred)
    Cpair = Counter(list(zip(labels_true, labels_pred)))
    s = set(labels_pred)
    mp = {}
    for l in s:
        mp[l] = 0
    for lt, lp in Cpair:
        mp[lp] = max(mp[lp], Cpair[(lt, lp)])
    return sum(mp.values())/len(labels_true)

def evaluate(labels_true, labels_pred):
    ari = metrics.adjusted_rand_score(labels_true, labels_pred)
    nmi = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    ami = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    acc = calacc(labels_true, labels_pred)
    return {'ARI':ari, 'ACC':acc, 'NMI':nmi, 'AMI':ami}
