import array

import numpy as np
from collections import defaultdict
import mixture_mi
from scipy.stats import wasserstein_distance
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import energy_distance



def auto_cluster_by_threshold(distance_matrix, percentile):
    d = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    threshold = np.percentile(d, percentile)

    return threshold


def merge2groups(groups, labels):
    max_len = 0
    new_groups = defaultdict(list)
    for g in groups:
        new_groups[labels[g]] += groups[g]

    return new_groups

def NormalizedClusteredMI(discs, conts, percentile=50):
    groups = {}

    event_size = len(set(discs))
    # print('Event size: ', event_size)
    # num_clusters = min(num_clusters, event_size)
    # print("Number of clusters: ", num_clusters)
    # print("Number of events: ", event_size)

    # if num_clusters < event_size:

    size = len(discs)
    event2id = defaultdict(int)
    ids = 0
    for i in range(size):
        if discs[i] not in event2id:
            event2id[discs[i]] = ids
            ids += 1
        label = event2id[discs[i]]
        if label not in groups:
            groups[label] = []
        groups[label].append(conts[i])

    n = len(groups)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i, j] = energy_distance(np.asarray(groups[i]), np.asarray(groups[j]))


    threshold = auto_cluster_by_threshold(distance_matrix, percentile=percentile)

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric='precomputed', linkage='complete')
    labels = clustering.fit_predict(distance_matrix)
    # print(labels)

    new_groups = merge2groups(groups, labels)

    new_discs = []
    new_conts = []
    for g in new_groups:
        for val in new_groups[g]:
            new_discs.append(g)
            new_conts.append(val)

    cnmi = mixture_mi.NMI(new_discs, new_conts)

    return cnmi