import array

import numpy as np
from collections import defaultdict
import mixture_mi
from scipy.stats import wasserstein_distance
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import energy_distance



def getClusterNum(distance_matrix, max_clusters):
    from kneed import KneeLocator
    def calculate_cluster_average_distances(distance_matrix, max_clusters):
        avg_distances = []
        for k in range(2, max_clusters + 1):
            clustering = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
            labels = clustering.fit_predict(distance_matrix)


            total_distance = 0
            cluster_counts = 0
            for cluster_id in np.unique(labels):
                cluster_points = np.where(labels == cluster_id)[0]
                if len(cluster_points) > 1:
                    cluster_distances = distance_matrix[np.ix_(cluster_points, cluster_points)]
                    total_distance += np.sum(cluster_distances) / 2
                    cluster_counts += len(cluster_points) * (len(cluster_points) - 1) / 2

            avg_distances.append(total_distance / cluster_counts if cluster_counts > 0 else 0)
        return avg_distances

    # 计算簇内平均距离
    avg_distances = calculate_cluster_average_distances(distance_matrix, max_clusters)

    # 自动识别肘部点
    knee_locator = KneeLocator(range(3, max_clusters + 1), avg_distances, curve='convex', direction='decreasing')
    best_k = knee_locator.knee
    print("Best k is: ", best_k)
    return best_k

def merge2groups(groups, labels):
    max_len = 0
    new_groups = defaultdict(list)
    for g in groups:
        new_groups[labels[g]] += groups[g]

    return new_groups

def NormalizedClusteredMI(discs, conts, num_clusters):
    groups = {}

    event_size = len(set(discs))
    # print('Event size: ', event_size)
    num_clusters = min(num_clusters, event_size)
    # print("Number of clusters: ", num_clusters)
    # print("Number of events: ", event_size)

    if num_clusters < event_size:

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

        # best_k = getClusterNum(distance_matrix, event_size)
        # print(distance_matrix)

        clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='precomputed', linkage='complete')
        labels = clustering.fit_predict(distance_matrix)
        # print(labels)

        new_groups = merge2groups(groups, labels)

        new_discs = []
        new_conts = []
        for g in new_groups:
            for val in new_groups[g]:
                new_discs.append(g)
                new_conts.append(val)
    else:
        new_discs = discs
        new_conts = conts
    cnmi = mixture_mi.NMI(new_discs, new_conts)

    return cnmi