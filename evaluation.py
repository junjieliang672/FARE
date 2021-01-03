import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import *
import sklearn.metrics as metrics


def criteria_n_cluster(trainX, testX, ground_truth, n_clusters=np.arange(4,20)):
    print(f'begin clustering, trying clusters {n_clusters}')
    ground_truth = np.array(ground_truth)
    best_cluster = None
    best_sih = None
    best_ami = None
    best_acc = None
    for n_cluster in n_clusters:
        km = KMeans(n_clusters=n_cluster, random_state=0)
        km.fit(trainX)
        label = km.predict(testX)
        ami, acc, sih = criteriaWithoutClustering(label,ground_truth,testX if len(n_clusters)>1 else None)
        if len(n_clusters)>1:
            print(f'{n_cluster} => {sih}')
            if best_cluster is None:
                best_cluster = n_cluster
                best_sih = sih
                best_ami = ami
                best_acc = acc
                print(f'init cluster {n_cluster}')
            elif best_sih < sih+0.01:
                print(f'find a better silhouette, now the best cluster is {n_cluster}')
                best_cluster = n_cluster
                best_sih = sih
                best_ami = ami
                best_acc = acc
        else:
            best_cluster = n_cluster
            best_sih = None
            best_ami = ami
            best_acc = acc
        print(f'the best cluster number is {best_cluster}, best silhouette is {best_sih}')
    return best_ami, best_acc,best_sih,best_cluster


def compute_ami(trainX, testX, ground_truth, n_cluster=6):
    ground_truth = np.array(ground_truth)
    km = KMeans(n_clusters=n_cluster, random_state=0)
    km.fit(trainX)
    label = km.predict(testX)
    return partial_ami(ground_truth,label)


def criteriaWithoutClustering(label, ground_truth,X=None):
    ami = partial_ami(ground_truth, label)
    acc = partial_acc(ground_truth, label)
    sih = None
    if X is not None:
        if len(set(label)) > 1:
            sih = silhouette_score(X,label)
    return ami, acc, sih


def compute_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size


def partial_acc(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)[1:,:]
    rowmax = np.amax(contingency_matrix,axis=0)
    return np.sum(rowmax) / np.sum(contingency_matrix)

def partial_ami(y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    keep = y_true >= 0
    y_true = y_true[keep]
    y_pred = y_pred[keep]
    return adjusted_mutual_info_score(y_true,y_pred,average_method='arithmetic')