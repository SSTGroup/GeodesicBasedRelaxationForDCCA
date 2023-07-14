import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


def kmeans_clustering_acc(data_points, labels, num_classes):
    """
    Compute a kmeans clustering of the data_points and assign labels via majority voting
    """

    clust_labels = KMeans(
        n_clusters=num_classes,
        random_state=33,
        n_init=5).fit_predict(data_points)

    prediction = np.zeros_like(clust_labels)
    for i in range(num_classes):
        ids = np.where(clust_labels == i)[0]
        prediction[ids] = np.argmax(np.bincount(labels[ids]))

    return accuracy_score(labels, prediction)