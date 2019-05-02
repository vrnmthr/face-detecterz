import numpy as np
import plotly
from sklearn import svm
from tqdm import tqdm
import dlib

from classifier import train, test
from dataset import FaceDataset
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist
"""
Plot ROC curve by plotting different cutoff values
"""


def chinese_whispers():
    print("Getting datasets")
    known = FaceDataset("embeddings/known", n=853)
    known_train, known_labels = known.train()

    dlib_arrays = []
    for sample in known_train:
        dlib_arrays.append(dlib.array(sample))

    labels = dlib.chinese_whispers_clustering(dlib_arrays, 0.5)
    num_classes = len(set(labels))
    print("Number of clusters: {}".format(num_classes))

    # Find biggest class
    biggest_class = None
    biggest_class_length = 0
    for i in range(0, num_classes):
        class_length = len([label for label in labels if label == i])
        if class_length > biggest_class_length:
            biggest_class_length = class_length
            biggest_class = i

    print("Biggest cluster id number: {}".format(biggest_class))
    print("Number of faces in biggest cluster: {}".format(biggest_class_length))


def euclidean_centroid_roc():
    print("Getting datasets")
    known = FaceDataset("embeddings/known", n=853)
    known_train, known_labels = known.train()
    known_data = []
    for i in range(853):
        centroid = np.mean(known_train[known_labels == i], axis=0)
        known_data.append(centroid)
    known_test, _ = known.test()

    unknown = FaceDataset("embeddings/unknown", n=500)
    unknown_data, _ = unknown.all()

    print("Calculating distances...")
    known_dists = cdist(known_test, known_data, metric="cosine")
    known_dists = np.min(known_dists, axis=1)
    unknown_dists = cdist(unknown_data, known_data, metric="cosine")
    unknown_dists = np.min(unknown_dists, axis=1)

    # TPR = rate of unknown faces correctly qualified as so
    # FPR = rate of known faces being qualified as unknown faces
    print("Calculating curve...")
    TPRs = []
    FPRs = []
    thresholds = np.linspace(0, 2, 1000)
    for t in thresholds:
        TPR = np.mean(unknown_dists > t)
        FPR = np.mean(known_dists > t)
        TPRs.append(TPR)
        FPRs.append(FPR)

    roc = plotly.graph_objs.Scatter(x=FPRs, y=TPRs, text=thresholds)
    layout = plotly.graph_objs.Layout(
        title='Euclidean Distance ROC curve',
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
    )
    fig = plotly.graph_objs.Figure(data=[roc], layout=layout)

    plotly.offline.plot(fig, filename="svm_roc.html")


def svm_roc():
    data = FaceDataset("embeddings/known", n=100)
    train_data, train_labels = data.train()

    clf = svm.SVC(kernel="linear", C=1.6, probability=True)
    clf, _ = train(clf, train_data, train_labels)

    unknown = FaceDataset("embeddings/unknown", n=100)
    unknown_data, _ = unknown.train()
    probs = clf.predict_proba(unknown_data)
    probs_max = np.max(probs, axis=1)

    thresholds = np.linspace(0, 1, 100)
    false_positives = []
    for t in thresholds:
        false = np.mean(probs_max > t)
        false_positives.append(false)

    true_positives = np.subtract(1, false_positives)
    roc = plotly.graph_objs.Scatter(x=false_positives, y=true_positives)
    plotly.offline.plot([roc], filename="svm_roc.html")


if __name__ == '__main__':
    chinese_whispers()
