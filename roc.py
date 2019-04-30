import numpy as np
import plotly
from sklearn import svm
from tqdm import tqdm

from classifier import train, test
from dataset import FaceDataset
from sklearn.neighbors import KNeighborsClassifier

"""
Plot ROC curve by plotting different cutoff values
"""


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
    svm_roc()
