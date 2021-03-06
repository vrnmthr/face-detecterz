import time

import numpy as np
import plotly
import torch
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from tqdm import tqdm

from classifiers.binary_face_classifier import BinaryFaceClassifier
from dataset import FaceDataset


def train(clf, data, labels):
    """
    :return: trained classifier, training time
    """
    start = time.time()
    # data = to_spherical(data)
    clf.fit(data, labels)
    duration = time.time() - start
    return clf, duration


def test(clf, data, labels):
    """
    :return: accuracy percentage, evaluation time
    """
    # TODO: this right now only evaluates over a single test/train split. Ideally we would do cross-validation?
    start = time.time()
    # data = to_spherical(data)
    pred = clf.predict(data)
    duration = time.time() - start
    correct = pred == labels
    return np.mean(correct), duration


def make_graphs(dir, clfs):
    """
    Plots accuracy vs. number of classes, training time vs. number of classes and eval time vs. number of classes
    """
    # TODO: we actually want time taken to "add" a class instead of "training time"; these two are not necessarily equal
    mlp2_out_acc = []
    mlp2_out_train = []
    mlp2_out_test = []
    mlp3_out_acc = []
    mlp3_out_train = []
    mlp3_out_test = []
    num_classes = [5, 10, 25, 100]
    result_shape = (len(clfs), len(num_classes))
    metrics = {
        "accs": np.empty(result_shape),
        "train_ts": np.empty(result_shape),
        "test_ts": np.empty(result_shape),
    }

    for k, name in tqdm(enumerate(clfs), total=len(clfs)):
        for i, dataset in enumerate(datasets):
            test_data, test_labels = dataset.test()
            train_data, train_labels = dataset.train()
            clf = clfs[name]
            if name == "2-Layer Multi-Perceptron":
                clf = Simple2MLP(numClasses=num_classes[i])
            if name == "3-Layer Multi-Perceptron":
                clf = Simple3MLP(numClasses=num_classes[i])
            clf, train_t = train(clf, train_data, train_labels)
            acc, test_t = test(clf, test_data, test_labels)
            metrics["accs"][k, i] = acc
            metrics["train_ts"][k, i] = train_t
            metrics["test_ts"][k, i] = test_t
    for metric in metrics:
        plots = []
        for k, name in enumerate(clfs):
            result = metrics[metric][k]
            result = np.mean(result, axis=1)
            plot = plotly.graph_objs.Scatter(x=num_classes, y=result, name=name)
            plots.append(plot)
        plotly.offline.plot(plots, filename="{}.html".format(metric))


if __name__ == '__main__':
    path = "embeddings/train"
    device = torch.device("cpu")
    clfs = {
        "linear svm": svm.SVC(kernel="linear", gamma="scale", C=1.6),
        "ridge": RidgeClassifier(alpha=2 ** -10, solver="lsqr"),
        "logistic": LogisticRegression(solver="lbfgs", multi_class="auto", C=18, max_iter=1000),
        "knn": KNeighborsClassifier(weights="distance", n_neighbors=8),
        "random forest": RandomForestClassifier(n_estimators=100),
        "nearest centroid": NearestCentroid(),
        # "qda": QuadraticDiscriminantAnalysis(),
        "lda": LinearDiscriminantAnalysis(),
        "binary-nn": BinaryFaceClassifier("data/binary_face_detector.pt", device)
    }
    make_graphs(path, clfs)
