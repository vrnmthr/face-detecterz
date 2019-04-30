import time

import numpy as np
import plotly
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier, NearestCentroid

from dataset import FaceDataset
from tqdm import tqdm


def train(clf, data, labels):
    """
    :return: trained classifier, training time
    """
    start = time.time()
    clf.fit(data, labels)
    duration = time.time() - start
    return clf, duration


def test(clf, data, labels):
    """
    :return: accuracy percentage, evaluation time
    """
    # TODO: this right now only evaluates over a single test/train split. Ideally we would do cross-validation?
    start = time.time()
    pred = clf.predict(data)
    duration = time.time() - start
    correct = pred == labels
    return np.mean(correct), duration


def make_graphs(dir, clfs):
    """
    Plots accuracy vs. number of classes, training time vs. number of classes and eval time vs. number of classes
    """
    # TODO: we actually want time taken to "add" a class instead of "training time"; these two are not necessarily equal
    num_classes = [2, 10, 50, 100]
    result_shape = (len(clfs), len(num_classes))
    metrics = {
        "accs": np.empty(result_shape),
        "train_ts": np.empty(result_shape),
        "test_ts": np.empty(result_shape),
    }
    datasets = [FaceDataset(dir, n=i) for i in num_classes]

    for k, name in tqdm(enumerate(clfs), total=len(clfs)):
        for i, dataset in enumerate(datasets):
            test_data, test_labels = dataset.test()
            train_data, train_labels = dataset.train()
            clf = clfs[name]
            clf, train_t = train(clf, train_data, train_labels)
            acc, test_t = test(clf, test_data, test_labels)
            metrics["accs"][k, i] = acc
            metrics["train_ts"][k, i] = train_t
            metrics["test_ts"][k, i] = test_t

    for metric in metrics:
        plots = []
        for k, name in enumerate(clfs):
            plot = plotly.graph_objs.Scatter(x=num_classes, y=metrics[metric][k], name=name)
            plots.append(plot)
        plotly.offline.plot(plots, filename="{}.html".format(metric))


if __name__ == '__main__':
    path = "embeddings/test3"
    # TODO: all sorts of grid searches need to be done over these to actually determine what the right hyperparams are
    # I've just sorta randomly initialized them for now with what I think would work best
    clfs = {
        "svm": svm.SVC(gamma="scale", kernel="rbf"),
        "knn": KNeighborsClassifier(weights="distance"),
        # "gaussian process": GaussianProcessClassifier(),
        "random forest": RandomForestClassifier(),
        "radius neighbors": RadiusNeighborsClassifier(weights="distance"),
        "nearest centroid": NearestCentroid(),
        # "gradient boost": GradientBoostingClassifier()
    }
    make_graphs(path, clfs)
