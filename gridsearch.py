import numpy as np
import plotly
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from classifier import train, test
from dataset import FaceDataset


def find_svm_hyperparams():
    """
    Finds best SVM hyperparams based on evaluating on test split
    """
    data = FaceDataset("embeddings/known", n=100)
    train_data, train_labels = data.train()
    test_data, test_labels = data.test()

    gammas = np.linspace(0.01, 10, 10)
    accs = []
    for gamma in tqdm(gammas):
        clf = svm.SVC(kernel="linear", gamma=gamma)
        clf, _ = train(clf, train_data, train_labels)
        acc, _ = test(clf, test_data, test_labels)
        accs.append(acc)

    s = plotly.graph_objs.Scatter(x=gammas, y=accs)
    plotly.offline.plot([s], filename="svm_grid.html")


def find_linear_svm_hyperparams():
    """
    Fits a linear SVM model
    """
    Cs = np.linspace(0.5, 2, 10)
    results = []

    for _ in range(10):
        data = FaceDataset("embeddings/known", n=100)
        train_data, train_labels = data.train()
        test_data, test_labels = data.test()
        accs = []
        for c in tqdm(Cs):
            clf = svm.SVC(kernel="linear", C=c)
            clf, _ = train(clf, train_data, train_labels)
            acc, _ = test(clf, test_data, test_labels)
            accs.append(acc)
        results.append(accs)

    results = np.mean(results, axis=0)
    s = plotly.graph_objs.Scatter(x=Cs, y=results)
    plotly.offline.plot([s], filename="svm_linear.html")
    print("C={}".format(Cs[np.argmax(results)]))


def find_knn_hyperparams():
    """
    Finds best KNN hyperparams
    """
    n_neighbors = np.arange(5, 10)
    ps = np.arange(1, 10)
    results = []

    for p in ps:
        result = []
        for _ in range(10):
            data = FaceDataset("embeddings/known", n=100)
            train_data, train_labels = data.train()
            test_data, test_labels = data.test()
            accs = []
            for n in n_neighbors:
                clf = KNeighborsClassifier(n_neighbors=n, weights="distance", p=p)
                clf, _ = train(clf, train_data, train_labels)
                acc, _ = test(clf, test_data, test_labels)
                accs.append(acc)
            result.append(accs)
        result = np.mean(result, axis=0)
        results.append(result)

    plots = []
    for i in range(len(ps)):
        p = plotly.graph_objs.Scatter(x=n_neighbors, y=results[i], name="p={}".format(ps[i]))
        plots.append(p)

    plotly.offline.plot(plots, filename="knn.html")
    print("C={}".format(n_neighbors[np.argmax(results)]))


if __name__ == '__main__':
    find_knn_hyperparams()
