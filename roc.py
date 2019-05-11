import dlib
import numpy as np
import plotly
import torch
from scipy.spatial.distance import cdist
from sklearn import svm
from tqdm import tqdm

from classifiers.binary_face_classifier import BinaryFaceClassifier
from dataset import FaceDataset

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
    known = FaceDataset("embeddings/test")
    known_train, known_labels = known.all()
    known_data = []
    for i in range(1000):
        centroid = np.mean(known_train[known_labels == i], axis=0)
        known_data.append(centroid)
    known_test, _ = known.test()

    unknown = FaceDataset("embeddings/dev")
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

    np.save("euc_tpr.npy", TPRs)
    np.save("euc_fpr.npy", FPRs)

    roc = plotly.graph_objs.Scatter(x=FPRs, y=TPRs, text=thresholds)
    layout = plotly.graph_objs.Layout(
        title='Euclidean Distance ROC curve',
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
    )
    fig = plotly.graph_objs.Figure(data=[roc], layout=layout)

    plotly.offline.plot(fig, filename="svm_roc.html")


def svm_unknown_classes():
    N_ITERS = 50
    t = []
    f = []
    for _ in tqdm(range(N_ITERS), total=N_ITERS):

        known = FaceDataset("embeddings/test", n=100)
        known_train, known_labels = known.train()
        known_test, _ = known.test()

        unknown = FaceDataset("embeddings/dev", n=100)
        unknown_data, _ = unknown.all()

        seed = FaceDataset("embeddings/train", n=100)
        seed_train, seed_labels = seed.all()

        # assign our unknown class to be 0 and increment all the labels in known by 1
        known_labels = known_labels + 1
        seed_labels = np.zeros(len(seed_labels))

        # train the SVM on the classes with the random seed
        full_training = np.concatenate([known_train, seed_train])
        full_labels = np.concatenate([known_labels, seed_labels])
        clf = svm.SVC(kernel="linear", gamma="scale", C=1.6, probability=True)
        clf.fit(full_training, full_labels)

        # run SVM on the unknown set
        unknown_probs = clf.predict_proba(unknown_data)
        pred = np.argmax(unknown_probs, axis=1)
        unknown_confs = np.max(unknown_probs, axis=1)
        unknown_confs[pred == 0] = 0

        # run SVM on known set
        known_probs = clf.predict_proba(known_test)
        pred = np.argmax(known_probs, axis=1)
        known_confs = np.max(known_probs, axis=1)
        known_confs[pred == 0] = 0

        # TPR = rate of unknown faces correctly qualified as so
        # FPR = rate of known faces being qualified as unknown faces
        TPRs = []
        FPRs = []
        thresholds = np.linspace(0, 1, 1000)
        for x in thresholds:
            TPR = np.mean(unknown_confs < x)
            FPR = np.mean(known_confs < x)
            TPRs.append(TPR)
            FPRs.append(FPR)

        t.append(TPRs)
        f.append(FPRs)

    t = np.mean(t, axis=0)
    f = np.mean(f, axis=0)
    print(t.shape)
    print(f.shape)
    np.save("svm_tpr.npy", t)
    np.save("svm_fpr.npy", f)


def binary_neural_roc():
    N_ITERS = 50
    t = []
    f = []
    clf = BinaryFaceClassifier("data/binary_face_detector.pt", torch.device("cpu"))
    for _ in tqdm(range(N_ITERS), total=N_ITERS):

        known = FaceDataset("embeddings/test", n=100)
        known_train, known_labels = known.train()
        known_test, _ = known.test()

        unknown = FaceDataset("embeddings/dev", n=100)
        unknown_data, _ = unknown.all()

        # train the classifier
        clf.fit(known_train, known_labels)

        # run on known set
        unknown_probs = clf.predict_proba(unknown_data)
        unknown_confs = np.max(unknown_probs, axis=1)

        # run SVM on unknown set
        known_probs = clf.predict_proba(known_test)
        known_confs = np.max(known_probs, axis=1)

        # TPR = rate of unknown faces correctly qualified as so
        # FPR = rate of known faces being qualified as unknown faces
        TPRs = []
        FPRs = []
        thresholds = np.linspace(0.3, 1, 1000)
        for x in thresholds:
            TPR = np.mean(unknown_confs < x)
            FPR = np.mean(known_confs < x)
            TPRs.append(TPR)
            FPRs.append(FPR)

        t.append(TPRs)
        f.append(FPRs)

    t = np.mean(t, axis=0)
    f = np.mean(f, axis=0)
    print(t.shape)
    print(f.shape)
    np.save("neural_tpr.npy", t)
    np.save("neural_fpr.npy", f)


if __name__ == '__main__':
    classes = ["neural", "svm", "euc"]
    rocs = []
    for alg in classes:
        FPRs = np.load("{}_fpr.npy".format(alg))
        TPRs = np.load("{}_tpr.npy".format(alg))
        roc = plotly.graph_objs.Scatter(x=FPRs, y=TPRs, name=alg)
        rocs.append(roc)

    layout = plotly.graph_objs.Layout(
        title='ROC curves',
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
    )
    fig = plotly.graph_objs.Figure(data=rocs, layout=layout)
    plotly.offline.plot(fig, filename="rocs.html")
