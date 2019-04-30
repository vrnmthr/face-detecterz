import glob

import numpy as np
import plotly
from sklearn.decomposition import PCA
from hypersphere import to_spherical, to_rectangular

data_map = {}

embeddings = []
labels = []

# loads embeddings out of embeddings/test
# ix = 0
# for face in glob.glob("embeddings/test/*"):
#     paths = glob.glob(os.path.join(face, "*.npy"))
#     for p in paths:
#         e = np.load(p)
#         embeddings.append(e)
#         labels.append(ix)
#     ix += 1
# embeddings = np.asarray(embeddings)


ix = 0
n = 10
for face in glob.glob("embeddings/test3/*.npy")[:n]:
    samples = np.load(face)
    embeddings.append(samples)
    labels.append(np.full(len(samples), ix))
    ix += 1
embeddings = np.concatenate(embeddings)
embeddings = to_spherical(embeddings)
labels = np.concatenate(labels)

pca = PCA(n_components=3)
reduced = pca.fit_transform(embeddings)
print(np.cumsum(pca.explained_variance_ratio_))

plot = plotly.graph_objs.Scatter(
    x=reduced[:, 0],
    y=reduced[:, 1],
    mode="markers",
    marker=dict(
        color=labels,  # set color equal to a variable
        colorscale='Rainbow',
    )
)

threeplot = plotly.graph_objs.Scatter3d(
    x=reduced[:, 0],
    y=reduced[:, 1],
    z=reduced[:, 2],
    mode="markers",
    marker=dict(
        color=labels,  # set color equal to a variable
        colorscale='Rainbow',
    )
)

plotly.offline.plot([plot], filename="embeddings.html")
plotly.offline.plot([threeplot], filename="embeddings3d.html")
