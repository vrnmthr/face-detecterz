import numpy as np
import glob
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_map = {}
count = 0
for np_name in glob.glob('/Users/Eleanor/Desktop/aligned_embeddings/*.npy'):
    data_map[count] = np.load(np_name)
    count += 1

data = np.concatenate(list(data_map.values()))

pca = PCA(n_components=3)

data_transform = pca.fit_transform(data)

print(data_transform)
print(data_transform.shape)
print(np.sum(pca.explained_variance_ratio_))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = data_transform[:, 0]
y = data_transform[:, 1]
z = data_transform[:, 2]

ax.scatter(xs=x, ys=y, zs=z)
plt.show()
print("showed")
