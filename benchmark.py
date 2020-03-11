import time
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pynanoflann
from contexttimer import Timer
from sklearn import neighbors


n_index_points = 200000
n_query_points = 1000
n_repititions = 5
data_dim = 3
n_neighbors = 100
index_type = np.float32

data = np.random.uniform(0, 100, size=(n_index_points, data_dim)).astype(index_type)
queries = np.random.uniform(0, 100, size=(n_query_points, data_dim)).astype(index_type)

algs = {
    'sklearn_brute': neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute'),
    'sklearn_ball_tree': neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree'),
    'sklearn_kd_tree': neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree'),
    'pynanoflann': pynanoflann.KDTree(n_neighbors=n_neighbors),
}

results = []
for rep in range(n_repititions):
    for alg_name, nn in algs.items():

        with Timer() as index_build_time:
            nn.fit(data)

        with Timer() as query_time:
            dist, idx = nn.kneighbors(queries)

        results.append((alg_name, index_build_time.elapsed, query_time.elapsed))


df = pd.DataFrame(results, columns=['Algorithm', 'Index build time, second', 'Query time, second'])
print(df)

fig, ax = plt.subplots(figsize=(18, 6))
sns.barplot(data=df, x='Algorithm', y=df.columns[2], ax=ax, ci=None)
ax.set_yscale("log", basey=4)
ylabels = ['{:.4f}'.format(x) for x in ax.get_yticks()]
ax.set_yticklabels(ylabels)
ax.set_title(f'n_index_points={n_index_points}, n_query_points={n_query_points}, dim={data_dim}')
plt.grid()
plt.savefig('benchmark_query.png')

fig, ax = plt.subplots(figsize=(18, 6))
sns.barplot(data=df[df.Algorithm != 'sklearn_brute'], x='Algorithm', y=df.columns[1], ax=ax, palette=['C1', 'C2', 'C3'], ci=None)
ax.set_title(f'n_index_points={n_index_points}, dim={data_dim}')
plt.grid()
plt.savefig('benchmark_index.png')

plt.show()
