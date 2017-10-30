from Cluster.KMeans import KMeans
import matplotlib.pyplot as plt


def generate_faux_data(n_data_points=5000, centers=[[6, 2], [4, 2], [2, 3], [5, 4]], noise=1.5):
    data = []
    for i in range(n_data_points):
        a, b = centers[np.random.choice(len(centers))]
        data.append(
            [a + noise*np.random.normal(), b + noise*np.random.normal()])
    return np.array(data)

X = generate_faux_data()

plt.figure()
for i in range(2, 10):
    k = KMeans(i, max_iter=300, random_state=None)
    k.fit(X)

    plt.plot(i, k.inertia, 'or')
plt.show()