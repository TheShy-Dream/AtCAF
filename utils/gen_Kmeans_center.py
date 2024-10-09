from sklearn.cluster import KMeans
import numpy as np
def gen_npy(data, dataset="mosi",n_clusters=50):  # 求data的聚类
    data = data.detach().numpy()
    print(data.shape)
    reshaped_data = data
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(reshaped_data)
    centers = kmeans.cluster_centers_
    print(centers.shape)
    np.save(f'kmeans_{dataset}-{n_clusters}.npy', centers)