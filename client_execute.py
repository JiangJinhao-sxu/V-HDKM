import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def calculate_distance(cluster1, cluster2):
    # 计算两个类簇中心之间的欧氏距离
    return np.linalg.norm(cluster1['center'] - cluster2)

def client_kmeans(local_data, server_clusters=None, k=4, random_seed=None):
    if not server_clusters:
        return perform_kmeans(local_data, k, random_seed)
    else:
        return adjust_clusters_with_radius(local_data, server_clusters, k, random_seed)


def perform_kmeans(data, num_clusters, random_seed=None):
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=random_seed, n_init='auto').fit(data)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    clusters = {i: [] for i in range(num_clusters)}  # 用于存储每个类簇中的数据点
    for idx, label in enumerate(labels):  # 将每个数据点分配到相应的类簇中
        clusters[label].append(data[idx])

    cluster_info = []  # 用于存储每个类簇的信息
    for i in range(num_clusters):
        cluster_center = centers[i]
        cluster_points = clusters[i]
        cluster_info.append({'center': cluster_center, 'num_points': len(cluster_points)})
    return cluster_info


def adjust_clusters_with_radius(local_data, server_clusters, k, random_seed=None):
    labels = []

    # 为每个数据点分配最近的类簇
    for point in local_data:
        min_dist = float('inf')  # 用于存储到最近类簇中心的距离
        closest_cluster_index = None  # 用于记录最近类簇的索引
        for i, cluster in enumerate(server_clusters):
            dist = np.linalg.norm(point - cluster['center'])
            if dist < min_dist:
                min_dist = dist
                closest_cluster_index = i
        labels.append(closest_cluster_index)

    # 根据分配的标签重新划分数据点
    updated_clusters = {i: [] for i in range(k)}
    for point, label in zip(local_data, labels):
        updated_clusters[label].append(point)

    new_clusters = []
    total_distance_all_clusters = 0  # 用来存储所有 k 个簇的总距离

    t_dist = 0
    # 对每个服务器类簇合并属于它的客户端类簇
    for i in range(k):
        if updated_clusters[i]:
            for poin in updated_clusters[i]:
                distan = calculate_distance(server_clusters[i], poin)
                t_dist += distan

    for i in range(k):
        if updated_clusters[i]:  # 如果该类簇有数据点
            cluster_points = np.array(updated_clusters[i])  # 当前类簇的所有数据点
            center = np.mean(cluster_points, axis=0)  # 计算聚类中心
            num_points = len(cluster_points)  # 当前类簇的点数

            # 计算每个点到该中心的欧几里得距离并累加
            distances = cdist(cluster_points, [center], 'euclidean')  # 计算所有点到中心的距离
            total_distance = np.sum(distances)  # 将所有距离累加

            # 计算每个点到聚类中心的平均半径
            avg_radius = np.mean(distances)  # 计算所有点到中心的平均距离

            # 将计算结果添加到新聚类中
            new_clusters.append({
                'center': center,
                'num_points': num_points,
                'avg_radius': avg_radius
            })

            # 累加当前簇的总距离到所有簇的总距离
            total_distance_all_clusters += total_distance

    # 返回只包含有数据点的类簇
    return new_clusters, t_dist

