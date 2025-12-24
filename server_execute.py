import numpy as np
from sklearn.cluster import KMeans

def calculate_distance(cluster1, cluster2):
    # 计算两个类簇中心之间的欧氏距离
    return np.linalg.norm(cluster1['center'] - cluster2['center'])

def merge_clusters(clusters, cluster_indices, is_first_merge=False):
    """
    合并多个类簇，计算新的类簇中心、样本数和平均半径。

    :param clusters: 当前的类簇列表
    :param cluster_indices: 当前合并的类簇的索引
    :param is_first_merge: 标志是否为第一次合并，第一次合并使用加权K-Means算法，之后基于类簇距离合并
    :return: 新合并的类簇
    """
    num_points_sum = 0
    weighted_sum_centers = np.zeros_like(clusters[0]['center'])

    # 计算加权中心点和样本总数
    for idx in cluster_indices:
        cluster = clusters[idx]
        num_points_sum += cluster['num_points']
        weighted_sum_centers += cluster['center'] * cluster['num_points']

    # 计算新的类簇中心
    new_center = weighted_sum_centers / num_points_sum

    # if is_first_merge:
    #     # 对于第一次合并，直接返回加权中心作为新的类簇中心
    #     return {
    #         'center': new_center,
    #         'num_points': num_points_sum
    #     }

    # 后续合并计算平均半径
    total_distance = 0
    num_points_sum = 0
    for idx in cluster_indices:
        cluster = clusters[idx]
        num_points_sum += cluster['num_points']
        total_distance +=  cluster['num_points'] * cluster['avg_radius']
    avg_radius = total_distance / num_points_sum

    return {
        'center': new_center,
        'num_points': num_points_sum,
        'avg_radius': avg_radius
    }


def server_process(client_clusters, k, random_seed=None, is_first_merge=False, previous_clusters=None):
    """
    服务器端处理函数，接收所有客户端传来的类簇，进行聚类，并返回新的类簇
    :param is_first_merge: 标志是否为第一次类簇合并
    :param previous_clusters: 上一次合并后的类簇（用于第二次及之后的合并）
    :return: 新的类簇
    """
    all_clusters = client_clusters
    num_clusters = len(all_clusters)

    # 如果是第一次合并，使用加权K-Means
    if is_first_merge:
        data_points = np.array([cluster['center'] for cluster in all_clusters])
        data_weights = np.array([cluster['num_points'] for cluster in all_clusters])

        # 使用加权 K-Means 进行合并
        kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init='auto')
        kmeans.fit(data_points, sample_weight=data_weights)

        new_clusters = []
        for i in range(k):
            # 找到属于当前 K-Means 类别的客户端类簇
            cluster_indices = [j for j in range(num_clusters) if kmeans.labels_[j] == i]

            # 获取合并后类簇的中心
            new_center = kmeans.cluster_centers_[i]

            # 计算合并后类簇的样本数
            total_num_points = sum(all_clusters[idx]['num_points'] for idx in cluster_indices)

            # 创建新的合并类簇
            new_clusters.append({
                'center': new_center,
                'num_points': total_num_points
            })

        return new_clusters  # 返回合并后的新类簇

    # 后续合并，使用上次合并后的类簇进行类簇分配
    else:
        updated_clusters = []  # 用来保存新的合并类簇
        cluster_assignments = [[] for i in range(k)]  # 用来记录每个服务器类簇分配的客户端类簇

        # 对每个客户端传来的类簇进行分配，找出距离最近的上一轮的服务器类簇
        for i, cluster in enumerate(all_clusters):
            min_dist = float('inf')
            closest_cluster_index = -1

            # 计算客户端类簇与上一次合并的服务器类簇之间的距离
            for j, prev_cluster in enumerate(previous_clusters):
                dist = calculate_distance(prev_cluster, cluster)
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster_index = j  # 获取最近的服务器类簇索引
            # 将该客户端类簇分配给最近的服务器类簇
            cluster_assignments[closest_cluster_index].append(i)

        # 对每个服务器类簇合并属于它的客户端类簇
        for i in range(k):
            if cluster_assignments[i]:
                new_cluster = merge_clusters(all_clusters, cluster_assignments[i])
                updated_clusters.append(new_cluster)

        return updated_clusters  # 返回更新后的合并类簇
