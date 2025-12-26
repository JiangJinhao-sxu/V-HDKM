import numpy as np
from sklearn.cluster import KMeans

def calculate_distance(cluster1, cluster2):

    return np.linalg.norm(cluster1['center'] - cluster2['center'])

def merge_clusters(clusters, cluster_indices, is_first_merge=False):
   
    num_points_sum = 0
    weighted_sum_centers = np.zeros_like(clusters[0]['center'])

    for idx in cluster_indices:
        cluster = clusters[idx]
        num_points_sum += cluster['num_points']
        weighted_sum_centers += cluster['center'] * cluster['num_points']

    new_center = weighted_sum_centers / num_points_sum

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

    all_clusters = client_clusters
    num_clusters = len(all_clusters)

    if is_first_merge:
        data_points = np.array([cluster['center'] for cluster in all_clusters])
        data_weights = np.array([cluster['num_points'] for cluster in all_clusters])

        kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init='auto')
        kmeans.fit(data_points, sample_weight=data_weights)

        new_clusters = []
        for i in range(k):
         
            cluster_indices = [j for j in range(num_clusters) if kmeans.labels_[j] == i]

            new_center = kmeans.cluster_centers_[i]

            total_num_points = sum(all_clusters[idx]['num_points'] for idx in cluster_indices)

            new_clusters.append({
                'center': new_center,
                'num_points': total_num_points
            })

        return new_clusters 

    else:
        updated_clusters = [] 
        cluster_assignments = [[] for i in range(k)] 

        for i, cluster in enumerate(all_clusters):
            min_dist = float('inf')
            closest_cluster_index = -1

            for j, prev_cluster in enumerate(previous_clusters):
                dist = calculate_distance(prev_cluster, cluster)
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster_index = j  
           
            cluster_assignments[closest_cluster_index].append(i)

       
        for i in range(k):
            if cluster_assignments[i]:
                new_cluster = merge_clusters(all_clusters, cluster_assignments[i])
                updated_clusters.append(new_cluster)

        return updated_clusters 
