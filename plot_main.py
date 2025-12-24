import os
import numpy as np
import pandas as pd
import random
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import cdist
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import MinMaxScaler
from glob import glob
import server_execute
import client_execute

def calculate_federated_ari(client_data, client_labels, final_clusters):
    predicted_labels = []

    for point in client_data:
        distances = cdist([point], [cluster['center'] for cluster in final_clusters], 'euclidean')[0]

        cluster_index = np.argmin(distances)
        predicted_labels.append(cluster_index)

    ari = adjusted_rand_score(client_labels, predicted_labels)
    return ari

def calculate_nmi(client_data, client_labels, final_clusters):
    predicted_labels = []

    for point in client_data:
        distances = cdist([point], [cluster['center'] for cluster in final_clusters], 'euclidean')[0]

        cluster_index = np.argmin(distances)
        predicted_labels.append(cluster_index)

    nmi = normalized_mutual_info_score(client_labels, predicted_labels)
    return nmi

def calculate_accuracy(client_data, client_labels, final_clusters):

    predicted_labels = []

    for point in client_data:
        distances = cdist([point], [cluster['center'] for cluster in final_clusters], 'euclidean')[0]

        cluster_index = np.argmin(distances)
        predicted_labels.append(cluster_index)

    cost_matrix = np.zeros((k, k))

    for i in range(len(client_labels)):
        cost_matrix[predicted_labels[i], client_labels[i]] += 1

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    predicted_labels_mapped = [col_ind[label] for label in predicted_labels]
    accuracy = accuracy_score(client_labels, predicted_labels_mapped)
    return accuracy

def calculate_local_zhibiao(client_data, client_labels, k ,random_seed):
    kmeans = KMeans(n_clusters=k, random_state = random_seed, n_init='auto').fit(client_data)
    predicted_labels = kmeans.labels_

    ari = adjusted_rand_score(client_labels, predicted_labels)

    nmi = normalized_mutual_info_score(client_labels, predicted_labels)

    cost_matrix = np.zeros((k, k))

    for i in range(len(client_labels)):
        cost_matrix[predicted_labels[i], client_labels[i]] += 1

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    predicted_labels_mapped = [col_ind[label] for label in predicted_labels]
    accuracy = accuracy_score(client_labels, predicted_labels_mapped)

    return ari, nmi,accuracy

# 设置随机种子
random.seed(2)
np.random.seed(2)

folder_path = r'data/100Leaves'

file_name = folder_path.split('data_duoshitu\\')[-1]

client_files = sorted(glob(os.path.join(folder_path, 'client_*.xlsx')),
                     key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

label_df = pd.read_excel(os.path.join(folder_path, 'label.xlsx'), header=0)
labels = label_df.iloc[:, 0].values

labels = labels.astype(int) - 1

clients_data = {}
clients_labels = {}
client_dimensions = {}

clients_data1 = {}
clients_labels1 = {}

clients_data_y = {}
clients_labels_y = {}

for idx, client_file in enumerate(client_files):

    client_name = int(os.path.basename(client_file).split('.')[0].split('_')[-1])

    client_df = pd.read_excel(client_file, header=0)
    raw_data = client_df.values

    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(raw_data)

    dim = client_df.shape[1]
    client_dimensions[client_name] = dim

    clients_data[client_name] = normalized_data

    clients_labels[client_name] = labels

num_clients = len(client_files)
k = len(set(labels))

for client_id in range(1, num_clients + 1):

    sampled_dimensions = np.random.choice(client_dimensions[client_id], size=client_dimensions[client_id], replace=True)

    unique_dimensions = np.unique(sampled_dimensions)

    clients_data_y[client_id] = clients_data[client_id][:, unique_dimensions]
    clients_data1[client_id] = clients_data[client_id][:, unique_dimensions]

    clients_labels_y[client_id] = labels
    clients_labels1[client_id] = labels

client_distance_matrices = {}
scaler = MinMaxScaler()
for client_id in range(1, num_clients + 1):
    client_features1 = clients_data1[client_id]

    data_set_normalized = client_features1

    data_set_transposed = data_set_normalized.T

    new_data = []

    for _ in range(k * k):

        sampled_data = np.random.choice(data_set_transposed.shape[0], size = data_set_transposed.shape[0], replace=True)

        mean_data = np.mean(data_set_transposed[sampled_data], axis=0)

        new_data.append(mean_data)

    new_data = np.array(new_data)

    data_set_transposed = np.vstack((data_set_transposed, new_data))

    client_distance_matrices[client_id] = data_set_transposed

client_data = {}
client_labels = {}

for client_id in range(1, num_clients + 1):
    client_data[client_id] = client_distance_matrices[client_id]
    client_labels[client_id] = labels

num_trials = 30
tolerance = 0.1
eps = np.finfo(float).eps

all_federated_aris = []
all_federated_nmis = []
all_federated_accuracies = []

all_loacl_aris = []
all_loacl_nmis = []
all_loacl_accuracies = []

all_loacl_aris1 = []
all_loacl_nmis1 = []
all_loacl_accuracies1 = []

dist = {}

for external_trial in range(num_trials):
    random_seed = external_trial + 1  # 使用 external_trial 作为随机种子
    print(f"External Trial {external_trial + 1} with Random Seed {random_seed}")

    client_models = {}

    for client in range(1, num_clients + 1):
        client_data_np = np.array(client_data[client])
        client_models[client] = client_execute.client_kmeans(client_data_np, k=k, random_seed=random_seed)

    previous_final_clusters = None
    iteration = 0
    trial_distances = []
    final_centers = []

    while True:
        print(f"Iteration {iteration + 1}")

        server_clusters = []
        for client in range(1, num_clients + 1):
            server_clusters.extend(client_models[client])


        final_clusters = server_execute.server_process(server_clusters, k, random_seed=random_seed,
                                                       is_first_merge=(iteration == 0),
                                                       previous_clusters=previous_final_clusters)

        final_centers = final_clusters


        if previous_final_clusters is not None:
            total_distance_center = 0
            selected_indices = set()
            matching_pairs = []

            for new_fc in final_clusters:
                distances = []
                for index, pfc in enumerate(previous_final_clusters):
                    if index not in selected_indices:
                        distance = np.linalg.norm(np.array(new_fc['center']) - np.array(pfc['center']))
                        distances.append((distance, index))

                if distances:
                    min_distance, min_index = min(distances, key=lambda x: x[0])
                    total_distance_center += min_distance
                    selected_indices.add(min_index)
                    matching_pairs.append((new_fc, previous_final_clusters[min_index]))

            max_change = total_distance_center

            if max_change < eps:
                break

        total_distance_client = {i: 0 for i in range(1, num_clients + 1)}
        total_distance_iteration = 0

        for client in range(1, num_clients + 1):
            local_data = np.array(client_data[client])
            client_models[client], total_distance_client[client] = client_execute.client_kmeans(
                local_data, server_clusters=final_clusters, k=k,
                random_seed=random_seed)
            total_distance_iteration += total_distance_client[client]

        trial_distances.append(total_distance_iteration)

        previous_final_clusters = final_clusters
        iteration += 1

    final_centers_transposed = np.array([fc['center'] for fc in final_centers]).T

    data_set_normalized1 = scaler.fit_transform(final_centers_transposed)

    for client in range(1, num_clients + 1):

        cli_data = np.array(clients_data1[client])

        concatenated_data = np.hstack((cli_data, data_set_normalized1))

        clients_data1[client] = concatenated_data

    local_aris = {}
    local_nmis = {}
    local_accuracies = {}

    for client in range(1, num_clients + 1):
        local_aris[client], local_nmis[client], local_accuracies[client] = calculate_local_zhibiao(
            np.array(clients_data1[client]),
            np.array(clients_labels1[client]),
            k,
            random_seed
        )

    all_loacl_aris.append(local_aris)
    all_loacl_nmis.append(local_nmis)
    all_loacl_accuracies.append(local_accuracies)

    local_aris1 = {}
    local_nmis1 = {}
    local_accuracies1 = {}

    for client in range(1, num_clients + 1):
        local_aris1[client], local_nmis1[client], local_accuracies1[client] = calculate_local_zhibiao(
            np.array(clients_data_y[client]),
            np.array(clients_labels_y[client]),
            k,
            random_seed
        )

    all_loacl_aris1.append(local_aris1)
    all_loacl_nmis1.append(local_nmis1)
    all_loacl_accuracies1.append(local_accuracies1)


average_aris = [sum(d.values()) / len(d) for d in all_loacl_aris]
average_nmis = [sum(d.values()) / len(d) for d in all_loacl_nmis]
average_accuracies = [sum(d.values()) / len(d) for d in all_loacl_accuracies]




# 计算联邦的30轮平均ARI
average_local_aris = {client: np.mean([local_aris[client] for local_aris in all_loacl_aris]) for
                      client in range(1, num_clients + 1)}

overall_average_ari = np.mean(list(average_local_aris.values()))

print(f"Average Federated ARIs for each client over 30 trials:{overall_average_ari}")

print("-------------------------------------------------------------------------")

# 计算30轮的平均NMI
average_local_nmis = {client: np.mean([local_nmis[client] for local_nmis in all_loacl_nmis]) for
                      client in range(1, num_clients + 1)}

overall_average_nmi = np.mean(list(average_local_nmis.values()))

print(f"Average Federated NMIs for each client over 30 trials:{overall_average_nmi}")

print("-------------------------------------------------------------------------")

# 计算30轮的平均ACC
average_local_accuracies = {client: np.mean([local_accuracies[client] for local_accuracies in all_loacl_accuracies])
                            for
                            client in range(1, num_clients + 1)}

overall_average_acc = np.mean(list(average_local_accuracies.values()))

print(f"Average Federated ACCs for each client over 30 trials:{overall_average_acc}")

print("-------------------------------------------------------------------------")



# 计算本地的30轮平均ARI
average_local_aris1= {client: np.mean([local_aris1[client] for local_aris1 in all_loacl_aris1]) for
                      client in range(1, num_clients + 1)}

overall_average_ari1 = np.mean(list(average_local_aris1.values()))

print(f"Average Local ARTs for each client over 30 trials:{overall_average_ari1}")

print("-------------------------------------------------------------------------")

# 计算30轮的平均NMI
average_local_nmis1 = {client: np.mean([local_nmis1[client] for local_nmis1 in all_loacl_nmis1]) for
                      client in range(1, num_clients + 1)}

overall_average_nmis1 = np.mean(list(average_local_nmis1.values()))

print(f"Average Local NMIs for each client over 30 trials:{overall_average_nmis1}")

print("-------------------------------------------------------------------------")

# 计算30轮的平均ACC
average_local_accuracies1 = {client: np.mean([local_accuracies1[client] for local_accuracies1 in all_loacl_accuracies1])
                            for
                            client in range(1, num_clients + 1)}

overall_average_accuracies1 = np.mean(list(average_local_accuracies1.values()))

print(f"Average Local ACCs for each client over 30 trials:{overall_average_accuracies1}")

print("-------------------------------------------------------------------------")

# ari, nmi,accuracy=calculate_local_zhibiao(data_set, labels , k ,random_seed=2)
# print(ari, nmi,accuracy)





