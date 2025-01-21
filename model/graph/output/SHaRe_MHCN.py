import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss
from data.social import Relation
import torch.nn.functional as F
from collections import defaultdict
import pickle
import time
import networkx as nx
import scipy.sparse as sps
import numpy as np
import math
import tqdm
import random
import os
from itertools import combinations
import concurrent.futures


class SHaRe_MHCN(GraphRecommender):
    def __init__(self, conf, training_set, valid_set, test_set, **kwargs):
        super(SHaRe_MHCN, self).__init__(conf, training_set, valid_set, test_set, **kwargs)
        self.n_layers = int(self.config['n_layer'])
        self.lambd = conf['lambda']
        self.social_data = Relation(conf, kwargs['social.data'], self.data.user)
        S = self.social_data.get_social_mat()
        self.model = SHaRe_MHCN_Encoder(self.data, self.emb_size, self.n_layers, S, conf)
        self.device = torch.device("cuda:" + str(conf['gpu']))

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        epoch_times = []
        for epoch in range(self.maxEpoch):
            start_time = time.time()

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch

                rec_user_emb, rec_item_emb = model()

                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                    neg_idx]
                if self.config['data'] == 'lastfm':
                    ssl_loss = model._ssl_loss(user_idx)
                else:
                    ssl_loss = model._ssl_loss_v2(user_idx)
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,
                                                                                          pos_item_emb,
                                                                                          neg_item_emb) / self.batch_size + \
                             self.config['lambda'] * ssl_loss

                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())

            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)

            # valid
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            if epoch % 5 == 0:
                stop = self.fast_evaluation_valid(epoch, model, 'pt/' + self.config['model.name'] + '_best.pt')
                measure = self.fast_evaluation_test(epoch)
                if stop:
                    break

        # test
        best_model = torch.load('pt/' + self.config['model.name'] + '_best.pt').to(self.device)
        with torch.no_grad():
            self.user_emb, self.item_emb = best_model()
        measure = self.fast_evaluation_test(epoch)
        # file = open('model_result/' + self.config['data'] + '/' + self.config['model.name'] + '_' + str(
        #    self.config['times']) + '.pkl', 'wb')
        # pickle.dump(measure, file)
        # file.close()

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.detach().cpu().numpy()


class LightGCNConv(nn.Module):
    def __init__(self):
        super(LightGCNConv, self).__init__()

    def forward(self, R, X):
        side_embeddings = torch.sparse.mm(R, X)

        return side_embeddings


class Granular_Decompose():
    def __init__(self, user_num, item_num, conf, S):
        self.user_num = user_num
        self.item_num = item_num
        self.conf = conf
        self.S = S

    def fast_clustering(self, graph):
        coeffs = {}
        for node in graph:
            neighbors = list(graph.neighbors(node))
            k = len(neighbors)
            if k < 2:
                coeffs[node] = 0.0
                continue

            # Count edges between neighbors
            edges_between_neighbors = sum(
                1 for i in range(k) for j in range(i + 1, k) if graph.has_edge(neighbors[i], neighbors[j]))
            coeffs[node] = (2 * edges_between_neighbors) / (k * (k - 1))

        return sum(coeffs.values()) / len(coeffs)

    def homosity(self, graph):
        user_nodes = list(node for node in graph.nodes() if node < self.user_num)
        user_graph = nx.subgraph(graph, user_nodes)
        uu_homosity = [data['weight'] for _, _, data in user_graph.edges(data=True)]
        if len(uu_homosity) == 0:
            return 0
        return np.mean(uu_homosity)

    def sampled_clustering(self, graph, sample_size):
        sampled_nodes = random.sample(graph.nodes(), min(sample_size, len(graph.nodes())))
        clustering_coeffs = nx.clustering(graph, nodes=sampled_nodes)
        return sum(clustering_coeffs.values()) / len(clustering_coeffs)

    def qity(self, graph):
        # density = nx.density(graph)
        # avg_clustering_coeff = self.fast_clustering_coefficient(graph)
        # avg_clustering_coeff = self.sampled_clustering(graph, 1000)
        # qity = density + avg_clustering_coeff
        qity = self.homosity(graph)
        return qity

    def init_GB_graph(self, graph, init_GB_num, user_num, item_num):
        # Obtain the degree of nodes and sort them
        degree_dict = dict(graph.degree())
        sorted_points = sorted(degree_dict, key=degree_dict.get, reverse=True)
        user_nodes = list(node for node in sorted_points if node < self.user_num)
        visited_center = []
        # if self.conf['data'] == 'yelp':
        #    visited_center.extend(user_nodes[:16])
        #    user_nodes = user_nodes[16:]
        user_graph = nx.subgraph(graph, user_nodes)
        user_degree_dict = dict(user_graph.degree(weight='weight'))
        sorted_nodes = sorted(user_degree_dict, key=user_degree_dict.get, reverse=True)

        # Select central node

        center_nodes = [sorted_nodes[0]]  # The first central node is the node with the highest degree
        center_paths = []
        center_paths.append(nx.single_source_shortest_path_length(graph, source=sorted_nodes[0]))

        distance_map = defaultdict(lambda: 10e6)
        distance_map.update(center_paths[0])
        max_degree = max(user_degree_dict.values())
        for center_num in tqdm.trange(1, init_GB_num, desc='find init center'):
            to_center_list = []
            max_distance = max([dis for dis in distance_map.values() if dis < 10e6])
            for node in sorted_nodes:
                if node not in center_nodes and distance_map[node] < 10e6:
                    min_to_center = distance_map[node]
                    node_degree = user_degree_dict[node]
                    if min_to_center > 1:
                        to_center_list.append((node, math.sqrt(
                            (min_to_center / (max_distance + 1)) * (node_degree / (max_degree + 1))), min_to_center,
                                               node_degree))
            next_node = max(to_center_list, key=lambda x: x[1])[0]
            center_nodes.append(next_node)
            new_paths = nx.single_source_shortest_path_length(graph, source=next_node)
            # Update distances
            for node, dist in new_paths.items():
                if dist < distance_map[node]:
                    distance_map[node] = dist
            # Add the new paths
            center_paths.append(new_paths)
            distances = np.array(list(distance_map.values()))
            if sum(distances == 10e6) + sum(distances <= 1) == len(distances):
                print('center enough:', center_num + 1)
                break

        # add center of independent island
        for node in sorted_nodes:
            if node not in center_nodes and distance_map[node] == 10e6:
                next_node = node
                center_nodes.append(next_node)
                new_paths = nx.single_source_shortest_path_length(graph, source=next_node)
                # Update distances
                for node, dist in new_paths.items():
                    if dist < distance_map[node]:
                        distance_map[node] = dist
                    # Add the new paths
                center_paths.append(new_paths)
        visited_center.extend(center_nodes)
        point_nodes = [node for node in sorted_points if node not in center_nodes]
        # Initialize path and cluster
        center_paths = []
        clusters = []
        for center in center_nodes:
            center_paths.append(nx.single_source_shortest_path_length(graph, source=center))
            clusters.append([center])
        # Assign nodes to the nearest central node cluster
        for point in tqdm.tqdm(point_nodes):
            point_to_center_len = 10e6
            min_homo = 0
            highest_homo_centers = []
            nearest_centers = []

            if point < self.user_num:
                for idx in range(len(clusters)):
                    users_in_cluster = [nei for nei in clusters[idx] if nei < self.user_num]
                    cluster_size = len(users_in_cluster)
                    cur_homo = self.S[point, users_in_cluster].sum() / ((1 + cluster_size) ** 2)
                    if cur_homo > min_homo:
                        min_homo = cur_homo
                        highest_homo_centers = [idx]
                    elif cur_homo == min_homo:
                        highest_homo_centers.append(idx)
                chosen_center_idx = min(highest_homo_centers, key=lambda idx: center_paths[idx].get(point, 10e6))
                clusters[chosen_center_idx].append(point)
            else:
                for idx, center_path in enumerate(center_paths):
                    distance = center_path.get(point, 10e6)
                    if distance < point_to_center_len:
                        point_to_center_len = distance
                        nearest_centers = [idx]  # Found a closer center
                    elif distance == point_to_center_len:
                        nearest_centers.append(idx)
                for chosen_center_idx in nearest_centers:
                    clusters[chosen_center_idx].append(point)

            # Randomly assign to one of the nearest centers if there are ties
            # chosen_center_idx = random.choice(nearest_centers)
            # clusters[chosen_center_idx].append(point)
        # Build initial subgraph list
        init_GB_list = [(nx.subgraph(graph, cluster), center_nodes[idx]) for idx, cluster in enumerate(clusters)]
        return init_GB_list, visited_center

    def split_ball(self, graph, user_num, item_num, visited_center):
        split_GB_list = []
        split_GB_node_list = []
        split_GB_num = 2
        degree_dict = dict(graph.degree())
        avg_degree = self.qity(graph)
        # user_degree_dict = {node: degree_dict[node] for node in degree_dict if node < user_num}
        # sorted_nodes = sorted(user_degree_dict, key=user_degree_dict.get, reverse=True)
        user_nodes = list(node for node in graph.nodes() if node < self.user_num)
        user_graph = nx.subgraph(graph, user_nodes)
        user_degree_dict = dict(user_graph.degree(weight='weight'))
        sorted_points = sorted(degree_dict, key=degree_dict.get, reverse=True)
        sorted_nodes = sorted(user_degree_dict, key=user_degree_dict.get, reverse=True)
        sorted_nodes = [center for center in sorted_nodes if center not in visited_center]
        sorted_nodes = [center for center in sorted_nodes if degree_dict[center] > degree_dict[sorted_nodes[0]] / 10]
        if len(graph) == 1 or len(sorted_nodes) < 3:
            split_GB_list.append(graph)
            split_GB_node_list.append(list(graph.nodes()) + [[None, avg_degree]])
            return split_GB_list, split_GB_node_list
        cent_start_id = 0
        while cent_start_id < len(sorted_nodes) - 1 and len(
                nx.single_source_shortest_path_length(graph, source=sorted_nodes[cent_start_id])) < len(graph) / 100:
            visited_center.append(sorted_nodes[cent_start_id])
            cent_start_id += 1
        center_nodes = [sorted_nodes[cent_start_id]]  # The first central node is the node with the highest degree
        center_paths = []
        center_paths.append(nx.single_source_shortest_path_length(graph, source=sorted_nodes[cent_start_id]))
        distance_map = defaultdict(lambda: 10e6)
        distance_map.update(center_paths[0])
        max_degree = max(user_degree_dict.values())
        for _ in range(1, split_GB_num):
            max_distance = max([dis for dis in distance_map.values() if dis < 10e6])
            to_center_list = []
            for node in sorted_nodes:
                if node not in center_nodes and distance_map[node] < 10e6:
                    min_to_center = distance_map[node]
                    node_degree = user_degree_dict[node]
                    # if min_to_center > 1:
                    to_center_list.append((node,
                                           math.sqrt(
                                               (min_to_center / (max_distance + 1)) * (node_degree / (max_degree + 1))),
                                           min_to_center, node_degree))
            if len(to_center_list) < 1:
                break
            next_node = max(to_center_list, key=lambda x: x[1])[0]
            center_nodes.append(next_node)
            new_paths = nx.single_source_shortest_path_length(graph, source=next_node)
            # Update distances
            for node, dist in new_paths.items():
                if dist < distance_map[node]:
                    distance_map[node] = dist
            # Add the new paths
            center_paths.append(new_paths)
        visited_center += center_nodes
        # center_nodes = sorted_nodes[:2]
        point_nodes = [point for point in sorted_points if point not in center_nodes]
        center_paths = []
        clusters = []
        # visited_center += center_nodes
        if len(center_nodes) < 2:
            split_GB_list.append(graph)
            split_GB_node_list.append(list(graph.nodes()) + [[None, avg_degree]])
            return split_GB_list, split_GB_node_list
        for center in center_nodes:
            center_paths.append(nx.single_source_shortest_path_length(graph, source=center))
            clusters.append([center])
        for point in point_nodes:
            point_to_center_len = 10e6
            min_homo = 0
            highest_homo_centers = []
            nearest_centers = []
            if point < self.user_num:
                for idx in range(len(clusters)):
                    users_in_cluster = [nei for nei in clusters[idx] if nei < self.user_num]
                    cluster_size = len(users_in_cluster)
                    cur_homo = self.S[point, users_in_cluster].sum() / ((1 + cluster_size) ** 2)
                    if cur_homo > min_homo:
                        min_homo = cur_homo
                        highest_homo_centers = [idx]
                    elif cur_homo == min_homo:
                        highest_homo_centers.append(idx)
                chosen_center_idx = min(highest_homo_centers, key=lambda idx: center_paths[idx].get(point, 10e6))
                clusters[chosen_center_idx].append(point)
            else:
                for idx, center_path in enumerate(center_paths):
                    distance = center_path.get(point, 10e6)
                    if distance < point_to_center_len:
                        point_to_center_len = distance
                        nearest_centers = [idx]  # Found a closer center
                    elif distance == point_to_center_len:
                        nearest_centers.append(idx)
                for chosen_center_idx in nearest_centers:
                    clusters[chosen_center_idx].append(point)

            # Randomly assign to one of the nearest centers if there are ties
            # chosen_center_idx = random.choice(nearest_centers)
            # clusters[chosen_center_idx].append(point)
        cluster_a = clusters[0]
        cluster_b = clusters[1]

        graph_a = nx.subgraph(graph, cluster_a)
        graph_b = nx.subgraph(graph, cluster_b)
        # After dividing the Granular-Ball, there may be two disconnected points dividing into one ball, and splitting is prohibited at this time
        if len(graph_a.edges()) == 0 or len(graph_b.edges()) == 0:
            split_GB_list.append(graph)
            split_GB_node_list.append(list(graph.nodes()) + [[None, avg_degree]])
        else:
            # Computational Quality
            avg_degree_a = self.qity(graph_a)
            avg_degree_b = self.qity(graph_b)

            # Determine whether it is splitting
            if avg_degree < max(avg_degree_a,
                                avg_degree_b):  # (avg_degree_a + avg_degree_b) / 2+0.05:  # (avg_degree_a*len(graph_b)+avg_degree_b*len(graph_a))/(len(graph_a)+len(graph_b)):#max((avg_degree_a, avg_degree_b)):#(avg_degree_a + avg_degree_b)/2:
                split_GB_list_a, split_GB_node_list_a = self.split_ball(graph_a, user_num, item_num, visited_center)
                split_GB_list.append((split_GB_list_a, (center_nodes[0], avg_degree_a)))
                split_GB_node_list.append((split_GB_node_list_a, (center_nodes[0], avg_degree_a)))
                split_GB_list_b, split_GB_node_list_b = self.split_ball(graph_b, user_num, item_num, visited_center)
                split_GB_list.append((split_GB_list_b, (center_nodes[1], avg_degree_b)))
                split_GB_node_list.append((split_GB_node_list_b, (center_nodes[1], avg_degree_b)))

            else:
                split_GB_list.append(graph)
                split_GB_node_list.append(list(graph.nodes()) + [[None, avg_degree]])
                return split_GB_list, split_GB_node_list
        return split_GB_list, split_GB_node_list

    def init_GB_graph_homo(self, graph, init_GB_num, user_num, item_num):
        # Obtain the degree of nodes and sort them
        degree_dict = dict(graph.degree())
        sorted_points = sorted(degree_dict, key=degree_dict.get, reverse=True)
        user_nodes = list(node for node in sorted_points if node < self.user_num)
        visited_center = []
        # if self.conf['data'] == 'yelp':
        #    visited_center.extend(user_nodes[:16])
        #    user_nodes = user_nodes[16:]
        user_graph = nx.subgraph(graph, user_nodes)
        user_degree_dict = dict(user_graph.degree(weight='weight'))
        sorted_nodes = sorted(user_degree_dict, key=user_degree_dict.get, reverse=True)

        # Select central node

        center_nodes = [sorted_nodes[0]]  # The first central node is the node with the highest degree
        center_paths = []
        center_paths.append(nx.single_source_shortest_path_length(graph, source=sorted_nodes[0]))

        distance_map = defaultdict(lambda: 10e6)
        distance_map.update(center_paths[0])
        max_degree = max(user_degree_dict.values())
        for center_num in tqdm.trange(1, init_GB_num, desc='find init center'):
            to_center_list = []
            max_distance = max([dis for dis in distance_map.values() if dis < 10e6])
            for node in sorted_nodes:
                if node not in center_nodes and distance_map[node] < 10e6:
                    min_to_center = distance_map[node]
                    node_degree = user_degree_dict[node]
                    if min_to_center > 1:
                        to_center_list.append((node, math.sqrt(
                            (min_to_center / (max_distance + 1)) * (node_degree / (max_degree + 1))), min_to_center,
                                               node_degree))
            next_node = max(to_center_list, key=lambda x: x[1])[0]
            center_nodes.append(next_node)
            new_paths = nx.single_source_shortest_path_length(graph, source=next_node)
            # Update distances
            for node, dist in new_paths.items():
                if dist < distance_map[node]:
                    distance_map[node] = dist
            # Add the new paths
            center_paths.append(new_paths)
            distances = np.array(list(distance_map.values()))
            if sum(distances == 10e6) + sum(distances <= 1) == len(distances):
                print('center enough:', center_num + 1)
                break

        # add center of independent island
        for node in sorted_nodes:
            if node not in center_nodes and distance_map[node] == 10e6:
                next_node = node
                center_nodes.append(next_node)
                new_paths = nx.single_source_shortest_path_length(graph, source=next_node)
                # Update distances
                for node, dist in new_paths.items():
                    if dist < distance_map[node]:
                        distance_map[node] = dist
                    # Add the new paths
                center_paths.append(new_paths)
        visited_center.extend(center_nodes)
        point_nodes = [node for node in sorted_points if node not in center_nodes]
        # Initialize path and cluster
        center_paths = []
        clusters = []
        for center in center_nodes:
            center_paths.append(nx.single_source_shortest_path_length(graph, source=center))
            clusters.append([center])
        # Assign nodes to the nearest central node cluster
        for point in tqdm.tqdm(point_nodes):
            point_to_center_len = 10e6
            nearest_centers = []
            for idx, center_path in enumerate(center_paths):
                distance = center_path.get(point, 10e6)
                if distance < point_to_center_len:
                    point_to_center_len = distance
                    nearest_centers = [idx]  # Found a closer center
                elif distance == point_to_center_len:
                    nearest_centers.append(idx)  # Found an equally distant center
            if nearest_centers:
                if point < self.user_num:
                    # if point is a user, chosen the cluster with max homosity for him
                    chosen_center_idx = max(nearest_centers, key=lambda idx: self.S[
                                                                                 point, [nei for nei in clusters[idx] if
                                                                                         nei < self.user_num]].sum() / (
                                                                                         (1 + len([nei for nei in
                                                                                                   clusters[idx] if
                                                                                                   nei < self.user_num])) ** 2))  # np.mean([self.S[point,nei] for nei in clusters[idx] if nei<self.user_num]))
                    # chosen_center_idx = max(nearest_centers, key=lambda idx: self.S[point,[nei for nei in clusters[idx] if nei<self.user_num]].sum()/((len(clusters[idx])+1)*(len(clusters[idx])+1))) #np.mean([self.S[point,nei] for nei in clusters[idx] if nei<self.user_num]))
                    clusters[chosen_center_idx].append(point)
                else:
                    # if point is an item, add it to the clusters with the same distances
                    for chosen_center_idx in nearest_centers:
                        clusters[chosen_center_idx].append(point)
            # Randomly assign to one of the nearest centers if there are ties
            # chosen_center_idx = random.choice(nearest_centers)
            # clusters[chosen_center_idx].append(point)
        # Build initial subgraph list
        init_GB_list = [(nx.subgraph(graph, cluster), center_nodes[idx]) for idx, cluster in enumerate(clusters)]
        return init_GB_list, visited_center

    def split_ball_homo(self, graph, user_num, item_num, visited_center):
        split_GB_list = []
        split_GB_node_list = []
        split_GB_num = 2
        degree_dict = dict(graph.degree())
        avg_degree = self.qity(graph)
        # user_degree_dict = {node: degree_dict[node] for node in degree_dict if node < user_num}
        # sorted_nodes = sorted(user_degree_dict, key=user_degree_dict.get, reverse=True)
        user_nodes = list(node for node in graph.nodes() if node < self.user_num)
        user_graph = nx.subgraph(graph, user_nodes)
        user_degree_dict = dict(user_graph.degree(weight='weight'))
        sorted_points = sorted(degree_dict, key=degree_dict.get, reverse=True)
        sorted_nodes = sorted(user_degree_dict, key=user_degree_dict.get, reverse=True)
        sorted_nodes = [center for center in sorted_nodes if center not in visited_center]
        sorted_nodes = [center for center in sorted_nodes if degree_dict[center] > degree_dict[sorted_nodes[0]] / 10]
        if len(graph) == 1 or len(sorted_nodes) < 3:
            split_GB_list.append(graph)
            split_GB_node_list.append(list(graph.nodes()) + [[None, avg_degree]])
            return split_GB_list, split_GB_node_list
        cent_start_id = 0
        while cent_start_id < len(sorted_nodes) - 1 and len(
                nx.single_source_shortest_path_length(graph, source=sorted_nodes[cent_start_id])) < len(graph) / 100:
            visited_center.append(sorted_nodes[cent_start_id])
            cent_start_id += 1
        center_nodes = [sorted_nodes[cent_start_id]]  # The first central node is the node with the highest degree
        center_paths = []
        center_paths.append(nx.single_source_shortest_path_length(graph, source=sorted_nodes[cent_start_id]))
        distance_map = defaultdict(lambda: 10e6)
        distance_map.update(center_paths[0])
        max_degree = max(user_degree_dict.values())
        for _ in range(1, split_GB_num):
            max_distance = max([dis for dis in distance_map.values() if dis < 10e6])
            to_center_list = []
            for node in sorted_nodes:
                if node not in center_nodes and distance_map[node] < 10e6:
                    min_to_center = distance_map[node]
                    node_degree = user_degree_dict[node]
                    # if min_to_center > 1:
                    to_center_list.append((node,
                                           math.sqrt(
                                               (min_to_center / (max_distance + 1)) * (node_degree / (max_degree + 1))),
                                           min_to_center, node_degree))
            if len(to_center_list) < 1:
                break
            next_node = max(to_center_list, key=lambda x: x[1])[0]
            center_nodes.append(next_node)
            new_paths = nx.single_source_shortest_path_length(graph, source=next_node)
            # Update distances
            for node, dist in new_paths.items():
                if dist < distance_map[node]:
                    distance_map[node] = dist
            # Add the new paths
            center_paths.append(new_paths)
        visited_center += center_nodes
        # center_nodes = sorted_nodes[:2]
        point_nodes = [point for point in sorted_points if point not in center_nodes]
        center_paths = []
        clusters = []
        # visited_center += center_nodes
        if len(center_nodes) < 2:
            split_GB_list.append(graph)
            split_GB_node_list.append(list(graph.nodes()) + [[None, avg_degree]])
            return split_GB_list, split_GB_node_list
        for center in center_nodes:
            center_paths.append(nx.single_source_shortest_path_length(graph, source=center))
            clusters.append([center])
        for point in point_nodes:
            point_to_center_len = 10e6
            nearest_centers = []
            for idx, center_path in enumerate(center_paths):
                distance = center_path.get(point, 10e6)
                if distance < point_to_center_len:
                    point_to_center_len = distance
                    nearest_centers = [idx]  # Found a closer center
                elif distance == point_to_center_len:
                    nearest_centers.append(idx)  # Found an equally distant center

            if nearest_centers:
                # chosen_center_idx = min(nearest_centers, key=lambda idx: len(clusters[idx]))
                # chosen_center_idx = max(nearest_centers, key=lambda idx: (-len(clusters[idx]),len(set(clusters[idx])&set(graph.neighbors(center_nodes[idx])))))#min(nearest_centers, key=lambda idx: len(clusters[idx]))
                if point < self.user_num:
                    # if point is a user, chosen the cluster with max homosity for him
                    chosen_center_idx = max(nearest_centers, key=lambda idx: self.S[
                                                                                 point, [nei for nei in clusters[idx] if
                                                                                         nei < self.user_num]].sum() / (
                                                                                         (1 + len([nei for nei in
                                                                                                   clusters[idx] if
                                                                                                   nei < self.user_num])) ** 2))  # np.mean([self.S[point,nei] for nei in clusters[idx] if nei<self.user_num]))
                    clusters[chosen_center_idx].append(point)
                else:
                    # if point is an item, add it to the clusters with the same distances
                    for chosen_center_idx in nearest_centers:
                        clusters[chosen_center_idx].append(point)

            # Randomly assign to one of the nearest centers if there are ties
            # chosen_center_idx = random.choice(nearest_centers)
            # clusters[chosen_center_idx].append(point)
        cluster_a = clusters[0]
        cluster_b = clusters[1]

        graph_a = nx.subgraph(graph, cluster_a)
        graph_b = nx.subgraph(graph, cluster_b)
        # After dividing the Granular-Ball, there may be two disconnected points dividing into one ball, and splitting is prohibited at this time
        if len(graph_a.edges()) == 0 or len(graph_b.edges()) == 0:
            split_GB_list.append(graph)
            split_GB_node_list.append(list(graph.nodes()) + [[None, avg_degree]])
        else:
            # Computational Quality
            avg_degree_a = self.qity(graph_a)
            avg_degree_b = self.qity(graph_b)

            # Determine whether it is splitting
            if avg_degree < max(avg_degree_a,
                                avg_degree_b):  # (avg_degree_a + avg_degree_b) / 2+0.05:  # (avg_degree_a*len(graph_b)+avg_degree_b*len(graph_a))/(len(graph_a)+len(graph_b)):#max((avg_degree_a, avg_degree_b)):#(avg_degree_a + avg_degree_b)/2:
                split_GB_list_a, split_GB_node_list_a = self.split_ball(graph_a, user_num, item_num, visited_center)
                split_GB_list.append((split_GB_list_a, (center_nodes[0], avg_degree_a)))
                split_GB_node_list.append((split_GB_node_list_a, (center_nodes[0], avg_degree_a)))
                split_GB_list_b, split_GB_node_list_b = self.split_ball(graph_b, user_num, item_num, visited_center)
                split_GB_list.append((split_GB_list_b, (center_nodes[1], avg_degree_b)))
                split_GB_node_list.append((split_GB_node_list_b, (center_nodes[1], avg_degree_b)))

            else:
                split_GB_list.append(graph)
                split_GB_node_list.append(list(graph.nodes()) + [[None, avg_degree]])
                return split_GB_list, split_GB_node_list
        return split_GB_list, split_GB_node_list

    def process_graph(self, init_GB_center_tuple):
        init_GB, center = init_GB_center_tuple
        split_GB_list, split_GB_node_list = self.split_ball(init_GB, self.user_num, self.item_num)
        return (split_GB_list, center), (split_GB_node_list, center)

    def get_GB(self, graph):
        init_GB_num = math.isqrt(len(graph))
        if self.conf['strategy'] == 'homo_first':
            init_GB_list, visited_center = self.init_GB_graph_homo(graph, init_GB_num, self.user_num, self.item_num)
        else:
            init_GB_list, visited_center = self.init_GB_graph(graph, init_GB_num, self.user_num, self.item_num)
        # print("init_GB_list",init_GB_list)
        GB_list = []
        GB_node_list = []
        # Use ProcessPoolExecutor for CPU-bound tasks

        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     # Submit tasks and handle results as they complete
        #     futures = {executor.submit(self.process_graph, item): item for item in init_GB_list}
        # for future in concurrent.futures.as_completed(futures):
        #     try:
        #         result = future.result()
        #         gb, gb_nodes = result
        #         GB_list.append(gb)
        #         GB_node_list.append(gb_nodes)
        #     except Exception as e:
        #         print(f"Error processing {futures[future]}: {e}")

        for (init_GB, center) in tqdm.tqdm(init_GB_list):
            if self.conf['strategy'] == 'homo_first':
                split_GB_list, split_GB_node_list = self.split_ball_homo(init_GB, self.user_num, self.item_num,
                                                                         visited_center)
            else:
                split_GB_list, split_GB_node_list = self.split_ball(init_GB, self.user_num, self.item_num,
                                                                    visited_center)
            avg_degree = self.qity(init_GB)
            GB_list.append((split_GB_list, (center, avg_degree)))
            GB_node_list.append((split_GB_node_list, (center, avg_degree)))

        # print('split ratio:{}, visited:{}, all:{} '.format(len(visited_center) / fea_num, len(visited_center), fea_num))
        return GB_list, GB_node_list

    def forward(self, A_mat):
        data_file = self.conf['data_path'] + self.conf['data'] + '/GB_node_array_' + self.conf['strategy'] + '.npy'
        graph = nx.from_scipy_sparse_matrix(A_mat)
        self.graph = graph
        if os.path.exists(data_file):
            GB_node_array = np.load(data_file, allow_pickle=True)
            return GB_node_array

        # if nx.is_connected(graph):
        GB_list, GB_node_list = self.get_GB(graph)
        """
        else:
            GB_list, GB_node_list = [], []
            connected_components = list(nx.connected_components(graph))
            connected_subgraphs = [graph.subgraph(component) for component in connected_components]
            for connected_subgraph in connected_subgraphs:
                sub_GB_list, sub_GB_node_list = self.get_GB(connected_subgraph)
                GB_list+=sub_GB_list
                GB_node_list+=sub_GB_node_list
        """
        GB_node_array = np.array(GB_node_list, dtype=object)
        np.save(data_file, GB_node_array)
        # print('split ratio:{}, visited:{}, all:{} '.format(len(visited_center) / fea_num, len(visited_center), fea_num))
        return GB_node_list


class TreeNode:
    def __init__(self, value, qity=None, is_center=True):
        self.value = value  # Center value or leaf value
        self.children = []
        self.qity = qity
        self.is_center = is_center  # To distinguish between center and leaf nodes

    def add_child(self, child_node):
        self.children.append(child_node)


class GatingLayer(nn.Module):
    def __init__(self, dim):
        super(GatingLayer, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(self.dim, self.dim)
        self.activation = nn.Sigmoid()

    def forward(self, emb):
        embedding = self.linear(emb)
        embedding = self.activation(embedding)
        embedding = torch.mul(emb, embedding)
        return embedding


class AttLayer(nn.Module):
    def __init__(self, dim):
        super(AttLayer, self).__init__()
        self.dim = dim
        self.attention_mat = nn.Parameter(torch.randn([self.dim, self.dim]))
        self.attention = nn.Parameter(torch.randn([1, self.dim]))

    def forward(self, *embs):
        weights = []
        emb_list = []
        for embedding in embs:
            weights.append(torch.sum(torch.mul(self.attention, torch.matmul(embedding, self.attention_mat)), dim=1))
            emb_list.append(embedding)
        score = torch.nn.Softmax(dim=0)(torch.stack(weights, dim=0))
        embeddings = torch.stack(emb_list, dim=0)
        mixed_embeddings = torch.mul(embeddings, score.unsqueeze(dim=2).repeat(1, 1, self.dim)).sum(dim=0)
        return mixed_embeddings


class SHaRe_MHCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, S, conf):
        super(SHaRe_MHCN_Encoder, self).__init__()
        self.data = data
        self.S = S
        self.A_mat, self.num_user, self.num_item = self.parse_dataset(self.data, S)
        self.granular = Granular_Decompose(self.num_user, self.num_item, conf, S)
        self.GB_node_list = self.granular.forward(self.A_mat)
        self.granular_tree = self.build_granular_tree(self.GB_node_list)
        self.granular_tree.qity = 0  # self.granular.qity(self.granular.graph)
        self.GB_flatten = []
        self.flatten__tree(self.granular_tree)
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj_big = data.norm_adj
        self.norm_adj = data.norm_interact
        self.interact_adj = data.interaction_mat
        self.embedding_dict = self._init_model()
        self.device = torch.device("cuda:" + str(conf['gpu']))
        self.sparse_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.interact_adj).to(self.device)
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).to(self.device)
        self.A_hat = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj_big).to(self.device)
        self.sparse_S = TorchGraphInterface.convert_sparse_mat_to_tensor(S).to(self.device)
        self.num_encoder_layer = 2
        self.relu = nn.ReLU(inplace=True)
        # self.omega = conf['omega']
        self.zeta = conf['zeta']
        self.weighted_S = self._social_rewiring(self.sparse_S)
        self.H_s, self.H_j, self.H_p = self.get_motif_adj_matrix(self.weighted_S, self.sparse_adj)
        # self.H_s, self.H_j, self.H_p = self.get_motif_adj_matrix(self.sparse_S, self.sparse_adj)

        # define gating layers
        self.gating_c1 = GatingLayer(self.latent_size)
        self.gating_c2 = GatingLayer(self.latent_size)
        self.gating_c3 = GatingLayer(self.latent_size)
        self.gating_simple = GatingLayer(self.latent_size)
        self.attention_layer = AttLayer(self.latent_size)
        self.encoder_layers = nn.ModuleList()
        for i in range(self.num_encoder_layer):
            layer = LightGCNConv()
            self.encoder_layers.append(layer)

    def flatten__tree(self, node, centers=[]):
        # print(" " * level + str(node.value))  # Print node value
        if isinstance(node.value, list):
            self.GB_flatten.append(
                ([user for user in node.value if user < self.num_user], centers + [(None, node.qity)]))
        else:
            for child in node.children:
                self.flatten__tree(child, centers + [(node.value, node.qity)])

    def parse_dataset(self, data, S):
        user_num = data.user_num
        item_num = data.item_num
        n = user_num + item_num
        interact_mat = data.interaction_mat.tocoo()
        row_ui, col_ui, val_ui = interact_mat.row, interact_mat.col, interact_mat.data
        row_ui, col_ui, val_ui = np.array(row_ui).reshape(-1, 1), np.array(col_ui).reshape(-1, 1), np.array(
            val_ui).reshape(-1, 1)
        col_ui = col_ui + user_num  # map item to item_id
        S_mat = S.tocoo()
        row_social, col_social, val_social = S_mat.row, S_mat.col, S_mat.data
        row_social, col_social, val_social = np.array(row_social).reshape(-1, 1), np.array(col_social).reshape(-1,
                                                                                                               1), np.array(
            val_social).reshape(-1, 1)
        ui_data = np.concatenate((row_ui, col_ui), axis=1)
        social_data = np.concatenate((row_social, col_social), axis=1)
        A_data = np.concatenate((ui_data, ui_data[:, ::-1], social_data, social_data[:, ::-1]), axis=0)
        val_data = np.concatenate((val_ui, val_ui, val_social, val_social), axis=0)

        A_mat = sps.csr_matrix((val_data[:, 0], (A_data[:, 0], A_data[:, 1])), shape=(n, n))
        return A_mat, user_num, item_num

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size)))})
        return embedding_dict

    def build_granular_tree(self, cluster_data):
        root = TreeNode("Root", is_center=True)  # Root node

        def add_cluster_to_tree(cluster, parent_node):
            center_value, qity = cluster[1]  # Center node value
            center_node = TreeNode(center_value, qity, is_center=True)
            parent_node.add_child(center_node)

            for sub_cluster in cluster[0]:  # Check for sub-clusters
                if isinstance(sub_cluster, tuple):  # If it's a tuple, it's a cluster
                    add_cluster_to_tree(sub_cluster, center_node)
                else:  # If it's a node, it's a leaf
                    value = sub_cluster[:-1]
                    _, qity = sub_cluster[-1]
                    leaf_node = TreeNode(value, qity, is_center=False)
                    center_node.add_child(leaf_node)

        for cluster in cluster_data:
            add_cluster_to_tree(cluster, root)

        return root

    def get_motif_adj_matrix(self, S, R):

        Y = R
        B = S.mul(S.t())
        U = S - B
        C1 = (torch.sparse.mm(U, (U))).mul(U.t())
        A1 = C1 + C1.t()
        C2 = (torch.sparse.mm(B, (U))).mul(U.t()) + (torch.sparse.mm(U, (B))).mul(U.t()) + (
            torch.sparse.mm(U, (U))).mul(B)
        A2 = C2 + C2.t()
        C3 = (torch.sparse.mm(B, (B))).mul(U) + (torch.sparse.mm(B, (U))).mul(B) + (torch.sparse.mm(U, (B))).mul(B)
        A3 = C3 + C3.t()
        A4 = (torch.sparse.mm(B, (B))).mul(B)
        C5 = (torch.sparse.mm(U, (U))).mul(U) + (torch.sparse.mm(U, (U.t()))).mul(U) + (
            torch.sparse.mm(U.t(), (U))).mul(U)
        A5 = C5 + C5.t()
        A6 = (torch.sparse.mm(U, (B))).mul(U) + (torch.sparse.mm(B, (U.t()))).mul(U.t()) + (
            torch.sparse.mm(U.t(), (U))).mul(B)
        A7 = (torch.sparse.mm(U.t(), (B))).mul(U.t()) + (torch.sparse.mm(B, (U))).mul(U) + (
            torch.sparse.mm(U, (U.t()))).mul(B)
        A8 = (torch.sparse.mm(Y, (Y.t()))).mul(B)
        A9 = (torch.sparse.mm(Y, (Y.t()))).mul(U)
        A9 = A9 + A9.t()
        A10 = torch.sparse.mm(Y, (Y.t())) - A8 - A9

        H_s = (A1 + A2 + A3 + A4 + A5 + A6 + A7)
        H_j = (A8 + A9)
        H_p = A10

        # add epsilon to avoid divide by zero Warning
        D_s = torch.sparse.sum(H_s, dim=1).to_dense()
        D_s[D_s == 0] = 1e-10
        Ds_inv = D_s.pow(-1)
        Ds_inv_sparse = torch.diag(Ds_inv)
        H_s = torch.sparse.mm(Ds_inv_sparse, H_s).to_sparse()

        D_j = torch.sparse.sum(H_j, dim=1).to_dense()
        D_j[D_j == 0] = 1e-10
        Dj_inv = D_j.pow(-1)
        Dj_inv_sparse = torch.diag(Dj_inv)
        H_j = torch.sparse.mm(Dj_inv_sparse, H_j).to_sparse()

        D_p = torch.sparse.sum(H_p, dim=1).to_dense()
        D_p[D_p == 0] = 1e-10
        Dp_inv = D_p.pow(-1)
        Dp_inv_sparse = torch.diag(Dp_inv)
        H_p = torch.sparse.mm(Dp_inv_sparse, H_p).to_sparse()

        return H_s, H_j, H_p

    def _cosine_similarity(self, embed):
        """
        """
        embed_norm = embed / embed.norm(dim=1)[:, None]
        cos_sim_matrix = torch.mm(embed_norm, embed_norm.t())

        return cos_sim_matrix

    def _social_rewiring(self, S):
        new_indices = []
        new_values = []
        all_indices = []
        center_cluster = {}
        cluster_indices = []
        for cluster, centers in self.GB_flatten:
            for center, qity in centers:
                if center == None:
                    continue
                if center not in center_cluster:
                    center_cluster[center] = [cluster]
                else:
                    center_cluster[center].append(cluster)
        for cluster, centers in tqdm.tqdm(self.GB_flatten):
            center_maxqity, _ = max(centers[1:-1], key=lambda x: x[1])
            user_in_cluster = set([nei for nei in cluster if nei < self.num_user])
            user_in_granular = []
            for cluster_in_granular in center_cluster[center_maxqity]:
                user_in_granular.extend([nei for nei in cluster_in_granular if nei < self.num_user])
            user_in_granular = set(user_in_granular)
            added_edges = list(combinations(user_in_granular, 2))
            added_edges = added_edges + [(edge[1], edge[0]) for edge in added_edges]
            added_edges = [edge for edge in added_edges if edge[0] in user_in_cluster]
            all_indices.extend(added_edges)
            cluster_indices.extend(added_edges)
            cluster_indices.extend([(node, node) for node in user_in_cluster])

        def add_edges(node):
            # print(" " * level + str(node.value))  # Print node value
            # GB_flatten.append((node.value, centers))
            if isinstance(node.value, int):
                for child in node.children:
                    if isinstance(child.value, int):
                        if node.value < self.num_user and child.value < self.num_user:
                            new_indices.append([node.value, child.value])
                            all_indices.append((node.value, child.value))
                            new_indices.append([child.value, node.value])
                            all_indices.append((child.value, node.value))
                            new_values.append(child.qity)
                            new_values.append(child.qity)
                    elif isinstance(child.value, list):
                        users_in_cluster = [nei for nei in child.value if nei < self.num_user]
                        cluster_size = len(users_in_cluster)
                        for value in child.value:
                            if node.value < self.num_user and value < self.num_user:
                                new_indices.append([node.value, value])
                                all_indices.append((node.value, value))
                                new_indices.append([value, node.value])
                                all_indices.append((value, node.value))
                                cur_homo = self.S[value, users_in_cluster].sum() / ((1 + cluster_size) ** 2)
                                new_values.append(cur_homo)
                                new_values.append(cur_homo)

            #    new_values.extend(added_values)
            for child in node.children:
                add_edges(child)

        # adding
        # top_values, top_indices = torch.topk(cos_sim_matrix.reshape(-1), int(cut_num))

        # rows = top_indices // cos_sim_matrix.size(1)
        # cols = top_indices % cos_sim_matrix.size(1)
        # new_indices = torch.stack((rows, cols))

        add_edges(self.granular_tree)
        org_indices = S._indices().t()
        # new_indices = new_indices.t()

        # Convert 2D indices to tuples for easy comparison
        org_indices_set = set([tuple(x) for x in org_indices.tolist()])
        all_indices_set = set([tuple(x) for x in all_indices])
        # Create the mask for new_indices
        mask_list = [tuple(x) in org_indices_set for x in new_indices]
        cut_mask_list = [tuple(x) in all_indices_set for x in org_indices.tolist()]
        cut_mask = torch.tensor(cut_mask_list).to(self.device)
        mask = torch.tensor(mask_list)
        new_values = torch.tensor(new_values, dtype=torch.float32)
        new_indices = torch.tensor(new_indices)
        new_value = new_values[~mask].t().to(self.device)
        new_indice = new_indices[~mask].t().to(self.device)
        # S_values=S._values()
        # S_values[cut_mask]=0
        S_rewired_values = torch.cat([S._values()[cut_mask], new_value])
        S_rewired_indices = torch.cat([S._indices()[:, cut_mask], new_indice], dim=1)

        self.cluster_S = torch.sparse.FloatTensor(torch.tensor(cluster_indices).t(), torch.ones(len(cluster_indices)),
                                                  S.shape).to(self.device)
        S_tmp = torch.sparse.FloatTensor(S_rewired_indices, S_rewired_values, S.shape)

        new_values = S_tmp._values().clone()

        # normalization
        norm_values = (new_values - new_values.min()) / (new_values.max() - new_values.min())

        weighted_S = torch.sparse.FloatTensor(S_tmp._indices(), norm_values, S_tmp.shape)

        D_s = torch.sparse.sum(weighted_S, dim=1).to_dense()
        D_s[D_s == 0] = 1e-10
        Ds_inv = D_s.pow(-1)
        Ds_inv_sparse = torch.diag(Ds_inv)
        weighted_S = torch.sparse.mm(Ds_inv_sparse, weighted_S).to_sparse()

        return weighted_S

    def forward(self):

        # ----------------------- initial embeddings -----------------------
        # initialize concatenated embeddings (users and items)
        U = self.embedding_dict['user_emb']
        V = self.embedding_dict['item_emb']
        E = torch.cat([U, V], dim=0)

        # ----------------------- SSL-Encoder ---------------------------
        # message propagation for each layer (both user and item phases)
        encoder_g_embeddings = E.clone()
        E_encoder = E.clone()
        for i in range(self.num_encoder_layer):
            E_encoder = self.encoder_layers[i](self.A_hat, E_encoder)
            encoder_g_embeddings = encoder_g_embeddings + E_encoder  # layer combination

        # average the sum of layers (a_k=1/K+1)
        encoder_out_embeddings = torch.div(encoder_g_embeddings, (self.num_encoder_layer + 1))
        self.U_encoder_out, _ = torch.split(encoder_out_embeddings, [self.data.user_num, self.data.item_num])

        user_embeddings = self.embedding_dict['user_emb']
        item_embeddings = self.embedding_dict['item_emb']

        # self-gating
        user_embeddings_c1 = self.gating_c1(user_embeddings)
        user_embeddings_c2 = self.gating_c2(user_embeddings)
        user_embeddings_c3 = self.gating_c3(user_embeddings)
        simple_user_embeddings = self.gating_simple(user_embeddings)

        all_embeddings_c1 = [user_embeddings_c1]
        all_embeddings_c2 = [user_embeddings_c2]
        all_embeddings_c3 = [user_embeddings_c3]
        all_embeddings_simple = [simple_user_embeddings]
        all_embeddings_i = [item_embeddings]

        for layer_idx in range(self.layers):
            mixed_embedding = self.attention_layer(user_embeddings_c1, user_embeddings_c2,
                                                   user_embeddings_c3) + simple_user_embeddings / 2

            # Channel S
            user_embeddings_c1 = torch.sparse.mm(self.H_s, user_embeddings_c1)
            norm_embeddings = F.normalize(user_embeddings_c1, p=2, dim=1)
            all_embeddings_c1 += [norm_embeddings]

            # Channel J
            user_embeddings_c2 = torch.sparse.mm(self.H_j, user_embeddings_c2)
            norm_embeddings = F.normalize(user_embeddings_c2, p=2, dim=1)
            all_embeddings_c2 += [norm_embeddings]

            # Channel P
            user_embeddings_c3 = torch.sparse.mm(self.H_p, user_embeddings_c3)
            norm_embeddings = F.normalize(user_embeddings_c3, p=2, dim=1)
            all_embeddings_c3 += [norm_embeddings]

            # item convolution
            new_item_embeddings = torch.sparse.mm(self.sparse_norm_adj.t(), mixed_embedding)
            norm_embeddings = F.normalize(new_item_embeddings, p=2, dim=1)
            all_embeddings_i += [norm_embeddings]
            simple_user_embeddings = torch.sparse.mm(self.sparse_norm_adj, item_embeddings)
            norm_embeddings = F.normalize(simple_user_embeddings, p=2, dim=1)
            all_embeddings_simple += [norm_embeddings]
            item_embeddings = new_item_embeddings

        # averaging the channel-specific embeddings
        user_embeddings_c1 = torch.stack(all_embeddings_c1, dim=0).sum(dim=0)
        user_embeddings_c2 = torch.stack(all_embeddings_c2, dim=0).sum(dim=0)
        user_embeddings_c3 = torch.stack(all_embeddings_c3, dim=0).sum(dim=0)
        simple_user_embeddings = torch.stack(all_embeddings_simple, dim=0).sum(dim=0)
        item_all_embeddings = torch.stack(all_embeddings_i, dim=0).sum(dim=0)

        # aggregating channel-specific embeddings
        user_all_embeddings = self.attention_layer(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3)
        user_all_embeddings += simple_user_embeddings / 2
        self.U = user_all_embeddings
        return user_all_embeddings, item_all_embeddings

    def _ssl_loss_v2(self, user_idx):
        all_embedd_sim = self._cosine_similarity(self.U_encoder_out)
        weighted_S = self.weighted_S.to_dense()[user_idx]
        weighted_S[list(range(len(user_idx))), user_idx] = 1
        pos_score = torch.matmul(weighted_S, torch.exp(all_embedd_sim / 0.1)).sum(dim=1)
        ttl_score = all_embedd_sim[user_idx]
        ttl_score = torch.exp(ttl_score / 0.1).sum(dim=1)
        ssl_loss = -torch.log(pos_score / ttl_score)
        ssl_loss = ssl_loss.mean()
        return ssl_loss

    def _ssl_loss(self, user_idx):
        all_embedd_sim = self._cosine_similarity(self.U_encoder_out)
        cluster_S = self.cluster_S.to_dense()[user_idx]
        pos_score = torch.matmul(cluster_S, torch.exp(all_embedd_sim / 0.1)).sum(dim=1)
        ttl_score = all_embedd_sim[user_idx]
        ttl_score = torch.exp(ttl_score / 0.1).sum(dim=1)
        ssl_loss = -torch.log(pos_score / ttl_score)
        ssl_loss = ssl_loss.mean()
        return ssl_loss


