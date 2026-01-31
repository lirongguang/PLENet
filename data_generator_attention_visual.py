import os
import pickle
import torch as t
import numpy as np
from torch.utils import data
import gzip
from time import time
from config import DefaultConfig
import torch
import dgl
import threading


class dataSet(data.Dataset):
    def __init__(self, root_dir, data_file_prefix='train'):
        super(dataSet, self).__init__()

        self.config = DefaultConfig()
        self.data_file_prefix = data_file_prefix

        with gzip.open(
            root_dir + '/inputs/' + self.config.dataset_name + '/{}.pkl.gz'.format(data_file_prefix),
            'rb'
        ) as f:
            self.dataset = pickle.load(f)
            self.protein_name_list = self.dataset[0]
            self.protein_data = self.dataset[1]

        with gzip.open(
            root_dir + '/inputs/' + self.config.dataset_name + '/node_and_edge_mean_std.pkl.gz',
            'rb'
        ) as f:
            mean_and_std = pickle.load(f)

            # Use ESM embeddings instead of BERT and XLNet embeddings
            self.node_feat_mean = mean_and_std['esm_mean']
            self.node_feat_std = mean_and_std['esm_std']

            if self.config.num_phychem_features == 1:
                self.phychem_feat_mean = mean_and_std['hydrophobicity_mean']
                self.phychem_feat_std = mean_and_std['hydrophobicity_std']
            else:
                self.phychem_feat_mean = mean_and_std['phychem_mean']
                self.phychem_feat_std = mean_and_std['phychem_std']

            self.residue_accessibility_mean = mean_and_std['residue_accessibility_mean']
            self.residue_accessibility_std = mean_and_std['residue_accessibility_std']

            self.edge_feat_mean = mean_and_std['edge_mean']
            self.edge_feat_std = mean_and_std['edge_std']

        self.max_seq_len = self.config.max_sequence_length
        self.protein_list_len = len(self.protein_name_list)

        self.all_graphs_receptor = self.generate_all_graphs(prot='r')
        self.all_graphs_ligand = self.generate_all_graphs(prot='l')

        for index in range(len(self.protein_name_list)):
            label = self.protein_data[index]['label']

            pos = label[label[:, 2] == 1]
            neg = label[label[:, 2] != 1]

            if self.data_file_prefix.startswith('train'):
                np.random.shuffle(neg)
                label = np.vstack([pos, neg[: self.config.pos_neg_ratio * len(pos)]])
                np.random.shuffle(label)
            self.protein_data[index]['label'] = label

    def __getitem__(self, index):

        complex_name = self.protein_name_list[index]
        complex_info = {
            'complex_name': complex_name,
            'complex_idx': index,
        }

        # Use ESM features instead of BERT and XLNet features
        _protESM_feature_receptor_ = self.protein_data[index]['r_esm_features']
        _protESM_feature_receptor_ = _protESM_feature_receptor_[:self.max_seq_len]

        phychem_features_receptor = (
            self.protein_data[index]['r_hydrophobicity']
            if self.config.num_phychem_features == 1
            else self.protein_data[index]['r_phychem_features']
        )
        phychem_features_receptor = phychem_features_receptor[:self.max_seq_len]
        residue_accessibility_receptor = self.protein_data[index]['r_residue_accessibility'][:self.max_seq_len]

        receptor_coordinates = self.protein_data[index].get(
            'r_coordinates', np.zeros((self.max_seq_len, 3))
        )
        ligand_coordinates = self.protein_data[index].get(
            'l_coordinates', np.zeros((self.max_seq_len, 3))
        )

        # Ensure coordinate shapes are aligned
        receptor_coordinates = receptor_coordinates[:self.max_seq_len]
        ligand_coordinates = ligand_coordinates[:self.max_seq_len]

        if receptor_coordinates.shape[0] < self.max_seq_len:
            pad = np.zeros((self.max_seq_len - receptor_coordinates.shape[0], 3))
            receptor_coordinates = np.vstack([receptor_coordinates, pad])

        if ligand_coordinates.shape[0] < self.max_seq_len:
            pad = np.zeros((self.max_seq_len - ligand_coordinates.shape[0], 3))
            ligand_coordinates = np.vstack([ligand_coordinates, pad])

        seq_len = _protESM_feature_receptor_.shape[0]
        complex_info['receptor_seq_length'] = seq_len

        if seq_len < self.max_seq_len:
            temp = np.zeros([self.max_seq_len, _protESM_feature_receptor_.shape[1]])
            temp[:seq_len, :] = _protESM_feature_receptor_
            _protESM_feature_receptor_ = temp

            temp = np.zeros([self.max_seq_len, phychem_features_receptor.shape[1]])
            temp[:seq_len, :] = phychem_features_receptor
            phychem_features_receptor = temp

            temp = np.zeros([self.max_seq_len, residue_accessibility_receptor.shape[1]])
            temp[:seq_len, :] = residue_accessibility_receptor
            residue_accessibility_receptor = temp

        _protESM_feature_receptor_ = _protESM_feature_receptor_[np.newaxis, :, :]
        phychem_features_receptor = phychem_features_receptor[np.newaxis, :, :]
        residue_accessibility_receptor = residue_accessibility_receptor[np.newaxis, :, :]
        G_receptor = self.all_graphs_receptor[index]

        _protESM_feature_ligand_ = self.protein_data[index]['l_esm_features']
        _protESM_feature_ligand_ = _protESM_feature_ligand_[:self.max_seq_len]

        phychem_features_ligand = (
            self.protein_data[index]['l_hydrophobicity']
            if self.config.num_phychem_features == 1
            else self.protein_data[index]['l_phychem_features']
        )
        phychem_features_ligand = phychem_features_ligand[:self.max_seq_len]
        residue_accessibility_ligand = self.protein_data[index]['l_residue_accessibility'][:self.max_seq_len]

        seq_len = _protESM_feature_ligand_.shape[0]
        complex_info['ligand_seq_length'] = seq_len

        if seq_len < self.max_seq_len:
            temp = np.zeros([self.max_seq_len, _protESM_feature_ligand_.shape[1]])
            temp[:seq_len, :] = _protESM_feature_ligand_
            _protESM_feature_ligand_ = temp

            temp = np.zeros([self.max_seq_len, phychem_features_ligand.shape[1]])
            temp[:seq_len, :] = phychem_features_ligand
            phychem_features_ligand = temp

            temp = np.zeros([self.max_seq_len, residue_accessibility_ligand.shape[1]])
            temp[:seq_len, :] = residue_accessibility_ligand
            residue_accessibility_ligand = temp

        _protESM_feature_ligand_ = _protESM_feature_ligand_[np.newaxis, :, :]
        phychem_features_ligand = phychem_features_ligand[np.newaxis, :, :]
        residue_accessibility_ligand = residue_accessibility_ligand[np.newaxis, :, :]
        G_ligand = self.all_graphs_ligand[index]

        if self.config.STANDARDIZE_NODE_FEATURES:
            _protESM_feature_receptor_ = (_protESM_feature_receptor_ - self.node_feat_mean) / self.node_feat_std
            _protESM_feature_ligand_ = (_protESM_feature_ligand_ - self.node_feat_mean) / self.node_feat_std
            phychem_features_receptor = (phychem_features_receptor - self.phychem_feat_mean) / self.phychem_feat_std
            phychem_features_ligand = (phychem_features_ligand - self.phychem_feat_mean) / self.phychem_feat_std
            residue_accessibility_receptor = (
                residue_accessibility_receptor - self.residue_accessibility_mean
            ) / self.residue_accessibility_std
            residue_accessibility_ligand = (
                residue_accessibility_ligand - self.residue_accessibility_mean
            ) / self.residue_accessibility_std

        label = self.protein_data[index]['label']
        label[label == -1] = 0

        positives = label[label[:, 2] == 1]
        pos_ligands = set(positives[:, 0])
        pos_receptors = set(positives[:, 1])

        ligand_labels = np.array([x[0] in pos_ligands for x in label], dtype=int)
        receptor_labels = np.array([x[1] in pos_receptors for x in label], dtype=int)

        complex_info['total_usable_residues'] = label.shape[0]

        return (
            torch.from_numpy(_protESM_feature_receptor_).type(torch.FloatTensor),
            G_receptor,
            torch.from_numpy(phychem_features_receptor).type(torch.FloatTensor),
            torch.from_numpy(residue_accessibility_receptor).type(torch.FloatTensor),
            torch.from_numpy(_protESM_feature_ligand_).type(torch.FloatTensor),
            G_ligand,
            torch.from_numpy(phychem_features_ligand).type(torch.FloatTensor),
            torch.from_numpy(residue_accessibility_ligand).type(torch.FloatTensor),
            complex_info,
            label,
            ligand_labels,
            receptor_labels,
            torch.from_numpy(receptor_coordinates).type(torch.FloatTensor),
            torch.from_numpy(ligand_coordinates).type(torch.FloatTensor)  # Newly added coordinates
        )

    def __len__(self):
        return self.protein_list_len

    def generate_all_graphs(self, prot):
        graph_list = {}
        M = self.config.neighbourhood_size - 1  # Expected number of neighbor columns

        for id_idx in range(self.protein_list_len):
            G = dgl.DGLGraph()
            G.add_nodes(self.max_seq_len)

            # Original neighborhood index array
            raw = self.protein_data[id_idx][prot + '_hood_indices']
            n_rows, current_len = raw.shape

            # Pad or truncate columns to match M
            if current_len < M:
                pad = np.zeros((n_rows, M - current_len), dtype=raw.dtype)
                neighborhood_indices = np.hstack([raw, pad])
            elif current_len > M:
                neighborhood_indices = raw[:, :M]
            else:
                neighborhood_indices = raw

            # neighborhood_indices now has shape (n_rows, M)
            edge_feat = self.protein_data[id_idx][prot + '_edge']
            if self.config.STANDARDIZE_EDGE_FEATURES:
                edge_feat = (edge_feat - self.edge_feat_mean) / self.edge_feat_std

            self.add_edges_custom(G, neighborhood_indices, edge_feat)
            graph_list[id_idx] = G

        return graph_list

    def add_edges_custom(self, G, neighborhood_indices, edge_features):
        t1 = time()
        size = min(neighborhood_indices.shape[0], self.max_seq_len)

        src = []
        dst = []
        temp_edge_features = []

        for center in range(size):

            # Keep only residues within max_seq_len
            mask = neighborhood_indices[center] < self.max_seq_len
            neighbors = neighborhood_indices[center][mask]

            src += neighbors.tolist()
            dst += [center] * len(neighbors)

            # Keep only valid edge features
            feats = edge_features[center]
            current_len = feats.shape[0]
            feat_dim = feats.shape[1]

            if current_len < 20:
                padding = np.zeros((20 - current_len, feat_dim), dtype=feats.dtype)
                feats = np.vstack([feats, padding])

            temp_edge_features += feats[mask].tolist()

        if len(src) != len(dst):
            print(
                'Source (src) and destination (dst) must have the same length: '
                'len(src)={} and len(dst)={}'.format(len(src), len(dst))
            )
            raise Exception

        G.add_edges(src, dst)
        G.edata['ex'] = torch.from_numpy(np.array(temp_edge_features).astype(np.float32))


class GraphCollate(object):
    def __init__(self, loader_name='train'):
        self.loader_name = loader_name

    def __call__(self, samples):
        configs = DefaultConfig()

        # Updated variable names to reflect ESM features
        (
            protesm_data_receptor,
            graph_batch_receptor,
            phychem_feat_receptor,
            residue_accessibility_receptor,
            protesm_data_ligand,
            graph_batch_ligand,
            phychem_feat_ligand,
            residue_accessibility_ligand,
            complex_info_batch,
            label_batch,
            ligand_label_batch,
            receptor_label_batch,
            receptor_coords_batch,
            ligand_coords_batch,
        ) = map(list, zip(*samples))

        protesm_data_receptor = torch.cat(protesm_data_receptor)
        graph_batch_receptor = dgl.batch(graph_batch_receptor)
        phychem_feat_receptor = torch.cat(phychem_feat_receptor)
        residue_accessibility_receptor = torch.cat(residue_accessibility_receptor)

        protesm_data_ligand = torch.cat(protesm_data_ligand)
        graph_batch_ligand = dgl.batch(graph_batch_ligand)
        phychem_feat_ligand = torch.cat(phychem_feat_ligand)
        residue_accessibility_ligand = torch.cat(residue_accessibility_ligand)

        # New: concatenate coordinates into tensors of shape (batch, L, 3)
        receptor_coords_batch = torch.cat(receptor_coords_batch)
        ligand_coords_batch = torch.cat(ligand_coords_batch)

        pair_residue_label = np.zeros([0, 3])
        ligand_label = np.zeros(0)
        receptor_labels = np.zeros(0)

        for i, temp in enumerate(label_batch):
            # Offset residue indices to align padded sequences
            temp[:, :2] = temp[:, :2] + i * configs.max_sequence_length
            pair_residue_label = np.concatenate([pair_residue_label, temp])
            ligand_label = np.concatenate([ligand_label, ligand_label_batch[i]])
            receptor_labels = np.concatenate([receptor_labels, receptor_label_batch[i]])

        return (
            protesm_data_receptor,
            graph_batch_receptor,
            phychem_feat_receptor,
            residue_accessibility_receptor,
            protesm_data_ligand,
            graph_batch_ligand,
            phychem_feat_ligand,
            residue_accessibility_ligand,
            complex_info_batch,
            torch.from_numpy(pair_residue_label).type(torch.LongTensor),
            torch.from_numpy(ligand_label).type(torch.LongTensor),
            torch.from_numpy(receptor_labels).type(torch.LongTensor),
            receptor_coords_batch,
            ligand_coords_batch,
        )
