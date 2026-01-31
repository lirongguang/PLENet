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
    def __init__(self):
        super(dataSet, self).__init__()
        self.config = DefaultConfig()
        self.SIGMA = 18.0

        self.neighbourhood_size = 21
        self.EDGE_FEATURES_ABSOLUTE_VALUED = False
        self.STANDARDIZE_EDGE_FEATURES = True
        self.STANDARDIZE_NODE_FEATURES = True
        save_dir = 'inputs/' + self.config.dataset_name

        with open(save_dir + "/proteins.txt", "r") as file:
            self.protein_name_list = [x.strip() for x in file.readlines()]
        
        self.ligand_name_list = self.protein_name_list
        self.receptor_name_list = self.protein_name_list

        with gzip.open(save_dir + '/l_ppisp_dist_matrix_map.pkl.gz', 'rb') as f:
            self.ligand_dist_matrix = pickle.load(f)

        with gzip.open(save_dir + '/r_ppisp_dist_matrix_map.pkl.gz', 'rb') as f:
            self.receptor_dist_matrix = pickle.load(f)

        with gzip.open(save_dir + '/l_ppisp_angle_matrix_map.pkl.gz', 'rb') as f:
            self.ligand_angle_matrix = pickle.load(f)

        with gzip.open(save_dir + '/r_ppisp_angle_matrix_map.pkl.gz', 'rb') as f:
            self.receptor_angle_matrix = pickle.load(f)

        with gzip.open(save_dir + '/all_labels.pkl.gz', 'rb') as f:
            self.all_labels = pickle.load(f)

        with gzip.open(save_dir + '/ESM_all_Receptor_features.pkl.gz', 'rb') as f:
            self.receptor_esm_features = pickle.load(f)

        with gzip.open(save_dir + '/ESM_all_Ligand_features.pkl.gz', 'rb') as f:
            self.ligand_esm_features = pickle.load(f)

        with gzip.open(save_dir + '/l_phychem_features.pkl.gz', 'rb') as f:
            self.ligand_phychem_features = pickle.load(f)

        with gzip.open(save_dir + '/r_phychem_features.pkl.gz', 'rb') as f:
            self.receptor_phychem_features = pickle.load(f)

        with gzip.open(save_dir + '/l_hydrophobicity_map.pkl.gz', 'rb') as f:
            self.ligand_hydrophobicity = pickle.load(f)

        with gzip.open(save_dir + '/r_hydrophobicity_map.pkl.gz', 'rb') as f:
            self.receptor_hydrophobicity = pickle.load(f)

        with gzip.open(save_dir + '/l_residue_accessibility.pkl.gz', 'rb') as f:
            self.ligand_residue_accessibility = pickle.load(f)

        with gzip.open(save_dir + '/r_residue_accessibility.pkl.gz', 'rb') as f:
            self.receptor_residue_accessibility = pickle.load(f)

        print("Loading residue coordinates...")
        try:
            with gzip.open(save_dir + '/l_residue_coordinates.pkl.gz', 'rb') as f:
                self.ligand_coordinates = pickle.load(f)
            print("Loaded ligand coordinates")
        except FileNotFoundError:
            print("Warning: l_residue_coordinates.pkl.gz not found, using empty coordinates")
            self.ligand_coordinates = {}

        try:
            with gzip.open(save_dir + '/r_residue_coordinates.pkl.gz', 'rb') as f:
                self.receptor_coordinates = pickle.load(f)
            print("Loaded receptor coordinates")
        except FileNotFoundError:
            print("Warning: r_residue_coordinates.pkl.gz not found, using empty coordinates")
            self.receptor_coordinates = {}

        self.graph_data = {}
        self.max_seq_len = self.config.max_sequence_length

        for i in range(len(self.protein_name_list)):
            p = self.protein_name_list[i]
            l = self.ligand_name_list[i]
            r = self.receptor_name_list[i]

            label_mask = (self.all_labels[p][:, 0] >= 0) & (self.all_labels[p][:, 0] < self.max_seq_len) & \
                         (self.all_labels[p][:, 1] >= 0) & (self.all_labels[p][:, 1] < self.max_seq_len)
            new_labels = self.all_labels[p][label_mask]
            pos = new_labels[new_labels[:, 2] == 1]
            if len(pos) == 0:
                continue
            max_seq_labels = self.all_labels[p][label_mask]

            l_coords = self._process_coordinates(self.ligand_coordinates.get(l, {}), self.max_seq_len)
            r_coords = self._process_coordinates(self.receptor_coordinates.get(r, {}), self.max_seq_len)

            self.graph_data[p] = {
                "l_esm_features": self.ligand_esm_features[l],
                "r_esm_features": self.receptor_esm_features[r],
                "l_hydrophobicity": self.ligand_hydrophobicity[l],
                "r_hydrophobicity": self.receptor_hydrophobicity[r],
                "l_residue_accessibility": self.ligand_residue_accessibility[l],
                "r_residue_accessibility": self.receptor_residue_accessibility[r],
                "l_phychem_features": self.ligand_phychem_features[l],
                "r_phychem_features": self.receptor_phychem_features[r],
                "l_coordinates": l_coords,
                "r_coordinates": r_coords,
                'label': max_seq_labels
            }

        self.protein_name_list = list(self.graph_data.keys())
        self.ligand_name_list = self.protein_name_list
        self.receptor_name_list = self.protein_name_list
        self.protein_list_len = len(self.protein_name_list)

        self.generate_all_graphs(prot='r')
        self.generate_all_graphs(prot='l')
        print('All graphs generated:')

        all_protein = tuple((list(self.graph_data.keys()), list(self.graph_data.values())))
        self.train_proteins = tuple((all_protein[0][:self.config.train_size + self.config.val_size],
                                     all_protein[1][:self.config.train_size + self.config.val_size]))
        self.test_proteins = tuple((all_protein[0][self.config.train_size+self.config.val_size:], 
                                   all_protein[1][self.config.train_size+self.config.val_size:]))

        pickle.dump(self.train_proteins, gzip.open(save_dir + '/train.pkl.gz', 'wb'))
        pickle.dump(self.test_proteins, gzip.open(save_dir + '/test.pkl.gz', 'wb'))

        print('total train', len(self.train_proteins[0]), 'total test', len(self.test_proteins[0]))

        esm_dim = 960  
        esm_mean, esm_std = self.generate_node_mean(self.receptor_esm_features, self.ligand_esm_features, esm_dim)
        phychem_mean, phychem_std = self.generate_node_mean(self.receptor_phychem_features,
                                                            self.ligand_phychem_features, 14)
        hydrophobicity_mean, hydrophobicity_std = self.generate_node_mean(self.receptor_hydrophobicity,
                                                                          self.ligand_hydrophobicity, 1)
        residue_accessibility_mean, residue_accessibility_std = self.generate_node_mean(
            self.receptor_residue_accessibility, self.ligand_residue_accessibility, 2)
        
        coordinates_mean, coordinates_std = self.generate_coordinates_mean()
        
        edge_mean, edge_std = self.generate_edge_mean()

        self.mean_std = {
            'esm_mean': esm_mean,
            'esm_std': esm_std,
            'phychem_mean': phychem_mean,
            'phychem_std': phychem_std,
            'hydrophobicity_mean': hydrophobicity_mean,
            'hydrophobicity_std': hydrophobicity_std,
            'residue_accessibility_mean': residue_accessibility_mean,
            'residue_accessibility_std': residue_accessibility_std,
            'coordinates_mean': coordinates_mean,
            'coordinates_std': coordinates_std,
            'edge_mean': edge_mean,
            'edge_std': edge_std
        }
        pickle.dump(self.mean_std, gzip.open(save_dir + '/node_and_edge_mean_std.pkl.gz', 'wb'))

    def _process_coordinates(self, coord_data, max_seq_len):

        if not coord_data or 'coordinates' not in coord_data:
            return np.zeros((max_seq_len, 3))
        
        coordinates = coord_data['coordinates']
        
        if len(coordinates) == 0:
            return np.zeros((max_seq_len, 3))
        
        if len(coordinates) > max_seq_len:
            coordinates = coordinates[:max_seq_len]
        
        elif len(coordinates) < max_seq_len:
            padding = np.zeros((max_seq_len - len(coordinates), 3))
            coordinates = np.vstack([coordinates, padding])
        
        return coordinates

    def generate_coordinates_mean(self):

        print("Computing coordinates mean and std...")
        
        all_coordinates = []
        
        for protein_data in self.train_proteins[1]:
            l_coords = protein_data['l_coordinates']
            r_coords = protein_data['r_coordinates']
            
            l_nonzero_mask = np.any(l_coords != 0, axis=1)
            r_nonzero_mask = np.any(r_coords != 0, axis=1)
            
            if np.any(l_nonzero_mask):
                all_coordinates.append(l_coords[l_nonzero_mask])
            if np.any(r_nonzero_mask):
                all_coordinates.append(r_coords[r_nonzero_mask])
        
        if len(all_coordinates) == 0:
            print("Warning: No valid coordinates found, using zero mean and unit std")
            return np.zeros(3), np.ones(3)
        
        all_coords = np.vstack(all_coordinates)
        
        mean = np.mean(all_coords, axis=0)
        std = np.std(all_coords, axis=0)
        
        std[std == 0] = 1.0
        
        print(f"Coordinates mean: {mean}")
        print(f"Coordinates std: {std}")
        
        return mean, std

    def __len__(self):
        return self.protein_list_len

    def generate_all_graphs(self, prot):

        for i, protein in enumerate(self.protein_name_list):
            protein = self.protein_name_list[i]
            l = self.ligand_name_list[i]
            r = self.receptor_name_list[i]

            print('Generating graphs for', protein, prot)
            if prot == 'l':
                neighborhood_indices = self.ligand_dist_matrix[l] \
                                           [:self.max_seq_len, :self.max_seq_len, 0].argsort()[:,
                                       1:self.neighbourhood_size]

                self.graph_data[protein][prot + '_hood_indices'] = neighborhood_indices

                if neighborhood_indices.max() > self.max_seq_len - 1 or neighborhood_indices.min() < 0:
                    print(prot + '_neighbourhood_indices value error')
                    print(neighborhood_indices.max(), neighborhood_indices.min())
                    raise

                dist = self.ligand_dist_matrix[l][:self.max_seq_len, :self.max_seq_len, 0]
                angle = self.ligand_angle_matrix[l][:self.max_seq_len, :self.max_seq_len]

                dist = np.array([dist[i, neighborhood_indices[i]] for i in range(dist.shape[0])])
                angle = np.array([angle[i, neighborhood_indices[i]] for i in range(angle.shape[0])])

                # pass through a gaussian function : f(x) = e^(-x^2 / sigma^2)
                # sigma = 18 (from pipcgn)

                dist = np.e ** (-np.square(dist) / self.SIGMA ** 2)
                edge_feat = np.array([dist, angle])
                edge_feat = np.transpose(edge_feat, (1, 2, 0))

                self.graph_data[protein][prot + '_edge'] = edge_feat

            else:
                neighborhood_indices = self.receptor_dist_matrix[r] \
                                           [:self.max_seq_len, :self.max_seq_len, 0].argsort()[:,
                                       1:self.neighbourhood_size]

                self.graph_data[protein][prot + '_hood_indices'] = neighborhood_indices

                if neighborhood_indices.max() > self.max_seq_len - 1 or neighborhood_indices.min() < 0:
                    print(prot + '_neighbourhood_indices value error')
                    print(neighborhood_indices.max(), neighborhood_indices.min())
                    raise

                dist = self.receptor_dist_matrix[r][:self.max_seq_len, :self.max_seq_len, 0]
                angle = self.receptor_angle_matrix[r][:self.max_seq_len, :self.max_seq_len]

                dist = np.array([dist[i, neighborhood_indices[i]] for i in range(dist.shape[0])])
                angle = np.array([angle[i, neighborhood_indices[i]] for i in range(angle.shape[0])])

                # pass through a gaussian function : f(x) = e^(-x^2 / sigma^2)
                # sigma = 18 (from pipcgn)

                dist = np.e ** (-np.square(dist) / self.SIGMA ** 2)
                edge_feat = np.array([dist, angle])
                edge_feat = np.transpose(edge_feat, (1, 2, 0))
                self.graph_data[protein][prot + '_edge'] = edge_feat

        return

    def generate_node_mean(self, receptor_features, ligand_features, dimension=1280):

        n = 0
        mean = np.zeros([dimension])
        std = np.zeros([dimension])
        for k in receptor_features:
            mean += sum(receptor_features[k])
            n += receptor_features[k].shape[0]
        for k in ligand_features:
            mean += sum(ligand_features[k])
            n += ligand_features[k].shape[0]
        mean /= n

        for k in receptor_features:
            temp = np.square(receptor_features[k] - mean)
            std += sum(temp)
        for k in ligand_features:
            temp = np.square(ligand_features[k] - mean)
            std += sum(temp)

        std = np.sqrt(std / n)

        return mean, std


    def generate_edge_mean(self):
        edge_features = self.train_proteins[1] + self.test_proteins[1]
        
        r_edges = []
        l_edges = []
        
        for k in range(len(edge_features)):
            r_edge = edge_features[k]['r_edge']
            l_edge = edge_features[k]['l_edge']
            
            print(f"Original r_edge shape: {r_edge.shape}")
            
            if r_edge.shape[1] < 20:
                padding = np.zeros((r_edge.shape[0], 20 - r_edge.shape[1], r_edge.shape[2]))
                r_edge = np.concatenate([r_edge, padding], axis=1)
                print(f"Padded r_edge shape: {r_edge.shape}")
            
            if l_edge.shape[1] < 20:
                padding = np.zeros((l_edge.shape[0], 20 - l_edge.shape[1], l_edge.shape[2]))
                l_edge = np.concatenate([l_edge, padding], axis=1)
            
            r_edges.append(r_edge)
            l_edges.append(l_edge)
        
        r = np.vstack(r_edges)
        l = np.vstack(l_edges)
        lr = np.vstack([l, r])
        dimension = lr.shape[2]

        mean = np.zeros([dimension])
        std = np.zeros([dimension])

        for i in range(dimension):
            mean[i] = np.mean(lr[:, :, i])
            std[i] = np.std(lr[:, :, i])

        return mean, std


if __name__ == '__main__':
    data = dataSet()
