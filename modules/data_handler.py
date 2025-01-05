import os
import dgl
import torch
import numpy as np
import scipy.sparse as sp

from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any

from utils.utils import load_data, get_target_neighbor_size, DATA_ROOT_DIR, DIR_NAME_DICT

class DataHandler():
    def __init__(self, config: Dict[str, Any]) -> None:
        '''
        A data handler class for (multi-target) graph injection attack experiments.

        Args:
            config (Dict[str, Any]): A dictionary of arguments. Refer to the 'template.json' file.
        '''

        # Arguments for DataHandler.
        self.multi_target = config['multi_target'] # If it sets to True, target sets will be considered. Else, nodes will be considered. Defaults to True.
        self.data_name = config['dataset']
        self.split = config['split']
        self.batch_size = config['batch_size']
        self.device = torch.device(config['cuda_id']) if torch.cuda.is_available() else torch.device("cpu")
        
        # Load the dataset and the split
        print(f"Start loading {self.data_name}-{self.split}")
        data = load_data(self.data_name, self.split, multi_target=self.multi_target)
        graph = data["graph"].to(self.device)
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        graph.ndata["degree"] = graph.in_degrees().to(self.device, torch.float32) - 1 # -1 for self-loops
        
        self.graph: dgl.DGLGraph = graph
        self.discrete = data["discrete"]
        self.num_nodes = graph.num_nodes()
        self.num_relations = len(graph.etypes)
        self.features = graph.ndata["x"] # ! Normalize, if necessary (Currently not normalized in this framework)
        self.feat_dim = self.features.shape[1]
        self.labels = graph.ndata["y"]
        self.num_classes = int(self.labels.max()) + 1
        self.idx_train = data["idx_train"] # indices of nodes in training set
        self.idx_valid = data["idx_valid"] # indices of nodes in validation set
        self.idx_test = data["idx_test"] # indices of nodes in test set
        
        if self.multi_target:
            self.target_assignment = data["target_assignment"] # target assignment matrix with the shape (num_target_sets, num_nodes)
            self.target_idx_train = data["target_idx_train"] # indices of target sets in training set
            self.target_idx_valid = data["target_idx_valid"] # indices of target sets in validation set
            self.target_idx_test = data["target_idx_test"] # indices of target sets in test set

            self.rho = config['node_budget']
            self.xi = config['edge_budget']
            self.node_budgets = self.calculate_node_budgets()
            self.edge_budgets = self.calculate_edge_budgets()
            
        else:
            self.node_budget = config['node_budget']
            self.edge_budget = config['edge_budget']
        
        # For GAGA, we need group aggregation sequences
        # Load (or calculate) group aggregation sequences and elements to calculate the sequences
        if "GAGA" in [config['surrogate_model'], config['victim_model']]:
            mask_train = torch.zeros(self.graph.num_nodes(), dtype=int, device=self.device)
            mask_train[self.idx_train] = 1
            
            self.adj = sp.csr_matrix(self.graph.adj_external(scipy_fmt='coo')).T # (N, N)
            self.adj.setdiag(0)
            self.graph.ndata['train_neg'] = (1 - self.labels) * mask_train
            self.graph.ndata['train_pos'] = self.labels * mask_train
            self.graph.ndata['train_masked'] = (1 - mask_train)
            
            sequences, feat_sum, d, adj2 = self.get_group_agg_sequences()

            self.sequences = sequences
            self.feat_sum = feat_sum
            self.d = d
            self.adj2 = adj2
        
        print(f"Finished loading {self.data_name}-{self.split}")

    def calculate_node_budgets(self) -> np.ndarray:
        '''
        Return numpy array which contains node budget for each target set
        (For multi-target setting only)
        '''
        # Get $B$ for each target set
        target_neighbor_size = get_target_neighbor_size(self.graph, self.target_assignment)
        upper_bound = target_neighbor_size.mean() # the upper bound of node budgets is the mean of $B$
        
        # Calculate node budgets based on $B$
        node_budgets = target_neighbor_size
        node_budgets[node_budgets > upper_bound] = upper_bound
        node_budgets = (self.rho * node_budgets + 0.5).astype(int)
        node_budgets[node_budgets < 1] = 1

        return node_budgets

    def calculate_edge_budgets(self) -> np.ndarray:
        '''
        Return numpy array which contains edge budget for each target set
        (For multi-target setting only)
        '''
        graph = self.graph
        degrees = graph.ndata["degree"].to(torch.float64)
        groups, nodes = self.target_assignment.nonzero()

        degree_budgets = torch.zeros(self.node_budgets.shape, dtype=float, device=self.device)
        degree_budgets.index_add_(0, torch.tensor(groups, dtype=int, device=self.device),
                                  degrees[nodes])
       
        group_sizes = torch.tensor(self.target_assignment.sum(axis=1), device=self.device).squeeze(-1)
        degree_budgets = (degree_budgets / group_sizes).cpu().numpy()
        
        upper_bound = self.xi * np.mean(degrees.cpu().numpy())
        
        degree_budgets[degree_budgets > upper_bound] = upper_bound
        degree_budgets = (degree_budgets + 0.5).astype(int)
        
        degree_budgets[degree_budgets < 1] = 1
        edge_budgets = self.node_budgets * degree_budgets
        
        return edge_budgets

    def get_train_set(self) -> np.ndarray:
        '''
        Return indices of target sets (if multi_target) or nodes (if not multi_target) in training set
        '''
        return self.target_idx_train if self.multi_target else self.idx_train

    def get_train_loader(self) -> DataLoader:
        '''
        Return DataLoader for training set
        '''
        t_ids = torch.LongTensor(self.get_train_set())
        node_budgets = torch.LongTensor(self.node_budgets[t_ids]) if self.multi_target else torch.ones_like(t_ids) * self.node_budget
        edge_budgets = torch.LongTensor(self.edge_budgets[t_ids]) if self.multi_target else torch.ones_like(t_ids) * self.edge_budget
        torch_dataset = TensorDataset(t_ids, node_budgets, edge_budgets)
        train_loader = DataLoader(dataset=torch_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        return train_loader
    
    def get_validation_set(self) -> np.ndarray:
        '''
        Return indices of target sets (if multi_target) or nodes (if not multi_target) in validation set
        '''
        return self.target_idx_valid if self.multi_target else self.idx_valid
    
    def get_test_set(self) -> np.ndarray:
        """
        Return indices of target sets (if multi_target) or nodes (if not multi_target) in test set
        """
        return self.target_idx_test if self.multi_target else self.idx_test
    
    
    ################### For GAGA ###################
    
    def get_group_agg_sequences(self, root_dir=DATA_ROOT_DIR):
        
        dir_name = DIR_NAME_DICT[self.data_name]
        dir_path = os.path.join(root_dir, dir_name, "data_split")
        seq_path = os.path.join(dir_path, f"{self.data_name}_gaga_sequence_{self.split}.npy")
        feat_sum_path = os.path.join(dir_path, f"{self.data_name}_gaga_feat_sum_{self.split}.npy")
        degree_path = os.path.join(dir_path, f"{self.data_name}_gaga_degree_{self.split}.npy")
        adj2_path = os.path.join(dir_path, f"{self.data_name}_gaga_adj2_{self.split}.npz")

        if os.path.exists(seq_path) and \
           os.path.exists(feat_sum_path) and \
           os.path.exists(degree_path) and \
           os.path.exists(adj2_path):
               
            sequence = torch.tensor(np.load(seq_path))
            feat_sum = torch.tensor(np.load(feat_sum_path))
            d = torch.tensor(np.load(degree_path))
            adj2 = sp.load_npz(adj2_path)
        
        else:
            print("No sequence detected. Generating group aggregation sequences ...")
            feat_sum, d, adj2 = self.calculate_group_agg_sequences(self.graph, self.graph.nodes())
            
            feat_sum = feat_sum.reshape((-1, 7, self.feat_dim))
            d = d.reshape((-1, 7, 1))
            mask = d > 0
            denom = torch.ones_like(d)
            denom[mask] = d[mask]
            
            sequence = feat_sum / denom

            np.save(seq_path, sequence.cpu().numpy())
            np.save(feat_sum_path, feat_sum.cpu().numpy())
            np.save(degree_path, d.cpu().numpy())
            sp.save_npz(adj2_path, adj2)
            
            print("Done !")
        
        sequence = sequence.to(dtype=torch.float32)
        feat_sum = feat_sum.to(device="cpu", dtype=torch.float32)
        d = d.to(device="cpu", dtype=torch.float32)
        
        return sequence, feat_sum, d, adj2
    
    def calculate_group_agg_sequences(self, graph: dgl.DGLGraph, n_ids: torch.Tensor):
        
        graph, n_ids_sub = dgl.khop_in_subgraph(graph, n_ids, k=2, output_device=self.device)
        graph = dgl.remove_self_loop(graph)
        features = graph.ndata['x']
        train_neg = graph.ndata['train_neg'].cpu()
        train_pos = graph.ndata['train_pos'].cpu()
        train_masked = graph.ndata['train_masked'].cpu()
        
        n_ids_sub = n_ids_sub.cpu().numpy()
        
        # 1-hop groups
        adj = sp.csr_matrix(graph.adj_external(scipy_fmt='coo')).T # (N, N)
        adj.setdiag(0)
        feat_sum_1, d_1 = self.group_aggregation(adj, features, n_ids_sub,
                                                 train_neg, train_pos, train_masked)
        
        # 2-hop groups
        adj2 = (adj@adj).astype(bool).astype(int)
        adj2.setdiag(0)
        feat_sum_2, d_2 = self.group_aggregation(adj2, features, n_ids_sub,
                                               train_neg, train_pos, train_masked)

        feat_sum = torch.concat([features[n_ids_sub], feat_sum_1, feat_sum_2], dim=-1)
        d = torch.concat([torch.ones((len(n_ids), 1), device=self.device), d_1, d_2], dim=-1)

        return feat_sum, d, adj2
    
    def group_aggregation(self, adj: sp.csr_matrix, features: torch.Tensor,
                          n_ids, train_neg, train_pos, train_masked):
        
        adj = adj[n_ids]
        
        adj_neg = adj.multiply(train_neg)
        adj_pos = adj.multiply(train_pos)
        adj_masked = adj.multiply(train_masked)
        
        feat_sum_neg, d_neg = self.feat_aggregation(adj_neg, features)
        feate_sum_pos, d_pos = self.feat_aggregation(adj_pos, features)
        feat_sum_masked, d_masked = self.feat_aggregation(adj_masked, features)
        
        # Target Masking & Self-loop
        feat_sum_masked += features[n_ids]
        d_masked += 1
        
        feat_sum_hop = torch.concat([feat_sum_neg, feate_sum_pos, feat_sum_masked], dim=-1)
        d_hop = torch.concat([d_neg, d_pos, d_masked], dim=-1)        
        
        return feat_sum_hop, d_hop
    
    def feat_aggregation(self, adj: sp.spmatrix, features: torch.Tensor) -> torch.Tensor:
        
        features = features.cpu()
        device = features.device
        
        src, dst = adj.nonzero()
        src = torch.tensor(src, dtype=int, device=device)
        dst = torch.tensor(dst, dtype=int, device=device)
        
        # Get mean features
        shape = (adj.shape[0], features.shape[-1])
        feat_sum = torch.zeros(shape, device=device)
        
        # To avoid OOM
        CHUNK_SIZE = 2 ** 16
        num_chunks = len(src) // CHUNK_SIZE + 1
        remaining = len(src)
        
        for i in range(num_chunks):
            
            chunk_size = min(CHUNK_SIZE, remaining)
            start = i * CHUNK_SIZE
            end = i * CHUNK_SIZE + chunk_size
            feat_sum.index_add_(0, src[start:end], features[dst[start:end]])
            remaining -= chunk_size
            torch.cuda.empty_cache()
            
        assert remaining == 0
        
        degrees = torch.tensor(adj.sum(axis=-1), device=device)
        
        return feat_sum.to(self.device), degrees.to(self.device)
    
    def get_updated_sequences(self, graph_perturbed: dgl.DGLGraph,
                              targets: torch.Tensor,
                              cand_ids: torch.Tensor,
                              atk_ids: torch.Tensor,
                              edge_weight: bool=False):
        
        device = graph_perturbed.device
        num_targets = len(targets)
        num_cands = 0 if cand_ids is None else len(cand_ids)
        num_atks = len(atk_ids)
        N_ptrbd = num_targets + num_cands + num_atks
        
        indices = torch.cat([targets, atk_ids] if cand_ids is None else [targets, cand_ids, atk_ids], dim=-1)
        subgraph = dgl.remove_self_loop(dgl.node_subgraph(graph_perturbed, indices))

        adj_ptrbd = torch.zeros((N_ptrbd, N_ptrbd), dtype=torch.float32, device=device)
        
        src, dst = subgraph.edges()
        adj_ptrbd[src, dst] = subgraph.edata['weight'] if edge_weight else 1
        adj2_ptrbd = (adj_ptrbd @ adj_ptrbd).to(dtype=bool).to(dtype=torch.float32)
        adj2_ptrbd -= torch.eye(N_ptrbd, dtype=torch.float32, device=device)
        
        adj_ptrbd = adj_ptrbd[:num_targets]
        adj2_ptrbd = adj2_ptrbd[:num_targets]
        
        ###################################################################
        
        # We only need to add "the part of adj2_ptrbd that are different from adj2" to "(feat_sum, d)".
                
        targets = targets.cpu()
        if cand_ids is not None:
            cand_ids = cand_ids.cpu()
        indices = targets if cand_ids is None else torch.cat([targets, cand_ids], dim=-1)
        
        adj_org = torch.zeros_like(adj_ptrbd)
        adj_org[self.adj[targets][:, indices].nonzero()] = 1
        
        adj2_org = torch.zeros_like(adj_ptrbd)
        adj2_org[self.adj2[targets][:, indices].nonzero()] = 1

        # 1. The part of adj_ptrbd that is different from adj
        adj_diff = torch.zeros_like(adj_ptrbd)
        mask = (adj_ptrbd - adj_org) > 0
        adj_diff[mask] = adj_ptrbd[mask]
        
        adj2_diff = torch.zeros_like(adj2_ptrbd)
        mask = (adj2_ptrbd - adj2_org) > 0
        adj2_diff[mask] = adj2_ptrbd[mask]
        
        # 2. Calculate the delta (difference) of (feat_sum, d)
        train_neg = subgraph.ndata['train_neg']
        train_pos = subgraph.ndata['train_pos']
        train_masked = subgraph.ndata['train_masked']
        features = subgraph.ndata['x']
        
        feat_sum_1, d_1 = self.group_agg_tensor(adj_diff, features, train_neg, train_pos, train_masked)
        feat_sum_2, d_2 = self.group_agg_tensor(adj2_diff, features, train_neg, train_pos, train_masked)
        
        feat_sum_diff = torch.concat([torch.zeros_like(features[:num_targets]),
                                      feat_sum_1, feat_sum_2], dim=-1).reshape((-1, 7, self.feat_dim))
        d_diff = torch.concat([torch.zeros((num_targets, 1), device=self.device), d_1, d_2], dim=-1).reshape((-1, 7, 1))
        
        # 3. Add the delta to (feat_sum, d)
        feat_sum_org = self.feat_sum[targets].to(device)
        d_org = self.d[targets].to(device)
        feat_sum = feat_sum_org + feat_sum_diff
        d = d_org + d_diff
        
        # 4. Divide feat_sum by d to get sequences
        mask = d > 0
        denom = torch.ones_like(d)
        denom[mask] = d[mask]
        sequences = feat_sum / denom
        
        return sequences

    def group_agg_tensor(self, adj: torch.Tensor, features: torch.Tensor,
                         train_neg: torch.Tensor, train_pos: torch.Tensor,
                         train_masked: torch.Tensor):
        
        adj_neg = adj * train_neg
        adj_pos = adj * train_pos
        adj_masked = adj * train_masked
        
        feat_sum_neg, d_neg = self.feat_agg_tensor(adj_neg, features)
        feate_sum_pos, d_pos = self.feat_agg_tensor(adj_pos, features)
        feat_sum_masked, d_masked = self.feat_agg_tensor(adj_masked, features)

        feat_sum_hop = torch.concat([feat_sum_neg, feate_sum_pos, feat_sum_masked], dim=-1)
        d_hop = torch.concat([d_neg, d_pos, d_masked], dim=-1)
        
        return feat_sum_hop, d_hop
    
    def feat_agg_tensor(self, adj: torch.Tensor, features: torch.Tensor):
        
        device = adj.device
        shape = (adj.shape[0], features.shape[-1])
        
        feat_sum = torch.zeros(shape, device=device)
        degrees = adj.sum(axis=-1).unsqueeze(-1)

        if degrees.sum() > 0:
            src, dst = adj.nonzero(as_tuple=True)
            feat_sum.index_add_(0, src, features[dst])
            
        return feat_sum, degrees
    
    ################################################
    