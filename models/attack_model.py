import dgl
import torch
import torch.nn as nn
import numpy as np

from typing import Optional

from modules.schedulers import ExpDecayTemperatureScheduler, ExpDecayEpsilonScheduler
from utils.utils import get_complment_indices, gumbel_softmax, st_top_k, st_gumbel_top_k

class MonTi(nn.Module):
    def __init__(self, feat_dim: int,
                 min_feat: torch.Tensor,
                 max_feat: torch.Tensor,
                 discrete: bool,
                 feat_budget: int,
                 emb_dim: int=64,
                 ff_dim: int=2048,
                 max_candidates: int=128,
                 num_layers: int=6,
                 num_heads: int=4,
                 initial_temp: float=1e2,
                 min_temp: float=1e-2,
                 temp_decay_rate: float=0.63,
                 initial_eps: float=1e2,
                 min_eps: float=1e-2,
                 eps_decay_rate: float=0.63,
                 dropout: float=0.0,
                 shared_parameters: bool=False,
                 without_candidates: bool=False,
                 pos_encoding: bool=True,
                 ada_budget: bool=True) -> None:
        
        super().__init__()

        self.discrete = discrete
        self.feat_budget = feat_budget
        self.feat_dim = feat_dim
        self.emb_dim = emb_dim
        self.max_candidates = max_candidates

        self.min_feat = min_feat
        self.max_feat = max_feat
        
        self.shared_parameters = shared_parameters
        self.without_candidates = without_candidates
        self.pos_encoding = pos_encoding
        self.ada_budget = ada_budget

        # Schedulers for Gumble-top-k
        self.temp_scheduler = ExpDecayTemperatureScheduler(initial_temp, min_temp, temp_decay_rate)
        self.eps_scheduler = ExpDecayEpsilonScheduler(initial_eps, min_eps, eps_decay_rate)

        # CandidateSelector
        if not without_candidates:
            self.candidate_selector = CandidateSelector(feat_dim, emb_dim, max_candidates, numerics=['degree', 'beta'], dropout=dropout)
        
        # NodeEncoder
        self.target_encoder = NodeEncoder(feat_dim, emb_dim, numerics=['degree'], dropout=dropout)
        
        if not without_candidates:
            self.candidate_encoder = self.target_encoder if shared_parameters else NodeEncoder(feat_dim, emb_dim, numerics=['degree'], dropout=dropout)

        # Positional Encoding (learnable embeddings)
        if pos_encoding:
            self.pos_encoding = nn.Embedding(3, emb_dim) # 0: target, 1: candidate 2: attack node
            self.one_tensor = torch.ones(1, dtype=int)

        # TransformerEncoder
        layer = nn.TransformerEncoderLayer(emb_dim, num_heads, ff_dim)
        norm = nn.LayerNorm(emb_dim)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers, norm=norm)

        # Linear projection after TransformerEncoder
        self.linear_feat = nn.Linear(emb_dim, feat_dim)
        self.linear_edge = nn.Linear(emb_dim, emb_dim)

        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)
    
    # subgraph: subgraph induced by exploration nodes (k-hop neighbors of targets)
    def forward(self, subgraph: dgl.DGLGraph,
                idx_targets,
                node_budget=5,
                edge_budget=50,
                cand_selection=True,
                masking=False):
        
        device = subgraph.device
        
        # 1. CandidateSelctor
        ## The number of directly connected target nodes -> beta
        if self.without_candidates:
            cand_indices = None
        else:
            N = subgraph.num_nodes()
            subgraph.ndata['beta'] = torch.zeros(N).to(device)
            subgraph.send_and_recv(subgraph.out_edges(idx_targets),
                            lambda edges: {'m': torch.ones_like(edges.dst['beta'])},
                            lambda nodes: {'beta': torch.sum(nodes.mailbox['m'], dim=1)})
            output = self.select_candidates(subgraph, idx_targets, cand_selection)
            if output is None:
                return None, None, None
            candidates, cand_indices = output

        # 2. Target Encoding & Candidate Encoding & Positional Encoding
        ## 2-1. Target Encoding
        x_t = subgraph.ndata['x'][idx_targets]
        h_t = subgraph.ndata['h'][idx_targets]
        z_t = self.target_encoder(x_t, h_t, subgraph, idx_targets)

        ## 2-2. Candidate Encoding
        if not self.without_candidates:
            x_v = subgraph.ndata['x'][cand_indices]
            h_v = subgraph.ndata['h'][cand_indices]
            if self.training:
                h_v = torch.einsum('nc, n->nc', h_v, candidates) # To apply straight through (ST) technique
            z_v = self.candidate_encoder(x_v, h_v, subgraph, cand_indices)

        ## 2-3. Generate random noises for attack nodes
        z_u = (torch.randn((node_budget, self.emb_dim), device=device) if node_budget > 1
               else torch.zeros((node_budget, self.emb_dim), device=device))

        ## 2-4. Positional Encoding
        if self.pos_encoding:
            one_tensor = self.one_tensor.to(device)
            z_t = z_t + self.pos_encoding(one_tensor * 0)
            if not self.without_candidates:
                z_v = z_v + self.pos_encoding(one_tensor)
            z_u = z_u + self.pos_encoding(one_tensor * 2)

        # 3. TransformerEncoder
        num_targets = len(idx_targets)
        num_candidates = 0 if self.without_candidates else len(cand_indices)
        start_atk = num_targets + num_candidates
        seq_len = num_targets + num_candidates + node_budget

        mask = None
        if masking:
            mask = (torch.ones((seq_len, seq_len)) * float('-inf')).to(device)
            mask[:start_atk, :start_atk] = 0
            mask[start_atk:] = 0
        z = torch.cat([z_t, z_u] if self.without_candidates else [z_t, z_v, z_u], dim=0)
        z = self.transformer_encoder(z, mask)
        z = self.activation(z)
        z = self.dropout(z)

        # 4. Projection to raw feature space
        x_atk = self.feature_projection(z[start_atk:])
    
        # 5. Edge Generation
        adj_attack = self.edge_generation(z, subgraph, num_targets, num_candidates, node_budget, edge_budget, random_edge=False)

        # Generated node attributes, edges_attack, candidates indices
        return x_atk, adj_attack, cand_indices
    
    def select_candidates(self, subgraph, idx_targets, cand_selection=True):
        N = subgraph.num_nodes()
        exp_nodes = get_complment_indices(N, idx_targets) # Indices of exploration nodes
        if len(exp_nodes) > self.max_candidates:
            if cand_selection:
                candidates, cand_indices = self.candidate_selector(subgraph, idx_targets, exp_nodes,
                                                        self.temp_scheduler.get_temperature(),
                                                        self.eps_scheduler.get_epsilon())
            else:
                cand_indices = np.random.choice(exp_nodes.numpy(), self.max_candidates, replace=False)
                cand_indices = torch.tensor(cand_indices, device=subgraph.device)
                candidates = torch.ones_like(cand_indices)
        elif len(exp_nodes) > 0:
            cand_indices = exp_nodes.to(subgraph.device)
            candidates = torch.ones_like(cand_indices)
        else:
            print("Warning: There is no exploration nodes in the subgraph")
            return None
        
        return candidates, cand_indices

    def feature_projection(self, z):
        
        z = self.linear_feat(z)
        
        if self.discrete:
            temperature =  self.temp_scheduler.get_temperature()
            epsilon = self.eps_scheduler.get_epsilon() if self.training else 1
            x, _ = st_gumbel_top_k(z, self.feat_budget, temperature, epsilon, dim=-1, randomness=self.training)
        else:
            z = torch.sigmoid(z)
            x = z * (self.max_feat - self.min_feat) + self.min_feat

        return x

    def edge_generation(self, z, subgraph, num_targets, num_candidates, node_budget, edge_budget, random_edge: bool=False):
        
        # Masking for preventing attack X attack edges & self-loops
        adj_attack_shape = (node_budget, z.size(0))
        adj_attack = torch.zeros(adj_attack_shape, device=subgraph.device)
        mask_tril = torch.tril(torch.ones(adj_attack_shape, dtype=torch.bool),
                               diagonal=num_targets + num_candidates - 1).to(subgraph.device) # -1 for removing self-loops
        
        if random_edge:
            rand = torch.rand_like(adj_attack)
            _, indices = rand[mask_tril].topk(edge_budget)
            src, dst = mask_tril.nonzero(as_tuple=True)
            adj_attack[src[indices], dst[indices]] = 1
            return adj_attack
        
        # 1. Linear & Normlaization		
        e = self.linear_edge(z)
        e = torch.nn.functional.normalize(e, dim=-1)
        
        # 2. Edge Score Matrix Generation        
        src, dst = mask_tril.nonzero(as_tuple=True)
        edge_scores = torch.sum(e[src] * e[dst], dim=-1)

        # 3. Gumbel-softmax on Edge Score Matrix
        temperature = self.temp_scheduler.get_temperature()
        epsilon = self.eps_scheduler.get_epsilon()
        edge_scores = gumbel_softmax(edge_scores, temperature, epsilon, dim=0, randomness=self.training)
        
        # 4. Generate Edges by ST-top-k
        edge_scores_mat = torch.zeros(adj_attack_shape, device=subgraph.device)

        ## 4-1. Firstly, each attack node gets one edge attached to the target node
        edge_scores_mat[src, dst] = edge_scores
        if self.ada_budget:
            adj_attack[:, :num_targets], _ = st_top_k(edge_scores_mat[:, :num_targets], 1, dim=-1)

            ## 4-2. Next, remaining budgets are consumed by ST-top-k on whole Edge Score Matrix
            edge_budget -= node_budget
            if edge_budget > 0:
                mask_empty = (adj_attack == 0)
                mask = mask_empty & mask_tril # Masking for preventing duplicated edges and self-loops
                edge_budget = min(edge_budget, mask.sum()) # Check the remaining edge space
                
                adj_attack_remaining, _ = st_top_k(edge_scores_mat[mask], edge_budget, dim=0)
                adj_attack[mask] = adj_attack_remaining
        else:
            assert edge_budget % node_budget == 0
            degree_budget = edge_budget // node_budget
            adj_attack, _ = st_top_k(edge_scores_mat, degree_budget, dim=-1)

        return adj_attack
    
    # It is called per end of one epoch
    def step(self):
        self.temp_scheduler.step()
        self.eps_scheduler.step()

class CandidateSelector(nn.Module):
    def __init__(self, feat_dim: int,
                 emb_dim: int=64,
                 max_candidates: int=128,
                 numerics=['degree', 'beta'],
                 dropout: float=0.5) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.emb_dim = emb_dim
        self.max_candidates = max_candidates
        self.numerics = numerics

        # Linear projection for raw attributes
        self.linear_raw = nn.Linear(feat_dim, emb_dim)

        # Linear projection for numeric informations (degree, # of connected target nodes)
        if len(self.numerics) > 0:
            self.linear_numeric = nn.Linear(len(numerics), emb_dim)

        # MLP for scoring
        self.mlp_score = nn.Sequential(
            nn.Linear(emb_dim*4 if len(self.numerics) > 0 else emb_dim*3, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1)
        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
    
    # target nodes, attack nodes
    def forward(self, subgraph: dgl.DGLGraph, idx_targets, exp_nodes, temperature, epsilon):

        # Linear projection of raw attributes -> x_v
        x_v = self.linear_raw(subgraph.ndata['x'][exp_nodes])
        
        # Representation of a exploration node -> h_v
        h_v = subgraph.ndata['h'][exp_nodes]

        # Representation of the target 'set' -> h_T	
        h_T = subgraph.ndata['h'][idx_targets].mean(dim=0).unsqueeze(0)
        h_T = h_T.expand((len(exp_nodes), self.emb_dim))

        # Linear projection of numeric informations
        if len(self.numerics) > 0:
            numerics = []
            for key in self.numerics:
                numerics.append(subgraph.ndata[key][exp_nodes])
            numerics = torch.stack(numerics, dim=-1)
            emb_numerics = self.linear_numeric(numerics)
            # Concatenation of all informations
            emb = torch.cat([x_v, h_v, h_T, emb_numerics], dim=-1)
        else:
            emb = torch.cat([x_v, h_v, h_T], dim=-1)

        emb = self.activation(emb)
        emb = self.dropout(emb)
        
        # Scoring by MLP		
        scores = self.mlp_score(emb).squeeze(-1)

        # Select candidates based on scores using ST Gumbel-top-k
        candidates, cand_indices = st_gumbel_top_k(scores, self.max_candidates, temperature,
                                                   epsilon, randomness=self.training)
        candidates = candidates[cand_indices]
        cand_indices = exp_nodes.to(subgraph.device)[cand_indices]

        return candidates, cand_indices

class NodeEncoder(nn.Module):
    
    def __init__(self, feat_dim: int,
                 emb_dim: int,
                 numerics=['degree'],
                 dropout: float=0.5) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.emb_dim = emb_dim
        self.numerics = numerics

        # Linear projection for raw attributes
        self.linear_raw = nn.Linear(feat_dim, emb_dim)

        # Linear projection for numeric informations (degree, # of connected target nodes)
        if len(self.numerics) > 0:
            self.linear_numeric = nn.Linear(len(numerics), emb_dim)

        # Linear projection for output
        
        self.linear_out = nn.Linear(emb_dim*3 if len(self.numerics) > 0 else emb_dim*2, emb_dim)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, h, subgraph, n_ids):

        # Linear projection of raw attributes
        x = self.linear_raw(x)

        # Linear projection of numeric informations
        if len(self.numerics) > 0:
            numerics = []
            for key in self.numerics:
                numerics.append(subgraph.ndata[key][n_ids])
            numerics = torch.stack(numerics, dim=-1)
            emb_numerics = self.linear_numeric(numerics)
            enc_input = torch.cat([x, h, emb_numerics], dim=-1)
        else:
            enc_input = torch.cat([x, h], dim=-1)

        enc_input = self.activation(enc_input)
        enc_input = self.dropout(enc_input)

        enc_output = self.linear_out(enc_input)

        return enc_output