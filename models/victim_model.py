import torch
import torch.nn as nn
import dgl
import numpy as np

from dgl.nn.pytorch.conv import GATConv, SAGEConv, GraphConv
    
class MLP(nn.Module):
    def __init__(self, feature_dim: int, emb_size: list, num_classes: int=2):
        super(MLP, self).__init__()
        self.num_layers = len(emb_size)
        self.emb_dim = emb_size
        
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = feature_dim if i == 0 else self.emb_dim[i-1]
            self.layers.append(nn.Linear(in_dim, self.emb_dim[i]))
        
        self.linear_layer = nn.Linear(self.emb_dim[-1], num_classes)
        self.activation = nn.LeakyReLU()
    
    def forward(self, blocks, prediction: bool=True):
        x = blocks[-1].dstdata['x']
        
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = self.activation(x)
        
        if prediction:
            x = self.linear_layer(x)
        return x
    
    def to_prob(self, blocks):
        pos_scores = torch.softmax(self.forward(blocks), dim=1)
        return pos_scores

class GCN(nn.Module):
    def __init__(self, feature_dim: int, emb_dim: list, num_classes: int=2):
        super(GCN, self).__init__()
        self.num_layers = len(emb_dim)
        self.emb_dim = emb_dim
        
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = feature_dim if i == 0 else self.emb_dim[i-1]
            self.layers.append(GraphConv(in_dim, self.emb_dim[i]))
        
        self.linear_layer = nn.Linear(self.emb_dim[-1], num_classes)
        self.activation = nn.LeakyReLU()
    
    def forward(self, blocks, edge_weight: bool=False, prediction: bool=True):
        x = blocks[0].srcdata['x']
        
        for i in range(self.num_layers):
            ew = blocks[i].edata['weight'] if edge_weight else None
            x = self.layers[i](blocks[i], x, edge_weight=ew)
            x = self.activation(x)
        
        if prediction:
            x = self.linear_layer(x)
        return x
    
    def to_prob(self, blocks, edge_weight: bool=False):
        pos_scores = torch.softmax(self.forward(blocks, edge_weight=edge_weight), dim=1)
        return pos_scores

class GraphSAGE(nn.Module):
    def __init__(self, feature_dim: int, emb_dim: list, num_classes: int=2):
        super(GraphSAGE, self).__init__()
        self.num_layers = len(emb_dim)
        self.emb_dim = emb_dim
        
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = feature_dim if i == 0 else self.emb_dim[i-1]
            self.layers.append(SAGEConv(in_dim, self.emb_dim[i], aggregator_type='mean'))
        
        self.linear_layer = nn.Linear(self.emb_dim[-1], num_classes)
        self.activation = nn.LeakyReLU()
    
    def forward(self, blocks, edge_weight: bool=False, prediction: bool=True):
        x = blocks[0].srcdata['x']
        
        for i in range(self.num_layers):
            ew = blocks[i].edata['weight'] if edge_weight else None
            x = self.layers[i](blocks[i], x, edge_weight=ew)
            x = self.activation(x)
        
        if prediction:
            x = self.linear_layer(x)
        return x
    
    def to_prob(self, blocks, edge_weight: bool=False):
        pos_scores = torch.softmax(self.forward(blocks, edge_weight=edge_weight), dim=1)
        return pos_scores
    
class GAT(nn.Module):
    def __init__(self, feature_dim: int, emb_dim: list, head_list: list,
             num_classes: int=2, is_concat: bool=True):
        super(GAT, self).__init__()
        self.num_layers = len(emb_dim)
        self.emb_dim = emb_dim
        self.head_list = head_list
        self.is_concat = is_concat
        
        # Stack GAT layers
        layer = GATConv
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = feature_dim if i == 0 else self.emb_dim[i-1]
            out_dim = self.emb_dim[i] // self.head_list[i] if self.is_concat else self.emb_dim[i]
            self.layers.append(layer(in_dim, out_dim, self.head_list[i]))
        
        self.linear_layer = nn.Linear(self.emb_dim[-1], num_classes)
        self.activation = nn.LeakyReLU()
    
    def forward(self, blocks, edge_weight: bool=False, prediction: bool=True):
        x = blocks[0].srcdata['x']
        
        for i in range(self.num_layers):
            ew = blocks[i].edata['weight'] if edge_weight else None
            x = self.layers[i](blocks[i], x, edge_weight=ew)
            if self.is_concat:
                x = x.reshape(-1, self.emb_dim[i])
            else:
                x = x.mean(dim=1).squeeze()

        if prediction:
            x = self.linear_layer(x)
        return x
    
    def to_prob(self, blocks, edge_weight: bool=False):
        pos_scores = torch.softmax(self.forward(blocks, edge_weight=edge_weight), dim=1)
        return pos_scores

class CAREGNN(nn.Module):
    def __init__(self, feature_dim: int, emb_dim: list, num_classes: int=2, n_etypes: int=1, step_size: float=0.02):
        super(CAREGNN, self).__init__()
        self.num_layers = len(emb_dim)
        self.emb_dim = emb_dim
        self.n_etypes = n_etypes
        self.ntypes = None
        self.etypes = None
        
        self.step_size = step_size # !
        self.threshold = [0.5 for i in range(self.n_etypes)]
        self.reward_logs = [list() for i in range(self.n_etypes)]
        self.distance_list = [list() for i in range(self.n_etypes)]
        self.terminal_state = [False for i in range(self.n_etypes)]
        self.prev_distance = None
        self.curr_distance = None
        
        if len(emb_dim) != 1:
            raise AssertionError("CARE-GNN is implemented with a single layer.")
        
        self.layers = nn.ModuleList()
        for _ in range(self.n_etypes):
            self.layers.append(SAGEConv(feature_dim, self.emb_dim[0], aggregator_type='mean'))
        
        self.distance_func = nn.Linear(feature_dim, 2)
        
        self.linear_layer = nn.Linear(self.emb_dim[0] * self.n_etypes, num_classes)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, g, blocks, edge_weight: bool=False, prediction: bool=True):
        # Define noe type, edge types and batch size.
        self.ntypes = g.ntypes[0]
        self.etypes = g.etypes
        batch_size = torch.LongTensor([blocks[-1].dstdata["x"].shape[0]])
        
        # Representation and score logits list for each relation.
        rep_list, score_list = [], []
        
        for i in range(self.n_etypes):
            # Initial feature for similarity-aware neighbor selector.
            x = blocks[0].srcdata['x']
            # Similarity-aware neighbor selector.
            new_block, score_logit, selected_neighbor_distance = self.neighbor_selector(g, blocks[0], i, x, self.distance_func, edge_weight)
            
            # Score logits from the distance score function.
            score_logit = score_logit[:batch_size]
        
            # Initial feature.
            x = new_block.srcdata['x']
            ew = new_block.edata['weight'] if edge_weight else None
            x = self.layers[i](new_block, x, edge_weight=ew) # GNN representation.
            x = self.relu(x)
            
            # Append representation and score logits. 
            score_list.append(score_logit)
            rep_list.append(x)
            
            if self.training:
                selected_neighbor_distance = selected_neighbor_distance.detach().cpu().numpy()
                self.distance_list[i].append(selected_neighbor_distance)
        
        score_logits = torch.cat(score_list)
        x = torch.cat(rep_list, dim=1)
        if not prediction:
            return x
        x = self.linear_layer(x)
        return x, score_logits
    
    def neighbor_selector(self, g: dgl.DGLGraph, block, i, x, distance_func, edge_weight: bool=False):
        # Specify the graph and block with input relation index.
        g = g[self.ntypes, self.etypes[i], self.ntypes]
        block = block[self.etypes[i]]
        
        # Score logits from the distance score function.
        score_logits = []
        # Src and dst index for generating new block with choose step.
        new_src, new_dst = [], []
        # Mapping indices of node representation for aggregation.
        idx_agg, idx_org = [], []
        # Original node index of batch nodes.
        src_nodes, dst_nodes = block.srcdata[dgl.NID], block.dstdata[dgl.NID]
        
        # Neighbor istance.
        neighbor_distance = [] 
        
        # For all dst nodes.
        for node in dst_nodes:
            # Under Sample the neighbor nodes.
            degree = g.in_degrees(node)
            sample_num_under = max(int(degree * self.threshold[i]), 1)
            
            # Find the edges of each dst node.
            edge_idx = torch.where(dst_nodes[block.edges()[1]] == node)[0]
            neighbor_nodes = block.edges()[0][edge_idx]
            target_node = block.edges()[1][edge_idx]
            
            # Calculate distance representations for neighbor nodes and target node.
            neighbor_score = distance_func(x[neighbor_nodes])[:, 0]
            target_score = distance_func(x[target_node[0]]).unsqueeze(0)
            score_logits.append(target_score)
            
            # Under Sampling indices and similarity scores.
            selected_neighbor_indices, selected_neighbor_distance = self.filter_neighs(target_score, neighbor_score, sample_num_under, degree)
            under_sampled_src = src_nodes[neighbor_nodes[selected_neighbor_indices]]
            under_sampled_dst = dst_nodes[target_node[selected_neighbor_indices]]

            sampled_src = under_sampled_src 
            sampled_dst = under_sampled_dst

            new_src.append(sampled_src)
            new_dst.append(sampled_dst)
            
            idx_agg.append(torch.cat([neighbor_nodes[selected_neighbor_indices], target_node[selected_neighbor_indices]]))
            idx_org.append(torch.cat([under_sampled_src, under_sampled_dst]))
            
            neighbor_distance.append(selected_neighbor_distance)
            
        # Concatenate the edge indices.
        new_src = torch.cat(new_src)
        new_dst = torch.cat(new_dst)
        
        # Generate graph and block with graph and edge index.
        frontier = dgl.graph((new_src, new_dst), num_nodes=g.number_of_nodes())
        if edge_weight:
            e_ids = g.edge_ids(new_src, new_dst)
            frontier.edata['weight'] = g.edata['weight'][e_ids]
        frontier.ndata['x'] = torch.zeros((g.number_of_nodes(), x.size()[1])).cuda()
        
        # representation mapping
        idx_agg = torch.cat(idx_agg)
        idx_org = torch.cat(idx_org)
        frontier.ndata['x'][idx_org] = x[idx_agg]
            
        new_block = dgl.to_block(frontier, dst_nodes=dst_nodes)
        score_logits = torch.cat(score_logits)
        neighbor_distance = torch.cat(neighbor_distance)
        
        return new_block, score_logits, neighbor_distance
    
    def filter_neighs(self, target_score, neighbor_score, sample_num_under, degree): # equation (2)
        # L1-distance of the batch node and their neighbors.
        target_score = target_score[:, 0]
        distance = torch.abs(self.tanh(target_score) - self.tanh(neighbor_score)).squeeze()
        sorted_score_distance, sorted_neighbor_indices = torch.sort(distance, dim=0, descending=False)

        # Under sample according to the similarity ranking.
        if degree > 1:
            selected_neighbor_indices = sorted_neighbor_indices[:sample_num_under]
            selected_neighbor_distanceance = sorted_score_distance[:sample_num_under]
        else:
            selected_neighbor_indices = sorted_neighbor_indices.unsqueeze(0)
            selected_neighbor_distanceance = sorted_score_distance.unsqueeze(0)

        return selected_neighbor_indices, selected_neighbor_distanceance
    
    def update_average_distance(self, n_train_nodes): # eqation (5)
        self.prev_distance = self.curr_distance
        self.curr_distance= [np.sum(np.concatenate(distance))/n_train_nodes for distance in self.distance_list]
        self.distance_list = [list() for _ in range(self.n_etypes)]
    
    def update_threshold(self, epoch):
        if (epoch+1) >= 10:
            for i in range(self.n_etypes):
                if np.sum(self.reward_logs[i][-10:]) <= 2:
                    self.terminal_state[i] = True

        if (epoch+1) > 2:
            for i in range(self.n_etypes):
                if self.terminal_state[i]:
                    continue
                else:
                    reward = 1 if self.prev_distance[i] - self.curr_distance[i] >= 0 else -1
                    new_threshold = self.threshold[i] + self.step_size * reward
                    self.threshold[i] = 0.999 if new_threshold > 1 else self.threshold[i]
                    self.threshold[i] = 0.001 if new_threshold < 0 else self.threshold[i]
        
    def to_prob(self, g, blocks):
        logits, _ = self.forward(g, blocks)
        pos_scores = torch.softmax(logits, dim=1)
        return pos_scores

class PCGNN(nn.Module):
    def __init__(self, feature_dim: int, emb_dim: list, num_classes: int=2, n_etypes: int=1, rho: float=0.5):
        super(PCGNN, self).__init__()
        self.n_etypes = n_etypes
        self.num_layers = len(emb_dim)
        self.emb_dim = emb_dim
        self.ntypes = None
        self.etypes = None
        self.train_pos = None
        self.train_pos_feat = None
        
        self.rho = rho
        
        if len(emb_dim) != 1:
            raise AssertionError("PC-GNN is implemented with a single layer.")
        
        self.layers = nn.ModuleList()
        for _ in range(self.n_etypes):
            self.layers.append(SAGEConv(feature_dim, self.emb_dim[0], aggregator_type='mean'))
        
        self.dist_funcs = nn.ModuleList()
        for _ in range(self.n_etypes):
            self.dist_funcs.append(nn.Linear(feature_dim, 2))
        
        self.linear_layer = nn.Linear(self.emb_dim[-1] * self.n_etypes, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
            
    def forward(self, g, blocks, edge_weight: bool=False, prediction: bool=True):
        # Define noe type, edge types and batch size.
        self.ntypes = g.ntypes[0]
        self.etypes = g.etypes
        batch_size = torch.LongTensor([blocks[-1].dstdata["x"].shape[0]])
        
        # Representation and score logits list for each relation.
        rep_list, score_list = [], []
        
        # For each relation
        for i in range(self.n_etypes):
            # initial feature for choose step.
            x = blocks[0].srcdata['x']
            # Choose step.
            new_block, score_logit = self.choose_step(g, blocks[0], i, x, edge_weight)
            
            # Score logits from the distance score function.
            score_logit = score_logit[:batch_size]
            
            # Initial feature.
            x = new_block.srcdata['x']
            ew = new_block.edata['weight'] if edge_weight else None
            x = self.layers[i](new_block, x, edge_weight=ew) # GNN representation.
            x = self.relu(x)
            
            # Append representation and score logits.          
            score_list.append(score_logit)
            rep_list.append(x)
        
        # Concatenate representation and score logit for all relations.
        score_logits = torch.cat(score_list)
        x = torch.cat(rep_list, dim=1)
        if not prediction:
            return x
        x = self.linear_layer(x)
        return x, score_logits
    
    def choose_step(self, g, block, i, x, edge_weight: bool=False):
        # Specify the graph and block with input relation index.
        g = g[self.ntypes, self.etypes[i], self.ntypes]
        block = block[self.etypes[i]]
        
        # Score logits from the distance score function.
        score_logits = []
        # Src and dst index for generating new block with choose step.
        new_src, new_dst = [], []
        # Mapping indices of node representation for aggregation.
        idx_agg, idx_org = [], []
        
        # Unique node index corresponding to the original graph.
        src_nodes, dst_nodes = block.srcdata[dgl.NID], block.dstdata[dgl.NID]
        
        # For all dst nodes.
        for node in dst_nodes:
            # Under Sample the neighbor nodes.
            degree = g.in_degrees(node)
            filter_under = max(int(degree * 0.5), 1)
            
            # Find the edges of each dst node.
            edge_idx = torch.where(dst_nodes[block.edges()[1]] == node)[0]
            neighbor_nodes = block.edges()[0][edge_idx]
            target_node = block.edges()[1][edge_idx]
            
            # Calculate the src and dst scores.
            neighbor_score = self.dist_funcs[i](x[neighbor_nodes])[:, 0]
            target_score = self.dist_funcs[i](x[target_node[0]]).unsqueeze(0)
            score_logits.append(target_score)
            
            selected_neighbor_indices = self.under_sampling(target_score, neighbor_score, filter_under, degree)
            under_sampled_src = src_nodes[neighbor_nodes[selected_neighbor_indices]]
            under_sampled_dst = dst_nodes[target_node[selected_neighbor_indices]]
            
            # For the training phase.
            if self.training and (node in self.train_pos):
                # Calculate distnace representation for fraud nodes.
                fraud_score = self.dist_funcs[i](self.train_pos_feat)[:, 0]
                filter_over = int(filter_under * self.rho)
                
                # Over sampling.
                selected_fraud_indices = self.over_sampling(target_score, fraud_score, filter_over)
                over_sampled_src = self.train_pos[selected_fraud_indices]
                over_sampled_dst = torch.LongTensor([node]).repeat(len(self.train_pos[selected_fraud_indices])).cuda()
                
                # Concatenate the under sampled and oversampled edge index.
                sampled_src = torch.cat([under_sampled_src, over_sampled_src])
                sampled_dst = torch.cat([under_sampled_dst, over_sampled_dst])                        
            else:
                sampled_src = under_sampled_src 
                sampled_dst = under_sampled_dst
            
            # Append the edge index for each dst node.
            new_src.append(sampled_src)
            new_dst.append(sampled_dst)
            idx_agg.append(torch.cat([neighbor_nodes[selected_neighbor_indices], target_node[selected_neighbor_indices]]))
            idx_org.append(torch.cat([under_sampled_src, under_sampled_dst]))
        
        # Concatenate the edge indices.
        new_src = torch.cat(new_src)
        new_dst = torch.cat(new_dst)
        
        # Generate graph and block with graph and edge index.
        frontier = dgl.graph((new_src, new_dst), num_nodes=g.number_of_nodes())
        if edge_weight:
            e_ids = g.edge_ids(new_src, new_dst)
            frontier.edata['weight'] = g.edata['weight'][e_ids]
        frontier.ndata['x'] = torch.zeros((g.number_of_nodes(), x.size()[1])).cuda()
        
        # representation mapping
        idx_agg = torch.cat(idx_agg)
        idx_org = torch.cat(idx_org)
        frontier.ndata['x'][idx_org] = x[idx_agg]
        if self.training:
            frontier.ndata['x'][self.train_pos] = self.train_pos_feat
            
        new_block = dgl.to_block(frontier, dst_nodes=dst_nodes)
        score_logits = torch.cat(score_logits)
        return new_block, score_logits
    
    def under_sampling(self, target_score, neighbor_score, sample_num_under, degree):
        # L1-distance of the batch node and their neighbors.
        target_score = target_score[:, 0]
        score_diff = torch.abs(self.sigmoid(target_score) - self.sigmoid(neighbor_score)).squeeze()
        sorted_score_diff, sorted_neighbor_indices = torch.sort(score_diff, dim=0, descending=False)

        # Under sampling according to the distance ranking.
        selected_neighbor_indices = sorted_neighbor_indices[:sample_num_under] if degree > 1 else sorted_neighbor_indices.unsqueeze(0)

        return selected_neighbor_indices
    
    def over_sampling(self, target_score, fraud_score, sample_num_over):
        # L1-distance of the batch node and frauds.
        target_score = target_score[:, 0]
        score_diff_fraud = torch.abs(self.sigmoid(target_score) - self.sigmoid(fraud_score)).squeeze()
        sorted_score_diff_fraud, sorted_fraud_indices = torch.sort(score_diff_fraud, dim=0, descending=False)
        
        # Over sampling according to the distance ranking.
        selected_fraud_indices = sorted_fraud_indices[:sample_num_over]
    
        return selected_fraud_indices
    
    def to_prob(self, g, blocks):
        logits, _ = self.forward(g, blocks)
        pos_scores = torch.softmax(logits, dim=1)
        return pos_scores

class GAGA(nn.Module):
    def __init__(self, feat_dim, emb_dim, num_classes, num_relations=1, n_hops=2,
                 n_heads=4, ff_dim=128, n_layers=3, dropout=0.1):
        r"""        
        Parameters
        ----------
        feat_dim : int
            Input feature size; i.e., number of  dimensions of the raw input feature.
        emb_dim : int
            Hidden size of all learning embeddings and hidden vectors. 
            (deotes by E)
        n_classes : int
            Number of classes. 
            (deotes by C)
        n_hops : int
            Number of hops (mulit-hop neighborhood information). (deNotes by K)
        n_relations : int
            Number of relations. 
            (deotes by R)
        n_heads : int
            Number of heads in MultiHeadAttention module.
        dim_feedforward : int
        n_layers : int
            Number of encoders layers. 
        dropout: float
            Dropout rate on feature. Default=0.1.
        """ 
        super(GAGA, self).__init__()

        self.n_hops = n_hops
        self.n_classes = num_classes
        
        # encoder that provides hop, relation and group encodings
        self.node_encoder = NodeEncoder(feat_dim=feat_dim,
                                        emb_dim=emb_dim, n_relations=num_relations,
                                        n_hops=n_hops, dropout=dropout,
                                        n_classes=num_classes)

        # define transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(emb_dim, n_heads, ff_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

        # MLP
        proj_emb_dim = emb_dim * num_relations
        self.linear_layer = nn.Linear(proj_emb_dim, num_classes)
        
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def cross_relation_agg(self, out):
        r"""Aggregate target node's outputs under all relations.
        Parameters
        ----------
        out : torch.Tensor
            The output tensor of Transformer Encoder.
            Shape = (S, N, E)
        """

        device = out.device
        n_tokens = out.shape[0]

        # extract output vector(s) of the target node under each relation
        block_len = 1 + self.n_hops * (self.n_classes + 1)
        indices = torch.arange(0, n_tokens, block_len, dtype=torch.int64).to(device)

        # (n_relations, N, E)
        mr_feats = torch.index_select(out, dim=0, index=indices)

        #  (N,E) tuple_len = n_relations
        mr_feats = torch.split(mr_feats, 1, dim=0)

        # (N,n_relations*E)
        agg_feats = torch.cat(mr_feats, dim=2).squeeze()

        return agg_feats

    def forward(self, x: torch.Tensor, prediction: bool=True):
        r"""
        Parameters
        ----------
        x : Tensor
            Input feature sequence. Shape (N, S, E)
        """
        # input feature sequence (N,S,E) -> (S,N,E)
        x = torch.transpose(x, 1, 0)

        # encoding in Section 4.3
        x_enc = self.node_encoder(x)

        # transformer encoder
        out = self.transformer_encoder(x_enc)

        # cross-relation aggregation, (S,N,E) -> (N,E)
        out = self.cross_relation_agg(out)

        # prediction
        if prediction:
            out = self.linear_layer(out)

        return out
    
    def to_prob(self, blocks):
        
        pos_scores = torch.softmax(self.forward(blocks), dim=1)
        return pos_scores

class NodeEncoder(nn.Module):
    def __init__(self, feat_dim, emb_dim, n_classes, n_hops, n_relations, dropout=0.1):
        r"""
        The shape of the output is (S, N, E), where S is input sequence length
        (S = n_relations * (n_hops * (n_classes + 1) + 1)),
        N is the batch size, E is the output embedding size.
        """
        super(NodeEncoder, self).__init__()
        
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_classes = n_classes

        # number of groups per relation (+1 for unlabeled nodes)
        self.n_groups = n_classes + 1

        # input sequence length per relation
        self.seq_len = n_hops * (n_classes + 1) + 1
        
        self.hop_embedding = nn.Embedding(n_hops + 1, emb_dim)
        self.rel_embedding = nn.Embedding(n_relations, emb_dim)
        self.group_embedding = nn.Embedding(n_classes + 1, emb_dim)

        # linear projection
        self.linear_layer = nn.Linear(feat_dim, emb_dim)
        self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        device = x.device

        hop_idx = torch.arange(self.n_hops + 1, dtype=torch.int64, device=device)
        rel_idx = torch.arange(self.n_relations, dtype=torch.int64, device=device)
        grp_idx = torch.arange(self.n_groups, dtype=torch.int64, device=device)
        
        # -------- HOP ENCODING STRATEGY --------
        hop_emb = self.hop_embedding(hop_idx)
        center_hop_emb = hop_emb[0].unsqueeze(0)
        hop_emb_list = [center_hop_emb]
        for i in range(1, self.n_hops + 1):
            hop_emb_list.append(hop_emb[i].repeat(self.n_groups, 1))
        hop_emb = torch.cat(hop_emb_list, dim=0).repeat(self.n_relations, 1)

        # -------- RELATION ENCODING STRATEGY --------
        rel_emb = self.rel_embedding(rel_idx)
        rel_emb = torch.repeat_interleave(rel_emb, self.seq_len, dim=0)

        # -------- GROUP ENCODING STRATEGY --------
        grp_emb = self.group_embedding(grp_idx)
        center_grp_emb = grp_emb[-1].unsqueeze(0) # masked
        hop_grp_emb = grp_emb.repeat(self.n_hops, 1) # (n_hop*n_groups, E)
        grp_emb = torch.cat((center_grp_emb, hop_grp_emb), dim=0).repeat(self.n_relations, 1)

        # linear projection
        out = self.activation(self.linear_layer(x))
        out = out + hop_emb.unsqueeze(1) + rel_emb.unsqueeze(1) + grp_emb.unsqueeze(1)
        out = self.dropout(out)

        # (seq_len, num_nodes (in batch), emb_dim)
        return out

# ====================================================================
