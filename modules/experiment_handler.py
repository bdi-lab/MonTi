import os, json

import dgl
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import TensorDataset, DataLoader

from tqdm.auto import tqdm
from typing import Optional, Dict, Any

from models.attack_model import MonTi
from models.victim_model import MLP, GCN, GraphSAGE, GAT, CAREGNN, PCGNN, GAGA
from modules.data_handler import DataHandler
from modules.result_manager import ResultManager
from modules.schedulers import EarlyStopScheduler
from utils.utils import set_seed, get_n_ids, misclassification_rate
from utils.constants import (DATA_ROOT_DIR, DIR_NAME_DICT, PRETRAINED_MODELS_DIR,
                             EVAL_METRICS_ATTACK, NEED_SAMPLER_MODELS, ExperimentState)

class ExperimentHandler():
    
    def __init__(self, config: Dict[str, Any],
                 data: DataHandler,
                 result_manager: Optional[ResultManager]=None,
                 check_clean_performance: bool=False) -> None:
        
        self.config: Dict[str, Any] = config
        self.data: DataHandler = data
        self.multi_target: bool = data.multi_target
        self.result_manager: Optional[ResultManager] = result_manager

        # Fix the random variables with seed.
        set_seed(config['seed'])
        
        # CUDA Device.
        self.device = torch.device(config['cuda_id']) if torch.cuda.is_available() else torch.device("cpu")

        # Load pretrained models.
        self.surrogate_type: str = self.config["surrogate_model"]
        self.surrogate_model: Optional[nn.Module] = None
        self.sampler_surrogate: Optional[dgl.dataloading.NeighborSampler] = None
        
        self.victim_type: str = self.config["victim_model"]
        self.victim_model: Optional[nn.Module] = None
        self.sampler_victim: Optional[dgl.dataloading.NeighborSampler] = None

        self.load_pretrained_models()
        if check_clean_performance:
            self.check_clean_performance()
            return
        
        # Precompute node representations using the surrogate model.
        self.get_node_representations()
        
        # Model to be trained.
        self.model: MonTi = self.initialize_attack_model()
        self.model.to(self.device)

        # Early-stopping scheduler
        self.es_scheduler = EarlyStopScheduler(patience=self.config["patience"],
                                               metrics=EVAL_METRICS_ATTACK)
        # Optimizer
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        self.epoch_st = 1
    
    # Load a pretrained surrogate model and a pretrained victim model.
    def load_pretrained_models(self) -> None:
        
        surrogate_model = self.load_pretrained_model(self.surrogate_type, flag="surrogate")
        self.surrogate_model = surrogate_model
        self.surrogate_model.to(self.device)
        
        if self.surrogate_type in NEED_SAMPLER_MODELS:
            self.sampler_surrogate = dgl.dataloading.MultiLayerFullNeighborSampler(self.surrogate_model.num_layers)
        
        victim_model = self.load_pretrained_model(self.victim_type, flag="victim")
        self.victim_model = victim_model
        self.victim_model.to(self.device)
        
        if self.victim_type in NEED_SAMPLER_MODELS:
            self.sampler_victim = dgl.dataloading.MultiLayerFullNeighborSampler(self.victim_model.num_layers)

    # Load a pretrained model according to given config
    def load_pretrained_model(self, model_name: str, flag="surrogate") -> nn.Module:
        
        assert flag in ["surrogate", "victim"]
            
        setting = f"{model_name}-{self.data.data_name}-{self.data.split}"
        config_path = os.path.join(PRETRAINED_MODELS_DIR, "configs", f"{setting}.json")
        with open(config_path) as f:
            model_config = json.load(f)

        feat_dim = self.data.features.shape[1]
        emb_sizes = model_config['emb_sizes']
        num_classes = self.data.num_classes
        num_relations = self.data.num_relations

        torch.cuda.empty_cache()
        model = None
        if model_name == "MLP":
            model = MLP(feat_dim, emb_sizes, num_classes=num_classes)
        elif model_name == "GCN":
            model = GCN(feat_dim, emb_sizes, num_classes=num_classes)
        elif model_name == "GraphSAGE":
            model = GraphSAGE(feat_dim, emb_sizes, num_classes=num_classes)
        elif model_name == "GAT":
            head_list = model_config['head_list']
            model = GAT(feat_dim, emb_sizes, head_list, num_classes=num_classes)
        elif model_name == "CAREGNN":
            step_size = model_config['step_size']
            model = CAREGNN(feat_dim, emb_sizes, num_classes=num_classes, n_etypes=num_relations, step_size=step_size)
        elif model_name == "PCGNN":
            rho = model_config['rho']
            model = PCGNN(feat_dim, emb_sizes, num_classes=num_classes, n_etypes=num_relations, rho=rho)
        elif model_name == "GAGA":
            model = GAGA(feat_dim, emb_sizes[0], num_classes=num_classes, num_relations=num_relations)
            
        assert (not (model is None))

        model_path = os.path.join(PRETRAINED_MODELS_DIR, "surrogate_models" if flag == "surrogate" else "victim_models", f"{setting}.pickle")
        model.load_state_dict(torch.load(model_path))

        # Freeze the model
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        return model

    # Check the performance of the surrogate/victim model on the original clean graph
    def check_clean_performance(self):
        print("\nEvaluating model performance on the clean graph:")

        misclf_rate = self.check_performance("surrogate", "valid", attack=False)['misclf_rate']
        print(f"Surrogate Model Validation Misclassification Rate: {(misclf_rate)*100:.2f}%")

        misclf_rate = self.check_performance("surrogate", "test", attack=False)['misclf_rate']
        print(f"Surrogate Model Test Misclassification Rate: {(misclf_rate)*100:.2f}%")

        misclf_rate = self.check_performance("victim", "valid", attack=False)['misclf_rate']
        print(f"Victim Model Validation Misclassification Rate: {(misclf_rate)*100:.2f}%")

        misclf_rate = self.check_performance("victim", "test", attack=False)['misclf_rate']
        print(f"Victim Model Test Misclassification Rate: {(misclf_rate)*100:.2f}%")
        
    # Check the performance of the surrogate/victim model on valid/test target sets.
    def check_performance(self, model: str="surrogate",
                          set_type: str="valid",
                          attack: bool=False) -> Dict[str, float]:
        
        assert model in ["surrogate", 'victim']
        assert set_type in ["valid", "test"]

        model_name = self.surrogate_type if model == "surrogate" else self.victim_type
        need_sequence = (model_name == "GAGA")

        answers = np.empty((0,))
        y_preds = np.empty((0,))
        if attack:
            loss_l = []

        target_idx = self.data.get_validation_set() if set_type == "valid" else self.data.get_test_set()
        with tqdm(target_idx) as pbar:
            pbar.set_description("Validation: " if set_type == "valid" else "Test: ")
            
            with torch.no_grad():
                for t_id in pbar:
                    # Get input for the model
                    n_ids = get_n_ids(self.data.target_assignment, t_id) if self.multi_target else [t_id]
                    n_ids = torch.tensor(n_ids, dtype=int)
                    
                    # attack -> perturbed graph / not attack -> clean graph
                    if attack:
                        node_budget = self.data.node_budgets[t_id] if self.multi_target else self.data.node_budget
                        edge_budget = self.data.edge_budgets[t_id] if self.multi_target else self.data.edge_budget
                        output = self.get_perturbed_graph(n_ids, node_budget, edge_budget,
                                                        need_sequence=need_sequence)
                        if output is None:
                            print("[Warning] perturbed graph is None")
                            continue

                        sequences = None
                        if need_sequence:
                            graph, sequences = output
                        else:
                            graph = output
                        loss = self.loss_function(graph, n_ids, training=False, sequences=sequences)
                        loss_l.append(loss.item())
                    else:
                        graph = self.data.graph
                        sequences = None
                        if model_name == "GAGA" and not attack:
                            sequences = self.data.sequences[n_ids].to(self.device)
                                
                    # Get output of the model
                    labels = graph.ndata['y'][n_ids].data.cpu().numpy()
                    logits = self.get_logits(graph, n_ids, model, training=False, sequences=sequences)
                    logits = torch.nn.functional.softmax(logits, dim=1).cpu().detach().numpy()
                            
                    answers = np.append(answers, labels)	
                    y_preds = np.append(y_preds, logits.argmax(axis=1))

                    misclf_rate = misclassification_rate(answers, y_preds)
                    if attack:
                        postfix_new = ", ".join([f"num_targets: {str(len(n_ids)):>4}",
                                                f"node_budget: {str(node_budget):>3}",
                                                f"edge_budget: {str(edge_budget):>5}",
                                                f"misclf_rate: {misclf_rate*100:.2f}%"])
                    else:
                        postfix_new = ", ".join([f"num_targets: {str(len(n_ids)):>4}",
                                                f"misclf_rate: {misclf_rate*100:.2f}%"])
                    pbar.set_postfix_str(postfix_new)
                
        
        misclf_rate = misclassification_rate(answers, y_preds)
        results = {"accuracy": 1-misclf_rate,
                   "misclf_rate": misclf_rate}
        if attack:
            results["loss"] = np.mean(loss_l)
        
        return results

    # Load (or precompute) GNN node representations from the surrogate model
    def get_node_representations(self):
        
        model_name = self.surrogate_type
        dir_name = DIR_NAME_DICT[self.data.data_name]
        h_path = os.path.join(DATA_ROOT_DIR, dir_name, f"h_{model_name}.npy")
        
        if os.path.exists(h_path):
            h = torch.tensor(np.load(h_path))
        else:
            h = self.calculate_node_representations()
            np.save(h_path, h.cpu().numpy())

        self.data.graph.ndata['h'] = h.to(self.data.graph.device)
  
    def calculate_node_representations(self) -> torch.Tensor:

        surrogate_type = self.surrogate_type
        sampler = self.sampler_surrogate
        
        graph = self.data.graph
        N = graph.num_nodes()
        h = torch.zeros((N, self.config['emb_dim']), device=self.device)
        indices = torch.arange(N, dtype=int, device=self.device)
        batch_size = self.config['batch_size_pre']

        if surrogate_type == "GAGA":
            torch_dataset = TensorDataset(self.data.sequences.to(self.device), indices, self.data.labels)
            data_loader = DataLoader(dataset=torch_dataset, batch_size=batch_size,
                                     num_workers=0, shuffle=False)
        else:
            data_loader = dgl.dataloading.DataLoader(graph, indices, sampler, batch_size=batch_size, 
                                                     num_workers=0, shuffle=False)

        print("\nCalculating node representations using the pretrained surrogate model ...")
        for batch in tqdm(data_loader):
            if surrogate_type == "GAGA":
                sequences, output_nodes, _ = batch
                h_batch = self.surrogate_model(sequences, prediction=False)
            else:
                _, output_nodes, blocks = batch
                if surrogate_type in ["CAREGNN", "PCGNN"]:
                    h_batch = self.surrogate_model(graph, blocks, prediction=False)
                else:
                    h_batch = self.surrogate_model(blocks, prediction=False)
            h[output_nodes] = h_batch
        
        return h.cpu()

    # Initialize MonTi according to given config
    def initialize_attack_model(self) -> nn.Module:
        
        features = self.data.features
        feat_dim = features.shape[1]
        min_feat, _ = features.min(dim=0)
        max_feat, _ = features.max(dim=0)
        feat_budget = max(int(features.sum(dim=1).mean() + 0.5), 1)
        
        model = MonTi(feat_dim=feat_dim,
                      min_feat=min_feat.to(self.device),
                      max_feat=max_feat.to(self.device),
                      discrete=self.data.discrete,
                      feat_budget=feat_budget,
                      emb_dim=self.config['emb_dim'],
                      ff_dim=self.config['ff_dim'],
                      max_candidates=self.config['max_candidates'],
                      num_layers=self.config['num_layers'],
                      num_heads=self.config['num_heads'],
                      initial_temp=self.config['initial_temp'],
                      min_temp=self.config['min_temp'],
                      temp_decay_rate=self.config['temp_decay_rate'],
                      initial_eps=self.config['initial_eps'],
                      min_eps=self.config['min_eps'],
                      eps_decay_rate=self.config['eps_decay_rate'],
                      dropout=self.config['dropout'],
                      shared_parameters=self.config['shared_parameters'],
                      without_candidates=self.config['without_candidates'],
                      pos_encoding=self.config['pos_encoding'],
                      ada_budget=self.config['ada_budget'])
        
        return model

    def get_logits(self,
                   graph: dgl.DGLGraph,
                   n_ids: torch.Tensor,
                   model: str="surrogate",
                   training: bool=True,
                   sequences: Optional[torch.Tensor]=None) -> torch.Tensor:
        
        assert model in ["surrogate", 'victim'] 
        model_type = self.surrogate_type if model == "surrogate" else self.victim_type
        fds_model = self.surrogate_model if model == "surrogate" else self.victim_model
        
        if model_type == "GAGA":
            # edge_weight is used during generating new sequences
            assert sequences is not None
            logits = fds_model(sequences)
        else:
            sampler = self.sampler_surrogate if model == "surrogate" else self.sampler_victim
            _, _, blocks = sampler.sample_blocks(graph, n_ids.to(graph.device))
            blocks = [block.to(self.device) for block in blocks]
            
            if self.config['surrogate_model'] in ["CAREGNN", "PCGNN"]:
                logits, _ = fds_model(graph, blocks, edge_weight=training)
            else:
                logits = fds_model(blocks, edge_weight=training)
        
        return logits
    
    # Loss function for the attack model
    def loss_function(self, graph_perturbed: dgl.DGLGraph, n_ids, training: bool=True, 
                      sequences: Optional[torch.Tensor]=None) -> torch.Tensor:
        
        # Attack loss
        logits = self.get_logits(graph_perturbed, n_ids, "surrogate", training, sequences)
        if torch.isnan(logits).sum() != 0:
            raise ValueError("Logits should not be nan.")

        # For multi-class classification models (benchmark)
        if self.data.num_classes > 2:
            labels_npy = self.data.labels[n_ids].unsqueeze(-1).cpu().numpy()
            logits_without_correct = logits.detach().cpu().numpy()
            np.put_along_axis(logits_without_correct, labels_npy, np.nan, axis=-1)
            labels_best_wrong = np.nanargmax(logits_without_correct, axis=-1)

            labels = torch.tensor(labels_npy, device=self.device)
            labels_best_wrong = torch.tensor(labels_best_wrong, device=self.device).unsqueeze(-1)
            logits_correct = torch.take_along_dim(logits, labels, dim=-1)
            logits_best_wrong = torch.take_along_dim(logits, labels_best_wrong, dim=-1)
        else:
            # Since we are considering targets are fraud (positive) nodes
            logits_correct = logits[:, 1]
            logits_best_wrong = logits[:, 0]
            
        attack_loss = nn.functional.relu(logits_correct - logits_best_wrong).mean()
        loss = attack_loss
        
        return loss
    
    # Perform an injection attack and get a perturbed graph using the attack model
    def get_perturbed_graph(self, n_ids, node_budget, edge_budget,
                            training: bool=False, need_sequence: bool=False):
        
        graph = self.data.graph
        n_ids = n_ids.to(graph.device)
        subgraph, idx_targets = dgl.khop_in_subgraph(graph, n_ids, k=self.config['k_exp'],
                                                     output_device=graph.device) # subgraph induced by exploration nodes

        # Get output of the model
        x_atk, adj_attack, cand_ids_sub = self.model(subgraph, idx_targets,
                                                     node_budget, edge_budget,
                                                     self.config['random_candidates'],
                                                     self.config['masking'])
        if (x_atk is None) or (adj_attack is None):
            return None

        # Make a perturbed graph (DGLGraph)
        data = {'x': x_atk}
        if need_sequence:
            data['train_neg'] = torch.zeros(len(x_atk), dtype=int, device=self.device)
            data['train_pos'] = torch.zeros(len(x_atk), dtype=int, device=self.device)
            data['train_masked'] = torch.ones(len(x_atk), dtype=int, device=self.device)
            
        graph_perturbed = dgl.add_nodes(graph, len(x_atk), data)
        if training:
            graph_perturbed.edata['weight'] = torch.ones(graph_perturbed.num_edges(), 
                                                         device=self.device)

        N = graph.num_nodes()
        atk_ids = torch.arange(len(x_atk), device=self.device) + N
        cand_ids = None if cand_ids_sub is None else subgraph.ndata[dgl.NID][cand_ids_sub] 
        indices = torch.cat([n_ids, atk_ids] if cand_ids_sub is None else [n_ids, cand_ids, atk_ids])
        src, dst = adj_attack.nonzero(as_tuple=True)

        data = {'weight': adj_attack[src, dst]} if training else None
        graph_perturbed.add_edges(src+N, indices[dst], data)
        graph_perturbed.add_edges(indices[dst], src+N, data) # symmetric (undirected edges)
        graph_perturbed = dgl.remove_self_loop(graph_perturbed)
        graph_perturbed = dgl.add_self_loop(graph_perturbed)
        
        if need_sequence:
            sequences = self.data.get_updated_sequences(graph_perturbed, n_ids, cand_ids,
                                                        atk_ids, training).reshape((-1, 7, self.data.feat_dim))
            return graph_perturbed, sequences
        
        return graph_perturbed

    def load_attack_model(self, best: bool=True) -> None:
        
        model_path = self.result_manager.best_model_path if best else self.result_manager.model_path        
        self.model.load_state_dict(torch.load(model_path))
        
    # Train the attack model
    def run(self) -> Dict:
        
        self.result_manager.start_train(self.epoch_st)
        
        for epoch in range(self.epoch_st, self.config["epochs"]+1+1):
            epoch_last = epoch-1
            validation = (((epoch_last) % self.config["valid_epoch"] == 0) or \
                          ((epoch_last) == self.config["epochs"])) and epoch_last > 0
            if validation:
                stop = self.validation(epoch_last) or (epoch == self.config["epochs"]+1)
                # Early Stopping
                if stop:
                    line = f"Early stopping at epoch {epoch_last}"
                    self.result_manager.write_log(line, ExperimentState.TEST)
                    break
            self.train_epoch()
    
        self.result_manager.end_train()
        test_results = self.test()
        
        return test_results

    def train_epoch(self):
        
        self.result_manager.start_epoch()
        self.model.train()
        torch.cuda.empty_cache()

        loss_l_epoch = [] # loss list for each epoch
        
        need_sequence = (self.surrogate_type == "GAGA")
        epoch = self.result_manager.epoch
        b_epoch = self.es_scheduler.b_epoch
        train_loader = self.data.get_train_loader()
        with tqdm(train_loader) as pbar:
            pbar.set_description(f'Epoch: {str(epoch).zfill(4)} ({epoch - b_epoch}/{self.es_scheduler.patience})')

            for t_ids, node_budgets, edge_budgets in pbar:
                torch.cuda.empty_cache()
                loss_l_batch = [] # loss list for each mini-batch (for training)
                
                for t_id, node_budget, edge_budget in zip(t_ids, node_budgets, edge_budgets):
                    node_budget = node_budget.item()
                    edge_budget = edge_budget.item()
                    
                    # Get input for the model
                    n_ids = get_n_ids(self.data.target_assignment, t_id) if self.multi_target else [t_id]
                    n_ids = torch.tensor(n_ids, dtype=int)
                    
                    # Use the model here
                    output = self.get_perturbed_graph(n_ids, node_budget, edge_budget, training=True,
                                                      need_sequence=need_sequence)
                    if output is None:
                        print("[Warning] perturbed graph is None")
                        continue
                    
                    sequences = None
                    if need_sequence:
                        graph_perturbed, sequences = output
                    else:
                        graph_perturbed = output
                        
                    loss = self.loss_function(graph_perturbed, n_ids, training=True, sequences=sequences)
                    loss_l_batch.append(loss)
                    loss_l_epoch.append(loss.item()) # for printing
                    
                    postfix_new = ", ".join([f"num_targets: {str(len(n_ids)):>4}",
                        f"node_budget: {str(node_budget):>3}",
                        f"edge_budget: {str(edge_budget):>5}",
                        f"loss: {np.mean(loss_l_epoch):.5f}"])
                pbar.set_postfix_str(postfix_new)
                    
                # Backpropagation
                train_loss = torch.stack(loss_l_batch).mean()
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
        
        # End of epoch
        train_loss = np.mean(loss_l_epoch)
        max_mem = torch.cuda.max_memory_reserved()
        self.result_manager.end_epoch(self.es_scheduler.b_epoch, self.es_scheduler.patience, train_loss, max_mem)

        torch.save(self.model.state_dict(), self.result_manager.model_path)
        torch.save(self.optimizer.state_dict(), self.result_manager.optimizer_path)
        
        self.model.step()

    def validation(self, epoch) -> bool:
        
        model = self.model
        model.eval()
        val_results = self.check_performance("surrogate", "valid", attack=True)
        update, stop, b_val_results = self.es_scheduler.step(epoch, val_results)
        
        if update:
            torch.save(model.state_dict(), self.result_manager.best_model_path)
        
        self.result_manager.write_results(val_results, ExperimentState.VALIDATION)
        self.result_manager.write_results(b_val_results, ExperimentState.VALIDATION_BEST)
        
        return stop

    def test(self) -> Dict:
        
        self.result_manager.write_log(f"Restore model from epoch {self.es_scheduler.b_epoch}", ExperimentState.TEST)
        self.load_attack_model(best=True)
        self.model.eval()
        test_results = self.check_performance("victim", "test", attack=True)
        self.result_manager.write_results(test_results, ExperimentState.END)
        
        return test_results
