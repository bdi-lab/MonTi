import os
import random

import torch
import dgl
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import accuracy_score

from typing import Dict, Any, Tuple
from utils.constants import (DATA_ROOT_DIR, DIR_NAME_DICT, GRAPH_FILE_NAME_DICT,
                             SINGLE_TARGET_DATASETS, PUBLIC_SPLIT_DATASETS, DISCRETE_DATASETS)

def create_dir(dir_path: str) -> None:
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    except OSError:
        print("Error: Failed to create the directory.")

def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility. 
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed) 

############################## FUNCTIONS FOR DATA LOADING ##############################

def load_data(data_name: str, split: int=0, root_dir: str=DATA_ROOT_DIR, multi_target: bool=True) -> Dict[str, Any]:
    """
    Load specified graph and data split.

    Args:
        data_name (str): name of dataset
        split (int, optional): the split id [0 ~ 9]. It is not used when single_target == True and there is a public split. Defaults to 0.
        root_dir (str, optional): the root directory of datasets. Defaults to "./Datasets".
        multi_target (bool, optional): if it sets to False, node-level split will be returned. Else, group-level split will be returned. Defaults to True.

    Returns:
        Dict[str, Any] whose keys are:
        
        if multi_target == False
            "graph", "discrete", "idx_train", "idx_valid", "idx_test"
        elif multi_target == True
            "graph", "discrete", "idx_train", "idx_valid", "idx_test", "target_assignment", "target_idx_train", "target_idx_valid", "target_idx_test"
    """
    
    assert (data_name in DIR_NAME_DICT.keys()) and (data_name in GRAPH_FILE_NAME_DICT.keys())
    
    if not (multi_target or (data_name in SINGLE_TARGET_DATASETS)) or \
        (multi_target and (data_name in SINGLE_TARGET_DATASETS)):
        raise NotImplementedError
    
    dir_name = DIR_NAME_DICT[data_name]
    dir_path = os.path.join(root_dir, dir_name)
    graph_file_name = GRAPH_FILE_NAME_DICT[data_name] + ".bin"
    graph_path = os.path.join(dir_path, graph_file_name)

    graphs, _ = dgl.load_graphs(graph_path)
    graph = graphs[0]

    split = "public" if data_name in PUBLIC_SPLIT_DATASETS else split
    idx_train = np.load(os.path.join(dir_path, "data_split", f"train_idx_surro_{split}.npy"))
    idx_valid = np.load(os.path.join(dir_path, "data_split", f"valid_idx_surro_{split}.npy"))
    idx_test = np.load(os.path.join(dir_path, "data_split", f"test_idx_surro_{split}.npy"))

    data = dict()
    data["graph"] = graph
    data["discrete"] = data_name in DISCRETE_DATASETS
    data["idx_train"] = idx_train
    data["idx_valid"] = idx_valid
    data["idx_test"] = idx_test
    
    if multi_target:
        target_assignment = sp.load_npz(os.path.join(dir_path, "data_split", f"target_assign_matrix_{split}.npz"))
        target_idx_train = np.load(os.path.join(dir_path, "data_split", f"train_target_mask_{split}.npy"))
        target_idx_valid = np.load(os.path.join(dir_path, "data_split", f"valid_target_mask_{split}.npy"))
        target_idx_test = np.load(os.path.join(dir_path, "data_split", f"test_target_mask_{split}.npy"))
        data["target_assignment"] = target_assignment
        data["target_idx_train"] = target_idx_train
        data["target_idx_valid"] = target_idx_valid
        data["target_idx_test"] = target_idx_test
    
    return data

def get_n_ids(target_assignment: sp.spmatrix, t_id: int) -> np.ndarray:
    """
    Get the node IDs of the target set with the given ID t_id.

    Args:
        target_assignment (sp.spmatrix): The target assignment matrix.
        t_id (int): The ID of the target set.

    Returns:
        np.ndarray: The node IDs of the target set.
    """
    n_ids = target_assignment[t_id].nonzero()[1]
    return n_ids

def get_target_neighbor_size(graph: dgl.DGLGraph, target_assignment: sp.spmatrix) -> np.ndarray:
    """
    Get the size of the set of neighbors (including target nodes themselves) for each target set.

    Args:
        graph (dgl.DGLGraph): The graph object.
        target_assignment (sp.spmatrix): The target assignment matrix.

    Returns:
        np.ndarray: The target neighbor size of each target set.
    """
    
    adj = sp.csc_matrix(graph.adj_external(scipy_fmt="coo"))
    groups, nodes = target_assignment.nonzero()
    row, col = adj[nodes].nonzero()

    target_neighbor_size = np.array(sp.coo_matrix((np.ones_like(col), (groups[row], col))).sum(1)).reshape(-1)
    
    return target_neighbor_size

################################# FUNCTIONS FOR MODELS #################################

def get_complment_indices(N: int, idx: torch.Tensor) -> torch.Tensor:
    """
    Get the indices of the complement indices of the given indices, considering the range [0, N-1].
    
    Args:
        N (int): The total number of indices.
        idx (Tensor): A tensor containing indices.

    Returns:
        Tensor: A tensor containing the indices of the complement indices.
    """
    all_indices = torch.arange(N)
    mask = torch.ones(N, dtype=bool)
    mask[idx] = 0
    complement_indices = all_indices[mask]

    return complement_indices

def sample_gumbel_noise(logits: torch.Tensor, epsilon: float=1e-20) -> torch.Tensor:
    """
    Generate Gumbel noise based on the logits.

    Args:
        logits (Tensor): A tensor containing logits.
        epsilon (float): A small number to prevent numerical instability.

    Returns:
        Tensor: A tensor with Gumbel noise added to the logits.
    """
    # Generate uniform random numbers and apply the Gumbel trick
    U = torch.rand_like(logits)
    return -torch.log(-torch.log(U + epsilon) + epsilon)

def gumbel_softmax(logits: torch.Tensor, temperature: float=1.0,
                   epsilon: float=1.0,
                   dim: int=-1,
                   randomness: bool=True) -> torch.Tensor:
    """
    Perform Gumbel-Softmax sampling.

    Args:
        logits (Tensor): A tensor containing logits.
        temperature (float): Temperature parameter for Gumbel-Softmax.
        epsilon (float): Exploration coefficient parameter for Gumbel-Softmax.
        dim (int): The dimension along which to perform Gumbel-Softmax sampling.
        randomness (bool): If set to False, conduct softmax without Gumbel noise.

    Returns:
        Tensor: A tensor with Gumbel-Softmax sampled probabilities.
    """

    if randomness:
        gumbels = sample_gumbel_noise(logits)*epsilon + logits  # Gumbel noise addition
        gumbels /= temperature
    else: 
        gumbels = logits
    
    # For stabilization
    gumbels_max, _ = gumbels.max(dim=dim, keepdim=True)
    gumbels -= gumbels_max
    
    y_soft = torch.softmax(gumbels, dim=dim)

    return y_soft

def st_top_k(y_soft: torch.Tensor, k: int, dim: int=-1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform ST (Straight-Through) top-k selection.

    Args:
        y_soft (Tensor): A tensor containing soft probabilities.
        k (int): Number of top elements to select.
        dim (int): The dimension along which to perform top-k selection.

    Returns:
        Tensor: A tensor with the top-k elements selected using the ST method.
    """

    # Apply ST (Straight-Through) Estimator
    _, indices = y_soft.topk(k, dim=dim)
    y_hard = torch.zeros_like(y_soft).scatter_(-1, indices, 1.0)

    # Use hard selection in forward pass and soft selection in backward pass
    y = y_hard - y_soft.detach() + y_soft
    
    return y, indices

def st_gumbel_top_k(logits: torch.Tensor, k: int,
                    temperature: float=1.0,
                    epsilon: float=1.0,
                    dim: int=-1,
                    randomness: bool=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform ST (Straight-Through) Gumbel top-k selection.

    Args:
        logits (Tensor): A tensor containing logits.
        k (int): Number of top elements to select.
        temperature (float): Temperature parameter for Gumbel-Softmax.
        epsilon (float): Exploration coefficient parameter for Gumbel-Softmax.
        dim (int): The dimension along which to perform Gumbel-Softmax sampling.
        randomness (bool): If set to False, conduct top-k selection without Gumbel noise.

    Returns:
        Tensor: A tensor with the top-k elements selected using the ST Gumbel method.
    """
    # Add Gumbel noise to logits and apply softmax
    y_soft = gumbel_softmax(logits, temperature, epsilon, dim, randomness)

    # Apply ST (Straight-Through) Estimator
    y, indices = st_top_k(y_soft, k, dim)
    
    return y, indices

########################################################################################

def misclassification_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1.0 - accuracy_score(y_true, y_pred)