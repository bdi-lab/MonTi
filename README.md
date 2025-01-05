# Unveiling the Threat of Fraud Gangs to Graph Neural Networks: Multi-Target Graph Injection Attacks against GNN-Based Fraud Detectors

This is the official code and data of the following [paper](https://arxiv.org/abs/2412.18370):

> Jinhyeok Choi, Heehyeon Kim, and Joyce Jiyoung Whang, Unveiling the Threat of Fraud Gangs to Graph Neural Networks: Multi-Target Graph Injection Attacks against GNN-Based Fraud Detectors, The 39th Annual AAAI Conference on Artificial Intelligence (AAAI), 2025

All codes are written by Jinhyeok Choi (cjh0507@kaist.ac.kr). When you use this code, please cite our paper.

```bibtex
@article{monti,
  author={Jinhyeok Choi and Heehyeon Kim and Joyce Jiyoung Whang},
  title={Unveiling the Threat of Fraud Gangs to Graph Neural Networks: Multi-Target Graph Injection Attacks against GNN-Based Fraud Detectors},
  year={2024},
  journal={arXiv preprint arXiv.2412.18370},
  doi = {10.48550/arXiv.2412.18370}
}
```

## Requirements

We used Python 3.8, Pytorch 1.13.1, and DGL 1.1.2 with cudatoolkit 11.7.

Running this code requires about 6 GiB of GPU memory.

## Usage

### Reproducing Results

To reproduce the results of the aforementioned setting, you can run the following command:

```console
python run.py --inference
```

### Training from Scratch

You can run the following command to train MonTi from scratch with the best configuration on the aforementioned setting:

```console
python run.py
```

Training logs and checkpoints will be saved in `./experimental_results`.

## Hyperparameters

The configuration and checkpoint of the best model on this setting is in `./best_models.`

The list of arguments of the configuration file:

```json
{
	"exp_name": "Name of the experiment.",
	"model": "Attack model type to be used, e.g., 'MonTi'.",
	"surrogate_model": "Surrogate model type to be used, e.g., 'GCN'.",
	"victim_model": "Victim model type to be used, e.g., 'GAGA'.",
	"dataset": "Dataset to be used, e.g., 'gossipcop_0.05'.",
	"multi_target": "Whether the attack is multi-target or not.",
	"split": "Split number to be used.",
	"seed": "Random seed for the reproducibility of results.",
	"node_budget": "Node budget contorl paramter rho.",
	"edge_budget": "Edge budget contorl paramter xi.",

	"k_exp": "Hop number to extract neighbors from target set.",
	"masking": "Whether MonTi use the attack node masking or not.",
	"max_candidates": "Maximum number of candidates to consider during candidate selection (MonTi).",
	"num_layers": "Number of layers in the Transformer encoder.",
	"num_heads": "Number of heads in the Transformer encoder.",
	"emb_dim": "Dimension of each embedding.",
	"ff_dim": "Dimension of the feed-forward network in the Transformer encoder.",
	"dropout": "Dropout rate used in the model.",

	"initial_temp": "Initial temperature used in the temperature scheduling.",
	"min_temp": "Minimum temperature used in the temperature scheduling.",
	"temp_decay_rate": "The rate at which the temperature is decayed over training epochs.",
	"initial_eps": "Initial exploration parameter.",
	"min_eps": "Minimum exploration parameter.",
	"eps_decay_rate": "The rate at which the exploration parameter is decayed over training epochs.",

	"lr": "Learning rate for training.",
	"weight_decay": "Weight decay rate used for regularization during training.",
	"epochs": "Total number of training epochs.",
	"valid_epoch": "Number of epochs after which the validation is performed.",
	"patience": "Number of epochs to wait before early stopping if no progress on the validation set.",
	"batch_size": "Size of the batches used during training.",
	"batch_size_pre": "Size of the batches for pre-computing node representations.",

    "cuda_id": "CUDA device ID (GPU ID) to be used for training if available."
}
```

## Description of each file and directory

### `./`

- `run.py`: The main script to start the model training and evaluation.

### `./best_models`

- `GAGA-gossipcop_0.05-best_config`: JSON file containing the best model configuration.
- `GAGA-gossipcop_0.05-best_model`: Pickle file containing the checkpoint of the trained model.

### `./Datasets`

- `GossipCop`
  - `data_split`: Contains data splits and target set assignment matrix.
  - `adj.npz`: Adjacency matrix of *GossipCop-S* (csc matrix).
  - `features.npy`: Node features of *GossipCop-S*.
  - `labels.npy`: Node labels of *GossipCop-S*.

### `./models`

- `pretrained_models`: Contains configurations and checkpoints of surrogate and victim models. (GCN and GAGA)
- `attack_model.py`: Contains the definition of MonTi.
- `victim_model.py`: Contains the definition of GCN and GAGA.

### `./modules`

- `data_handler.py`: Manages data loading and preprocessing.
- `experiment_handler.py`: Handles the setup and execution of experiments.
- `result_manager.py`: Manages the logging and saving of experiment results.
- `schedulers.py`: Contains schedulers for early stopping and exponential decaying.

### `./utils`

- `constants.py`: Defines constants used across the project.
- `utils.py`: Helper functions used throughout the project.
