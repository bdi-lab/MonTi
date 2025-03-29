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
  doi={10.48550/arXiv.2412.18370}
}
```

## Requirements

We used Python 3.8, Pytorch 1.13.1, and DGL 1.1.2 with cudatoolkit 11.7.

## Usage

We conducted our experiments using GeForce RTX 2080 Ti, RTX 3090, or RTX A6000 with Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz, running on Ubuntu 18.04.5 LTS.

**Note: For experiments using GAGA, the initial pre-processing step may take 5-10 minutes before the actual training/inference begins for the first run.**

### Setup

Before running the commands below, please download the necessary pre-trained models and datasets:
1.  Download `best_models.zip` and `Datasets.zip` from the following Google Drive link:
   https://drive.google.com/drive/folders/1E2FE8zDIS4XZA6Fxe5l3hTrXUnfhJ2gr?usp=sharing
2.  Extract the downloaded files into their corresponding directories:
    * Extract `best_models.zip` into the `./best_models` directory.
    * Extract `Datasets.zip` into the `./Datasets` directory.

### Reproducing Results

To reproduce the results of the best models, you can run the following command:

```console
python run.py --config_path "./best_models/configs/GCN-GAGA-gossipcop_0.05-best_config.json" --inference
```

You can modify the `config_path` to experiment with different configurations and reproduce the other results.

### Training from Scratch

You can run the following command to train MonTi from scratch with the best configuration:

```console
python run.py --config_path "./best_models/configs/GCN-GAGA-gossipcop_0.05-best_config.json"
```

You can modify the `config_path` to experiment with different configurations and reproduce the other results.

Training logs and checkpoints will be saved in `./experimental_results`.

## Hyperparameters

The configuration files and checkpoints of the best models are in `./best_models.`

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
- `template.json`: The template configuration file for multi-target attack experiments.
- `template_single.json`: The template configuration file for single-target attack experiments.

### `./best_models`

- `checkpoints`: Pickle files containing the checkpoint of the trained model.
- `configs`: JSON files containing the best model configuration.

### `./Datasets`

- `Benchmark`: Contains DGLGraph file, data splits, and target set assignment matrix for _OGB-Prod_ and _PubMed_.
- `GossipCop`: Contains DGLGraph file, data splits, and target set assignment matrix for _GossipCop-S_.
- `YelpChi_RTR`: Contains DGLGraph file, data splits, and target set assignment matrix for _YelpChi_.

### `./models`

- `pretrained_models`: Contains configurations and checkpoints of surrogate and victim models.
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
