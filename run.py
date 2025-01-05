import torch
import argparse, json
import os

from modules.result_manager import ResultManager
from modules.data_handler import DataHandler
from modules.experiment_handler import ExperimentHandler

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./best_models/configs/GCN-GAGA-gossipcop_0.05-best_config.json',
                        help="Path to the experiment config file")
    parser.add_argument('--check_clean_performance', action='store_true',
                        help="Check the performance of the victim model on the clean graph without logging. The config file should be the one for attack experiment")
    parser.add_argument('--inference', action="store_true",
                        help="Only inference, no training. The config file should be the one in best_models directory")
    args = vars(parser.parse_args())
    return args

def main(args) -> None:

    os.environ["OMP_NUM_THREADS"] = "8"
    torch.set_num_threads(8)
    torch.cuda.empty_cache()

    # Read the config file
    config_path = args['config_path']
    with open(config_path) as f:
        config = json.load(f)
    config['config_path'] = config_path
    
    ########### About ablation studies ###########
    config.setdefault("random_candidates", False)
    config.setdefault("shared_parameters", False)
    config.setdefault("without_candidates", False)
    config.setdefault("pos_encoding", True)
    config.setdefault("ada_budget", True)
    ##############################################
    
    check_clean_performance = args['check_clean_performance']
    inference = args['inference']
    assert not (check_clean_performance and inference)
    
    result = ResultManager(config) if not (check_clean_performance or inference) else None
    data = DataHandler(config)
    exp = ExperimentHandler(config, data, result, check_clean_performance)

    if check_clean_performance:
        return
    
    elif inference:
        model_path = os.path.join("best_models", "checkpoints", f"{exp.surrogate_type}-{exp.victim_type}-{data.data_name}-best_model.pickle")
        exp.model.load_state_dict(torch.load(model_path))
        exp.model.eval()
        exp.check_performance(model="victim", set_type="test", attack=True)
    
    else:
        exp.run()

if __name__ == '__main__':
    args = get_arguments()
    main(args)