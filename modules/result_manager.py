import os
import time
import logging
import logging.handlers

from typing import Dict
from utils.constants import RESULT_ROOT_DIR, ExperimentState, LOG_FORMAT, LOG_DATE_FORMAT

class ResultManager:
    """
    ResultManager manages and saves results of model training and testing based on config.
        - Experiment log txt file is saved in self.log_path.
        - Model checkpoint pickle file is saved in self.model_path.
        - Best model checkpoint pickle file is saved in self.best_model_path.
    """
    
    def __init__(self, config: Dict, root_dir=RESULT_ROOT_DIR) -> None:
        
        exp_name = config["exp_name"]
        dataset = config["dataset"]
        model = config["model"]
        seed = config["seed"]
        
        node_budget = config["node_budget"]
        edge_budget = config["edge_budget"]
        surrogate_model = config["surrogate_model"]
        victim_model = config["victim_model"]
        
        dir_name = f"{dataset}-{seed}-{model}-{node_budget}-{edge_budget}-{surrogate_model}-{victim_model}"
        config_name = config["config_path"].split('/')[-1][:-5] # To remove '.json'
        
        save_dir = os.path.join(root_dir, exp_name, dir_name)
        log_dir = os.path.join(save_dir, "logs")
        model_dir = os.path.join(save_dir, "saved_models")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        self.log_path = os.path.join(log_dir, f"{config_name}.log")
        self.model_path = os.path.join(model_dir, f"{config_name}.pickle")
        self.best_model_path = os.path.join(model_dir, f"{config_name}_best.pickle")
        self.optimizer_path = os.path.join(model_dir, f"{config_name}_opt.pickle")
         
        self.init_log(config)
    
    def init_log(self, config: Dict) -> None:
        
        logging.basicConfig(filename=self.log_path, filemode='w',
                            level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        streamHandler = logging.StreamHandler() # For printing to console
        streamHandler.setFormatter(formatter)

        logger = logging.getLogger() # root logger
        logger.addHandler(streamHandler)
        
        
        self.write_log(config, ExperimentState.CONFIG)

    def write_log(self, line: str, exp_state: ExperimentState, log_level=logging.INFO) -> None:
        
        line = f"[{exp_state.name}] {line}"
        logging.log(log_level, line)
    
    def write_results(self, results: Dict[str, float], exp_state: ExperimentState) -> None:
        
        line = "<PERFORMANCE> " + " / ".join([f"{metric}: {value:.4f}" for metric, value in results.items()])
        self.write_log(line, exp_state)

    def start_train(self, epoch_st: int) -> None:
        
        self.start_time = time.time()
        self.elapsed_time = 0
        self.mean_epoch_time = 0
        self.epoch = epoch_st
    
    def start_epoch(self) -> None:
        
        self.epoch_start_time = time.time()
    
    def end_epoch(self, b_epoch, patience, loss, max_mem) -> None:
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - self.epoch_start_time
        self.elapsed_time = epoch_end_time - self.start_time
        
        # Cumulative moving average of epoch time
        self.mean_epoch_time = float(self.mean_epoch_time*(self.epoch-1) + epoch_time) / float(self.epoch)
        
        line = f"Epoch: {str(self.epoch).zfill(4)} ({self.epoch - b_epoch}/{patience})"
        line += f" / Best Epoch: {str(b_epoch).zfill(4)} / Loss: {loss:.5f}"
        line += f" / Epoch Time: {epoch_time:.3f}s / Mean Epoch Time: {self.mean_epoch_time:.3f}s / Elapsed Time: {self.elapsed_time:.3f}s"
        line += f" / Max GPU Mem: {max_mem/(2.**30):.4f} GiB"
        self.write_log(line, ExperimentState.TRAIN if b_epoch != 0 else ExperimentState.START)
        
        self.epoch += 1
    
    def end_train(self) -> None:
        
        self.elapsed_time = time.time() - self.start_time
        line = f"Total training time: {self.elapsed_time:.4f}s"
        self.write_log(line, ExperimentState.TEST)
    