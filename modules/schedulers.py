from typing import List, Dict, Tuple
from utils.constants import EPSILON, EVAL_METRICS_ATTACK

class EarlyStopScheduler():
    
    def __init__(self, patience, metrics: List[str]=EVAL_METRICS_ATTACK) -> None:
        
        self.patience = patience
        self.metrics = metrics # HB (Higher is Better) metrics only
        
        self.b_epoch = 0
        self.b_results = dict()
        
        for metric in metrics:
            self.b_results[metric] = EPSILON # low value to prevent zero division
    
    # It is called per validation
    def step(self, epoch: int, results: Dict[str, float]) -> Tuple[bool, bool, Dict[str, float]]:
        
        gain = 0
        
        # results should contain all self.metrics
        for metric in self.metrics:
            
            curr = results[metric]
            best = self.b_results[metric]
            gain += (curr - best) / best
        
        # Renew best results
        update = False
        if gain > 0:
            
            self.b_epoch = epoch
            self.b_results = results
            update = True
        
        stop = (epoch - self.b_epoch) >= self.patience
        
        return update, stop, self.b_results

class ExpDecayEpsilonScheduler:
    
    def __init__(self, initial_temp: float=1e2,
                 min_temp: float=1e-2,
                 decay_rate: float=0.8) -> None:
        self.current_eps: float = initial_temp
        self.min_eps: float = min_temp
        self.decay_rate: float = decay_rate

    def step(self) -> None:
        # Apply exponential decay
        self.current_eps = max(self.min_eps, self.current_eps * self.decay_rate)

    def get_epsilon(self) -> float:
        return self.current_eps

class ExpDecayTemperatureScheduler:
    
    def __init__(self, initial_temp: float=1e2,
                 min_temp: float=1e-2,
                 decay_rate: float=0.8) -> None:
        
        self.current_temp: float = initial_temp
        self.min_temp: float = min_temp
        self.decay_rate: float = decay_rate

    def step(self) -> None:
        # Apply exponential decay
        self.current_temp = max(self.min_temp, self.current_temp * self.decay_rate)

    def get_temperature(self) -> float:
        return self.current_temp