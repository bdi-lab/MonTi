import enum
from enum import Enum

DATA_ROOT_DIR = "./Datasets"
RESULT_ROOT_DIR = "./experimental_results"
CONFIG_ROOT_DIR = "./configs"
PRETRAINED_MODELS_DIR = "./models/pretrained_models"

DIR_NAME_DICT = {
    "gossipcop_0.05": "GossipCop",
    "yelpchi_rtr": "YelpChi_RTR",

    "ogbproducts": "Benchmark/ogbproducts",
    "pubmed": "Benchmark/pubmed"
}

GRAPH_FILE_NAME_DICT = {
    "gossipcop_0.05": "GossipCop_0.05",
    "yelpchi_rtr": "YelpChi",
    
    "ogbproducts": "ogbproducts",
    "pubmed": "pubmed"
}

SINGLE_TARGET_DATASETS = ["ogbproducts", "pubmed"]
PUBLIC_SPLIT_DATASETS = ["ogbproducts", "pubmed"]
DISCRETE_DATASETS = []

NEED_SAMPLER_MODELS = ["GCN", "GraphSAGE", "GAT", "CAREGNN", "PCGNN"]
ATTACK_MODELS = ["MonTi"]

EVAL_METRICS_ATTACK = ['misclf_rate']

EPSILON = 1e-10
VICTIM_SEED = 717

LOG_MAX_BYTES = 10 * 1024 * 1024 # Bytes
LOG_LINE_PATTERN = r"\[[A-Z0-9\/\:\_\s]+\]"
PERFORMANCE_TOKEN = "<PERFORMANCE>"
LOG_FORMAT = '[%(asctime)s.%(msecs)03d] [%(levelname)s]: %(message)s'
LOG_DATE_FORMAT = '%Y/%m/%d %H:%M:%S'

class ExperimentState(Enum):
    UNDEFINED = enum.auto()
    CONFIG = enum.auto() # for configuration lines
    START = enum.auto() # training before the first validation
    TRAIN = enum.auto() # training after the first validation
    VALIDATION = enum.auto() # validation (current performance)
    VALIDATION_BEST = enum.auto() # validation (best performance)
    TEST = enum.auto() # test
    END = enum.auto() # end of the experiment