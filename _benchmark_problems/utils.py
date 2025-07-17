import argparse
from collections import OrderedDict

import botorch
import numpy as np
import random
import torch

# from .function_realworld_bo.functions_realworld_bo import *
# from .function_realworld_bo.functions_mujoco import *
# from .function_realworld_bo.functions_xgboost import *
# from .functions_bo import *
# from .highdim_functions import *
# from .lasso_benchmark import *
from mocahesp.function_wrapper import *

def get_arguments():
    parser = argparse.ArgumentParser(description='Process inputs')
    parser.add_argument('-f', '--func', help='specify the test function')
    parser.add_argument('-d', '--dim', type=int, help='specify the problem dimensions', default=10)
    parser.add_argument('-n', '--max_evals', type=int, help='specify the maxium number of evaluations to collect in the search')
    parser.add_argument('--shifted', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--solver', type=str, help='specify the solver', default='bo')
    parser.add_argument('--seed', type=int, help='', default=1)
    parser.add_argument('-td', '--targetdim', type=int, help='specify the target dimensions for projection-based methods', default=10)
    parser.add_argument('-init', '--n_init', type=int, help='number of initialization', default=20)


    args = parser.parse_args()


    dim = args.dim
    func = args.func.lower()
        
    f = get_objective_function(func_name=func, shifted=args.shifted)
    max_evals = args.max_evals
    dict = {
        'func_name': func,
        "f": f,
        'max_evals': max_evals,
        'solver': args.solver,
        'seed': args.seed,
        'shifted': args.shifted,
        'n_init': args.n_init,
    }
    return dict

def get_objective_function(func_name, shifted=False):
    if func_name == 'ackley53m':
        f = Ackley53m(shifted=shifted)
    elif func_name == 'ackley20c':
        f = Ackley20c(shifted=shifted)
    elif func_name == 'antibody':
        f = AntiBodyDesign(shifted=shifted)
    elif func_name == 'maxsat28':
        f = MaxSAT28(shifted=shifted)
    elif func_name == 'maxsat125':
        f = MaxSAT125(shifted=shifted)
    elif func_name == 'cco':
        f = CellularNetworkOpt(shifted=shifted)
    elif func_name == 'labs':
        f = LABS(shifted=shifted)
    elif func_name == 'svm':
        f = SVMMixed(shifted=shifted)
    else:
        raise NotImplementedError()
    return f

def get_bound(bounds):
    if isinstance(bounds, OrderedDict):
        return np.array([val for val in bounds.values()])
    else:
        return np.array(bounds)
    
def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    botorch.manual_seed(seed)

def print_both(text, filename=''):
    print(text)