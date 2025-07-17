# coding: utf-8
import datetime
import logging
import os
import sys
import time

import numpy as np

from _benchmark_problems.utils import get_arguments, set_seed
from mocahesp.mocahesp_bo import MOCAHESPBO
from mocahesp.mocahesp_casmo import MOCAHESPCasmo
from mocahesp.mocahesp_bounce import MOCAHESPBounce

# Arguments
input_dict = get_arguments()
obj_func = input_dict['f']
func_name = input_dict['func_name']
max_evals = input_dict['max_evals']
shifted = input_dict['shifted']
solver = input_dict['solver']
seed = input_dict['seed']
n_init = input_dict['n_init']

set_seed(seed=seed)
np.set_printoptions(linewidth=np.inf)

print(f'MOCA-HESP-{solver.upper()}: {obj_func.name}-{obj_func.input_dim}D function with max_evals={max_evals}')

logging.root.setLevel(logging.DEBUG)
logging.getLogger('').addHandler(logging.StreamHandler(sys.stdout))    
logging.info('*'*20 + f'Start time: {datetime.datetime.now()}')

map_mocahesp = {
    'bo': MOCAHESPBO,
    'casmo': MOCAHESPCasmo,
    'bounce': MOCAHESPBounce
}

mocahesp = map_mocahesp[solver](obj_func=obj_func,
                                 n_init=n_init,
                                 max_evals=max_evals, 
                                 func_name=func_name,
                                 )
start_time = time.time()
results = mocahesp.optimize()
end_time = time.time()
logging.info('*'*20 + f' Elapsed time: {datetime.timedelta(seconds=end_time-start_time)} ' + '*'*20)
