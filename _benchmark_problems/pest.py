import logging
from copy import deepcopy

import numpy as np
import torch

# from .base import TestFunction

PESTCONTROL_N_CHOICE = 5
PESTCONTROL_N_STAGES = 25


def _pest_spread(curr_pest_frac, spread_rate, control_rate, apply_control):
    if apply_control:
        next_pest_frac = (1.0 - control_rate) * curr_pest_frac
    else:
        next_pest_frac = spread_rate * (1 - curr_pest_frac) + curr_pest_frac
    return next_pest_frac


def _pest_control_score(x, seed=None):
    logging.debug(f"running pest w/ seed {seed}")
    U = 0.1
    n_stages = x.size
    n_simulations = 100

    init_pest_frac_alpha = 1.0
    init_pest_frac_beta = 30.0
    spread_alpha = 1.0
    spread_beta = 17.0 / 3.0

    control_alpha = 1.0
    control_price_max_discount = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.0}
    tolerance_develop_rate = {1: 1.0 / 7.0, 2: 2.5 / 7.0, 3: 2.0 / 7.0, 4: 0.5 / 7.0}
    control_price = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.5}
    # below two changes over stages according to x
    control_beta = {1: 2.0 / 7.0, 2: 3.0 / 7.0, 3: 3.0 / 7.0, 4: 5.0 / 7.0}

    payed_price_sum = 0
    above_threshold = 0

    if seed is not None:
        init_pest_frac = np.random.RandomState(seed).beta(init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,))
    else:
        init_pest_frac = np.random.beta(init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,))
    curr_pest_frac = init_pest_frac
    for i in range(n_stages):
        if seed is not None:
            spread_rate = np.random.RandomState(seed).beta(spread_alpha, spread_beta, size=(n_simulations,))
        else:
            spread_rate = np.random.beta(spread_alpha, spread_beta, size=(n_simulations,))
        do_control = x[i] > 0
        if do_control:
            if seed is not None:
                control_rate = np.random.RandomState(seed).beta(control_alpha, control_beta[x[i]], size=(n_simulations,))
            else:
                control_rate = np.random.beta(control_alpha, control_beta[x[i]], size=(n_simulations,))
            next_pest_frac = _pest_spread(curr_pest_frac, spread_rate, control_rate, True)
            # torelance has been developed for pesticide type 1
            control_beta[x[i]] += tolerance_develop_rate[x[i]] / float(n_stages)
            # you will get discount
            payed_price = control_price[x[i]] * (
                    1.0 - control_price_max_discount[x[i]] / float(n_stages) * float(np.sum(x == x[i])))
        else:
            next_pest_frac = _pest_spread(curr_pest_frac, spread_rate, 0, False)
            payed_price = 0
        payed_price_sum += payed_price
        above_threshold += np.mean(curr_pest_frac > U)
        curr_pest_frac = next_pest_frac

    return payed_price_sum + above_threshold


class BasePestControl():
    """
	Pest Control Problem.

	"""
    def __init__(self, random_seed=0, shifted=False, **kwargs):
        super(BasePestControl, self).__init__(**kwargs)
        self.name = 'pest'
        self.categorical_idx_m = list(range(PESTCONTROL_N_STAGES))    
        self.discrete_idx_m = []
        self.continuous_idx_m = []
        self.bounds_m = np.array(
            [[0, PESTCONTROL_N_CHOICE-1] for _ in range(PESTCONTROL_N_STAGES) ]      
        )

        self.discrete_dim_m = len(self.discrete_idx_m)
        self.categorical_dim_m = len(self.categorical_idx_m)
        self.continuous_dim_m = len(self.continuous_idx_m)
        self.categorical_vertices_m = [self.bounds_m[cat_idx][1] - self.bounds_m[cat_idx][0] + 1 
                                       for cat_idx in self.categorical_idx_m]

        self.dim_m = self.categorical_dim_m + self.discrete_dim_m + self.continuous_dim_m
        self.seed = random_seed
        self.shifted = shifted
        if self.shifted:
            self.offset = np.random.RandomState(seed=2024).choice(PESTCONTROL_N_CHOICE, PESTCONTROL_N_STAGES)
            print(f'offset={np.around(self.offset, 0)}')
            self.name = f'shifted-{self.name}'
        else:
            self.offset = np.zeros(PESTCONTROL_N_STAGES)


    def eval(self, _x: np.ndarray):
        if _x.ndim == 2:
            x = _x.squeeze(0)
        else:
            x = deepcopy(_x)
        assert len(x) == self.dim_m
        if self.shifted:
            x = (x + self.offset) % PESTCONTROL_N_CHOICE
        evaluation = _pest_control_score(x, seed=self.seed)
        res = float(evaluation) * np.ones((1,))
        return res
    
    def func_core(self, x):
        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                X = np.copy(x)
            elif x.ndim == 2:
                X = x.flatten()
            else: 
                raise NotImplementedError()
        elif isinstance(x, list):  
            X = np.array(x) 
        elif isinstance(x, torch.Tensor):
            X = (x.cpu() if x.is_cuda else x).numpy()
        else:
            raise NotImplementedError() 
        return self.eval(X)
