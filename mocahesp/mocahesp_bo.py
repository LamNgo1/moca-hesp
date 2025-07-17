'''Code implementation for MOCAHESP-BO'''

import collections
import logging
import math
import pickle
import time
import warnings
from copy import deepcopy

import cma
import gpytorch
import numpy as np
import scipy
import torch
from botorch.exceptions import BotorchWarning
from category_encoders import TargetEncoder

from mocahesp.mocahesp_meta import MOCAHESPMeta

from .gp import train_gp
from .utils import from_unit_cube, latin_hypercube, to_unit_cube

warnings.simplefilter("ignore", BotorchWarning)



class MOCAHESPBO(MOCAHESPMeta):
    def __init__(
            self, 
            obj_func, 
            max_evals, 
            func_name='',
            n_init=20,
            **kwargs,
    ):
        super().__init__(obj_func=obj_func, max_evals=max_evals, n_init=n_init)
        self.solver = 'bo'

    def optimize(self):
        '''Run optimization'''        

        while self.total_eval < self.max_evals:            
            self.observed_x = np.zeros((0, self.dim))   # global dataset
            self.observed_fx = np.zeros((0, 1))         # global dataset
            
            # Make sure initialization covers all choices, if not target encoder will fail
            max_choice = np.max(self.cat_vertices)
            if max_choice > self.n_init:
                self.n_init = max(self.n_init, max_choice)
                logging.info(f'n_init is updated to {self.n_init}')
            X_init = np.empty((self.n_init, self.cat_dim))
            for i, vertices in enumerate(self.cat_vertices):
                required_vertices = np.arange(vertices)
                n_extra_vertices = self.n_init - len(required_vertices)
                extra_vertices = np.zeros(n_extra_vertices)
                if n_extra_vertices:
                    extra_vertices = np.random.choice(vertices, n_extra_vertices)
                all_vertices = np.append(required_vertices, extra_vertices)
                np.random.shuffle(all_vertices)
                X_init[:, i] = all_vertices
                ...
            X_init[:, self.cat_idx] = to_unit_cube(X_init[:, self.cat_idx], self.bound_ex[self.cat_idx][:, 0], self.bound_ex[self.cat_idx][:, 1])
            assert all([len(np.unique((X_init)[:,i]))==self.cat_vertices[i] for i in range(self.cat_dim)])

            if self.cont_dim > 0 :
                X_init_cat = deepcopy(X_init)
                X_init_cont = latin_hypercube(self.n_init, self.cont_dim)
                X_init_cont = from_unit_cube(X_init_cont, self.lb[self.cont_idx], self.ub[self.cont_idx])
                X_init = np.empty((self.n_init, self.dim))
                X_init[:, self.cat_idx] = X_init_cat
                X_init[:, self.cont_idx] = X_init_cont
                assert X_init.min() <= 1 and X_init.max() >= 0
                
            X_init_for_eval = self._convert_to_x_eval(X_init)
            fX_init = np.array([self.f(x) for x in X_init_for_eval]).reshape(-1, 1)
            self.X_init = deepcopy(X_init)
            self.fX_init = deepcopy(fX_init)
            self.observed_x = np.vstack((self.observed_x, deepcopy(X_init)))
            self.observed_fx = np.vstack((self.observed_fx, deepcopy(fX_init)))
            logging.info(f'{self.total_eval}/{self.max_evals} - fbest: {self.observed_fx.min():.4f}')
            
            # EXP3 parameters and uniformly select cma encoding
            exp3_T = np.round(self.max_evals/self.popsize)
            self.exp3_weights = [1, 1]
            self.exp3_count = [0, 0]
            self.exp3_K = len(self.exp3_weights)
            self.exp3_eta = min(1 , np.sqrt((self.exp3_K * np.log(self.exp3_K)) / ((np.e - 1) * exp3_T)))
            self.cma_enc = self.choose_cma_enc()

            # Initialize CMA EvolutionStrategy based on the selected encoding
            if self.cma_enc == 0:
                mean0 = X_init[fX_init.argmin()]        
                domain_length = self.ub[0] - self.lb[0] 
                sigma0 = 0.3*domain_length
            elif self.cma_enc == 1:
                cols = (np.arange(self.cat_dim)).tolist()
                X_init_up = from_unit_cube(X_init, self.bound_ex[:, 0], self.bound_ex[:, 1])
                X_init_up[:, self.cat_idx] = np.round(X_init_up[:, self.cat_idx])
                self.enc = TargetEncoder(cols=cols, handle_unknown='error', handle_missing='error', return_df=False).fit(X=X_init_up, y=fX_init)
                encoded_x_init = self.enc.transform(X_init_up, y=fX_init)
                self.encoded_lb = np.zeros(self.dim)
                self.encoded_ub = np.zeros(self.dim)
                self.encoded_lb[self.cat_idx], self.encoded_ub[self.cat_idx] = self.get_bound_enc()
                self.encoded_lb[self.cont_idx] = self.bound_ex[self.cont_idx, 0]
                self.encoded_ub[self.cont_idx] = self.bound_ex[self.cont_idx, 1]
                mean0 = to_unit_cube(encoded_x_init, self.encoded_lb, self.encoded_ub)[fX_init.argmin()]
                sigma0 = 0.3
                ...

            # This will assume to starting covariance matrix to be eye matrix
            minstd = np.zeros(self.dim)
            minstd[self.cat_idx] = 0.1
            minstd[self.obj_func.discrete_idx_m] = 0.1
            cma_opts = {
                'popsize': self.popsize, 
                'bounds': [0, 1],
                'seed': np.nan,
                'integer_variables': list(self.cat_idx) + list(self.obj_func.discrete_idx_m),
                'minstd': minstd.tolist(),
            }                         
            self.es = cma.CMAEvolutionStrategy(mean0.flatten(), sigma0, cma_opts)

            # Record values
            local_fbest_hist = np.array([self.observed_fx.min()])
            local_fbest = self.observed_fx.min()
            
            # Main loop
            while not self.es.stop() and self.total_eval < self.max_evals:           
                x_fevals, fx_fevals = self._run_bo_steps()
                
                self.observed_x = np.vstack((self.observed_x, deepcopy(x_fevals)))
                self.observed_fx = np.vstack((self.observed_fx, deepcopy(fx_fevals)))
                assert len(np.unique(self.observed_x, axis=0)) == len(self.observed_x)
                assert len(self.history_fx) + len(self.observed_fx) == self.total_eval

                # use exp3 to select the next cma encoding
                current_arm = self.cma_enc
                exp3_lb, exp3_ub = (-self.observed_fx).min(), (-self.observed_fx).max()
                reward = (-fx_fevals.min() - exp3_lb) / (exp3_ub - exp3_lb)
                prob = self.calculate_exp3_probabilities()[current_arm]
                estimated_reward = reward/prob
                self.exp3_weights[current_arm] *= np.exp(self.exp3_eta* estimated_reward / self.exp3_K)           
                self.cma_enc = self.choose_cma_enc()
                

                # Update cma mean, covariance matrix and step-size based on selected encoding
                assert len(x_fevals) == len(fx_fevals) == self.popsize
                if self.cma_enc == 0:
                    encoded_x_unit = deepcopy(self.observed_x)
                elif self.cma_enc == 1:
                    cols = (np.arange(self.cat_dim)).tolist()
                    observed_x_up = from_unit_cube(self.observed_x, self.bound_ex[:, 0], self.bound_ex[:, 1]) 
                    observed_x_up[:, self.cat_idx] = np.round(observed_x_up[:, self.cat_idx])
                    self.enc = TargetEncoder(cols=cols, handle_unknown='error', handle_missing='error', return_df=False).fit(X=observed_x_up, y=self.observed_fx)
                    encoded_x = self.enc.transform(observed_x_up, y=self.observed_fx)
                    self.encoded_lb = np.zeros(self.dim)
                    self.encoded_ub = np.zeros(self.dim)
                    self.encoded_lb[self.cat_idx], self.encoded_ub[self.cat_idx] = self.get_bound_enc()
                    self.encoded_lb[self.cont_idx] = self.bound_ex[self.cont_idx, 0]
                    self.encoded_ub[self.cont_idx] = self.bound_ex[self.cont_idx, 1]
                    encoded_x_unit = to_unit_cube(encoded_x, self.encoded_lb, self.encoded_ub)
                    ...
                else:
                    raise NotImplementedError()
            
                encoded_x_init, encoded_x_prev, encoded_x_curr = np.vsplit(encoded_x_unit, [self.n_init, len(encoded_x_unit)-self.popsize])
                fx_init, fx_prev, fx_curr = np.vsplit(self.observed_fx, [self.n_init, len(encoded_x_unit)-self.popsize])
                mean0 = encoded_x_init[fx_init.argmin()]
                sigma0 = 0.3
                self.es = cma.CMAEvolutionStrategy(mean0.flatten(), sigma0, cma_opts)
                if len(encoded_x_prev):
                    self.es.feed_for_resume(encoded_x_prev, fx_prev.flatten())

                self.es.ask()
                self.es.tell(encoded_x_curr, fx_curr.flatten())
                logging.info(f'sigma_vec.scaling={np.around(self.es.sigma_vec.scaling, 2)}')
                
                # Update fbest
                if len(fx_fevals):
                    if local_fbest > fx_fevals.min():
                        local_fbest = fx_fevals.min()
                local_fbest_hist = np.append(local_fbest_hist, local_fbest)

                logging.info(f'\n{self.total_eval}/{self.max_evals}({self.obj_func.name})-' +\
                    f' fbest: {np.vstack((self.history_fx, self.observed_fx)).min():.4f}' +\
                    f'; fbest (this restart): {local_fbest:.4f}' +\
                    f'; sigma/sigma0: {np.around(self.es_sigma/self.es_sigma0, 4)}\n' +\
                    f'xbest (this restart): {self.str_x(self.observed_x[np.argmin(self.observed_fx), :])}\n'
                )    

            if self.es.stop():
                logging.info('Stop: ', self.es.stop()) # Print the reason why cmaes package stops
                self.history_x = np.vstack((self.history_x, deepcopy(self.observed_x)))
                self.history_fx = np.vstack((self.history_fx, deepcopy(self.observed_fx)))
        
        return self.get_optimization_results()

    
    def str_x(self, x_):
        """
        Convert x_ to a string representation (for print only)
        """
        x = self.decode_input(x_.flatten())
        x_out = x[self.cat_idx].astype(int).tolist() + np.around(x[self.cont_idx], 2).tolist()
        return x_out
    

    def _run_bo_steps(self):
        '''Run BO steps with current selected encoder'''
        x_bo = np.zeros((0, self.dim))
        fx_bo = np.zeros((0, 1))
           
        for cnt in range(self.popsize):
            x_gp = np.vstack((deepcopy(self.observed_x), x_bo))
            fx_gp = np.vstack((deepcopy(self.observed_fx), fx_bo))
            x_gp_unit = deepcopy(x_gp) # already in unit cube

            # Standardize fx for training GP            
            assert x_gp_unit.min() >= 0 and x_gp_unit.max() <= 1
            mu, std = np.median(fx_gp), fx_gp.std()
            std = 1.0 if std < 1e-6 else std
            fx_gp_unit = (deepcopy(fx_gp) - mu) / std
            with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                gp = train_gp(torch.tensor(x_gp_unit), torch.tensor(fx_gp_unit.flatten()), True, 50)

            # After training GP, let's start to sample TS points
            n_cand = min(100*self.dim, 5000) # number of TS sampling points
            x_cand = np.random.multivariate_normal(self.es_mean, self.es_sigma**2*self.es_cov, size=int(n_cand*1.2))
            mask = self._is_in_ellipse(x_cand)
            x_cand = x_cand[mask][:n_cand, :] # in continuous encoded space Z_e
            # The following step adjust x_cand points sampled by mvn to points that are decoded from a categorical value
            if self.cma_enc == 0:
                bound_tf = cma.BoundTransform([0, 1])
                x_cand = np.array([bound_tf.repair(x) for x in x_cand]) 
                x_cand_encoded = from_unit_cube(x_cand, self.bound_ex[:, 0], self.bound_ex[:, 1]) # scaled to the extended bounds
                x_cand_up_round = deepcopy(x_cand_encoded)
                x_cand_up_round[:, self.cat_idx] = np.round(x_cand_up_round[:, self.cat_idx]) # decode to nearest integer values, i.e., rounding in the case of ordinal encoding
            elif self.cma_enc == 1:
                bound_tf = cma.BoundTransform([0, 1])
                x_cand = np.array([bound_tf.repair(x) for x in x_cand])
                x_cand_abs = np.abs(x_cand)
                x_cand_encoded = from_unit_cube(x_cand_abs, self.encoded_lb, self.encoded_ub)  # scaled to the extended bounds
                x_cand_up_round = self._decode_to_nearest_value(self.enc, x_cand_encoded) # decode to nearest integer values
                if self.cont_dim > 0:
                    assert np.allclose(x_cand_encoded[:, self.cont_idx], x_cand_up_round[:, self.cont_idx]) 
            x_cand_round_unit = to_unit_cube(x_cand_up_round, self.bound_ex[:, 0], self.bound_ex[:, 1]) # scale back to unit cube to predict with GP
            
            if 1: # in case of no valid candidates, use the original x_cand
                x_back_up = deepcopy(x_cand)
                x_back_up_round_unit = deepcopy(x_cand_round_unit)
                x_back_up_up_round = deepcopy(x_cand_up_round)
                mask2 = self._is_in_ellipse(x_cand_round_unit)
                x_cand_up_round = x_cand_up_round[mask2]
                x_cand_round_unit = x_cand_round_unit[mask2]
                x_cand = x_cand[mask2]
                if len(x_cand_round_unit) == 0:
                    logging.info(f'No valid candidate found, revert to the original one')
                    x_cand = deepcopy(x_back_up)
                    x_cand_round_unit = deepcopy(x_back_up_round_unit)
                    x_cand_up_round = deepcopy(x_back_up_up_round)

            # Remove candidates that are already in the observed data
            for xx in np.vstack((self.observed_x, x_bo)):
                mask = ~np.all(x_cand_round_unit == xx, axis=1)
                x_cand_up_round = x_cand_up_round[mask]
                x_cand_round_unit = x_cand_round_unit[mask]
                x_cand = x_cand[mask]
            assert len(x_cand_round_unit) == len(x_cand) == len(x_cand_up_round)
 
            # GP prediction
            with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                x_torch = torch.tensor(x_cand_round_unit).to(dtype=torch.float32)
                y_cand = gp.likelihood(gp(x_torch)).sample(torch.Size([1])).t().numpy()

            # Select the best candidate and update the BO history
            x_min = x_cand_round_unit[y_cand.argmin(axis=0), :]
            x_min_for_eval = self._convert_to_x_eval(x_min)
            fx_min = np.array([self.f(x) for x in x_min_for_eval]).reshape(-1, 1)
            x_bo = np.vstack((x_bo, x_min))
            fx_bo = np.vstack((fx_bo, fx_min))
            del gp, x_torch
            logging.info(f'Iter {self.total_eval + cnt}:' +\
                f' fx: {fx_min.squeeze():.4f}' +\
                f' fbest: {np.vstack((self.history_fx, self.observed_fx, fx_bo)).min():.4f}' +\
                f' x: {self.str_x(x_min.squeeze())}'
            )   
            ...

        return x_bo, fx_bo
