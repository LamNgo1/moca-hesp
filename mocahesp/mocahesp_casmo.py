'''Code implementation for MOCAHESP-Casmo'''

import logging
import math
import warnings
from copy import deepcopy

import cma
import numpy as np
from botorch.exceptions import BotorchWarning
from category_encoders import TargetEncoder

from casmopolitan_mocahesp.bo.localbo_utils import onehot2ordinal
from casmopolitan_mocahesp.bo.optimizer import Optimizer
from casmopolitan_mocahesp.bo.optimizer_mixed import MixedOptimizer
from mocahesp.mocahesp_meta import MOCAHESPMeta

from .gp import train_gp
from .utils import from_unit_cube, latin_hypercube, to_unit_cube

# from cmabo._warm_start import get_warm_start_mgd



warnings.simplefilter("ignore", BotorchWarning)


class MOCAHESPCasmo(MOCAHESPMeta):
    def __init__(
            self, 
            obj_func,
            max_evals, 
            func_name='',
            n_init=20,
            **kwargs,
    ):
        super().__init__(obj_func=obj_func, max_evals=max_evals, n_init=n_init, func_name=func_name)
        self.solver = 'casmo'
    
    def str_x(self, x_, decode=True):
        if decode:
            x = self.decode_input(x_.flatten())
        else:
            x = deepcopy(x_.flatten())
        x_out = x[self.cat_idx].astype(int).tolist() + np.around(x[self.cont_idx], 2).tolist()
        return x_out
    
    def init_cma(self, x_init_, fx_init, cma_opts):
        assert x_init_.min() <= 1 and x_init_.max() >= 0

        self.exp3_mean, self.exp3_std = fx_init.mean(), fx_init.std()
        T = np.round(self.max_evals/self.popsize)
        self.exp3_weights = [1, 1]
        self.exp3_count = [0, 0]
        self.exp3_K = len(self.exp3_weights)
        self.exp3_eta = min(1 , np.sqrt((self.exp3_K * np.log(self.exp3_K)) / ((np.e - 1) * T)))
        self.cma_enc = self.choose_cma_enc()

        if self.cma_enc == 0: # ordinal
            encoded_x_init_unit = deepcopy(x_init_)
        elif self.cma_enc == 1: # target
            # transform to target
            cols = (np.arange(self.cat_dim)).tolist()
            X_init_up = from_unit_cube(x_init_, self.bound_ex[:, 0], self.bound_ex[:, 1])    
            X_init_up[:, self.cat_idx] = np.round(X_init_up[:, self.cat_idx])     
            self.enc = TargetEncoder(cols=cols, handle_unknown='error', handle_missing='error', return_df=False).fit(X=X_init_up, y=fx_init)
            encoded_x_init = self.enc.transform(X_init_up, y=fx_init)
            self.encoded_lb = np.zeros(self.dim)
            self.encoded_ub = np.zeros(self.dim)
            self.encoded_lb[self.cat_idx], self.encoded_ub[self.cat_idx] = self.get_bound_enc()
            self.encoded_lb[self.cont_idx] = self.bound_ex[self.cont_idx, 0]
            self.encoded_ub[self.cont_idx] = self.bound_ex[self.cont_idx, 1]
            encoded_x_init_unit = to_unit_cube(encoded_x_init, self.encoded_lb, self.encoded_ub)
        else:
            raise NotImplementedError()
        mean0 = encoded_x_init_unit[fx_init.argmin()]        
        sigma0 = 0.3                
        self.es = cma.CMAEvolutionStrategy(mean0.flatten(), sigma0, cma_opts)
        self.pending = [] # points pending for update
        self.local_fbest = fx_init.min()

        return 0
        
    def update_cma(self, x, fx, cma_opts):
        assert x.min() <= 1 and x.max() >= 0
        self.pending.append((deepcopy(x), deepcopy(fx)))
        if len(self.pending) == self.popsize:
            current_arm = self.cma_enc
            fx_lambda = np.array([ffx for _, ffx in self.pending])
            fx_reward = fx_lambda.min()
            exp3_lb, exp3_ub = (-self.observed_fx).min(), (-self.observed_fx).max()
            reward = (-fx_reward - exp3_lb) / (exp3_ub - exp3_lb)
            prob = self.calculate_exp3_probabilities()[current_arm]
            estimated_reward = reward/prob
            self.exp3_weights[current_arm] *= np.exp(self.exp3_eta* estimated_reward / self.exp3_K)           
            self.cma_enc = self.choose_cma_enc()
            # assert len(x_fevals) == len(fx_fevals) == self.popsize
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
            else:
                raise NotImplementedError()
            
            encoded_x_init, encoded_x_prev = np.vsplit(encoded_x_unit, [self.n_init])
            fx_init, fx_prev = np.vsplit(self.observed_fx, [self.n_init])
            mean0 = encoded_x_init[fx_init.argmin()]
            sigma0 = 0.3
            self.es = cma.CMAEvolutionStrategy(mean0.flatten(), sigma0, cma_opts)
            if len(encoded_x_prev):
                self.es.feed_for_resume(encoded_x_prev, fx_prev.flatten())
            logging.info(f'sigma_vec.scaling={np.around(self.es.sigma_vec.scaling, 2)}')
            if len(fx_prev):
                if self.local_fbest > fx_prev.min():
                    self.local_fbest = fx_prev.min()
            logging.info(f'{self.total_eval}/{self.max_evals}({self.obj_func.name})-' +\
                f' fbest: {np.vstack((self.history_fx, self.observed_fx)).min():.4f}' +\
                f'; fbest_local: {self.local_fbest:.4f}' +\
                f'; sigma/sigma0: {np.around(self.es_sigma/self.es_sigma0, 4)}\n' +\
                f'x_local: {self.str_x(self.observed_x[np.argmin(self.observed_fx), :])}'
            ) 
            ... 
            self.pending = []
        return self.es.stop()

    def adjust_to_discrete_tr(self, x_cat_, mean_cat_up_round, L_H, is_mixed: bool):
        x_cat = deepcopy(x_cat_)
        count = 0
        for i, x in enumerate(x_cat):     
            x_temp = deepcopy(x_cat[i]) 
            d = np.count_nonzero(x_temp != mean_cat_up_round)
            if d > L_H:
                count += 1 
                retry = 0     
                non_identical_idx = np.where(x_temp != mean_cat_up_round)[0]
                n_min_fix, n_max_fix = d - L_H, len(non_identical_idx) - 1
                while retry < 10:  
                    retry += 1
                    x_temp = deepcopy(x_cat[i]) 
                    n_fix = np.random.choice(np.arange(n_min_fix, n_max_fix))
                    keep_axis = np.random.choice(non_identical_idx, n_fix, replace=False)
                    x_temp[keep_axis] = mean_cat_up_round[keep_axis]
                    if not any(np.all(x_temp == x_cat, axis=1)):
                        break
                    if is_mixed:
                        # if this is mixed problem, dont need to care about duplicate
                        break
                    ...
                    # reset x_temp and try to fix again
                assert np.count_nonzero(x_temp != mean_cat_up_round) == d - n_fix <= L_H
                x_cat[i] = x_temp
                ...
        # if count > 0:
        #     logging.info(f'Fixed {count} data points')
        max_hamming_dist = max(np.count_nonzero(x_cat != mean_cat_up_round,axis=1))
        assert L_H >= max_hamming_dist, f'max_hamming_dist exceeds threshold ({max_hamming_dist} > {L_H})'
        return x_cat

    def remove_observed_x(self, x_input):
        x_output = deepcopy(x_input)
        if self.cont_dim == 0:
            observed_x_up = from_unit_cube(np.vstack((self.history_x, self.observed_x)), self.bound_ex[self.cat_idx, 0], self.bound_ex[self.cat_idx, 1]) 
            for xx in observed_x_up:
                mask = ~np.all(x_output == xx, axis=1)
                x_output = x_output[mask]
        return x_output

    def optimize(self):
        self.local_fbest = np.inf
        kwargs = {}
        if self.func_name == 'pest':
            kwargs = {
                'length_max_discrete': 25,
                'length_init_discrete': 20,
            }
        elif self.func_name == 'ackley53m':
            kwargs = {
                'length_max_discrete': 50,
                'length_init_discrete': 30,
            }
        elif self.func_name == 'ackley20c':
            kwargs = {
                'length_max_discrete': 20,
                'length_init_discrete': 20,
            }
        elif self.func_name == 'maxsat60':
            kwargs = {
                'length_max_discrete': 60,
            }
        elif self.func_name == 'maxsat28':
            kwargs = {
                'length_max_discrete': 28,
            }
        elif self.func_name == 'maxsat43':
            kwargs = {
                'length_max_discrete': 43,
            }
        elif self.func_name == 'labs':
            kwargs = {
                'length_max_discrete': 50,
            }
        elif self.func_name == 'antibody':
            kwargs = {
                'length_max_discrete': 11,
                'length_init_discrete': 10,
            }
        elif self.func_name == 'maxsat125':
            kwargs = {
                'length_max_discrete': 125,
            }
        elif self.func_name == 'cco':
            kwargs = {
                'length_max_discrete': 15,
                'length_init_discrete': 15,
            }
        else:
            # Check CASMOPOLITAN paper for more details to set these parameters
            raise NotImplementedError()
        

        def create_candidates(n_cand, length, **kwargs):
            """
            This function generates candidates for MOCA-HESP-Casmo, which satisfies the Mahalanobis distance constraint (as described in MOCA-HESP) and the Hamming distance constraint (as in CASMOPOLITAN).
            """
            length_discrete = kwargs["length_discrete"] if kwargs.get("length_discrete") else None
            length_init_discrete = kwargs["length_init_discrete"] if kwargs.get("length_init_discrete") else None
            vertices = np.array(self.cat_vertices)
            D = len(vertices)
            L_H = length_discrete
            if self.cma_enc == 0:
                mean_cat_up = deepcopy(self.es_mean[self.cat_idx]).reshape(1, -1)
                mean_cat_up = from_unit_cube(mean_cat_up, self.bound_ex[self.cat_idx, 0], self.bound_ex[self.cat_idx, 1])
                mean_cat_up_round = np.round(mean_cat_up.flatten())
            elif self.cma_enc == 1:
                mean_cat_up = deepcopy(self.es_mean).reshape(1, -1)
                mean_cat_up[:, self.cat_idx] = from_unit_cube(mean_cat_up[:, self.cat_idx], self.encoded_lb[self.cat_idx], self.encoded_ub[self.cat_idx])
                mean_cat_up_round = self._decode_to_nearest_value(self.enc, mean_cat_up).flatten()
                mean_cat_up_round = mean_cat_up_round[self.cat_idx]
                ...
            if np.any(mean_cat_up_round > np.max(self.obj_func.bounds_m)) or np.any(mean_cat_up_round < np.min(self.obj_func.bounds_m)):
                logging.info(f'***ERROR: mean_cat_up_round = {mean_cat_up_round}')
                mean_cat_up_round = np.clip(mean_cat_up_round, self.obj_func.bounds_m[self.cat_idx, 0], self.obj_func.bounds_m[self.cat_idx, 1])
                logging.info(f'***FIXED: mean_cat_up_round = {mean_cat_up_round}')
            eival, eivec = np.linalg.eigh(self.es.sigma**2 *self.es_cov)
            new_eigval = np.sqrt(eival)

            # trust region


            if len(self.cont_idx) != 0:
                dims_to_scale = []
                for i in range(len(eival)):
                    eivec_i = eivec[:, i]
                    if any(eivec_i[self.cont_idx] != 0):
                        dims_to_scale.append(i)
                new_eigval[dims_to_scale] *= length
            
            new_eigval = np.square(new_eigval)
            new_cov = eivec @ np.diag(new_eigval) @ np.linalg.inv(eivec)
            
            # This flow is similar to MOCA-HESP-BO
            x_cand = np.random.multivariate_normal(self.es_mean, new_cov, size=int(1.2*n_cand))
            x_cand = x_cand[self._is_in_ellipse_custom(self.es_mean, new_cov, x_cand)][:n_cand,:]
            x_cand_unit = np.array([self.bound_tf.repair(x) for x in x_cand])
            # x_cand_unit = to_unit_cube(x_cand, self.lb, self.ub) 
            if self.cma_enc == 0:
                x_cand_up = deepcopy(x_cand_unit)
                x_cand_up[:, self.cat_idx] = from_unit_cube(x_cand_up[:, self.cat_idx], self.bound_ex[self.cat_idx, 0], self.bound_ex[self.cat_idx, 1])
                x_cand_up[:, self.cat_idx] = np.round(x_cand_up[:, self.cat_idx])
            elif self.cma_enc == 1:
                x_cand_encoded_up = deepcopy(x_cand_unit)
                x_cand_encoded_up[:, self.cat_idx] = from_unit_cube(x_cand_encoded_up[:, self.cat_idx], self.encoded_lb[self.cat_idx], self.encoded_ub[self.cat_idx])
                x_cand_up = self._decode_to_nearest_value(self.enc, x_cand_encoded_up)
            x_cand_up = np.unique(x_cand_up, axis=0)
            
            x_cand_up_backup = deepcopy(x_cand_up)
            # Constrain some dims to the mean, to satisfy the Hamming distance
            if L_H < D:
                x_cand_cat, x_cand_cont = np.hsplit(x_cand_up, [self.cat_dim])
                x_cand_cat_fixed = self.adjust_to_discrete_tr(x_cand_cat, mean_cat_up_round, L_H, is_mixed=self.cont_dim > 0)
                x_cand_up = np.hstack((x_cand_cat_fixed, x_cand_cont))
                x_cand_up = np.unique(x_cand_up, axis=0)

                x_cand_up_backup = deepcopy(x_cand_up)
                if self.cma_enc == 0:
                    temp = to_unit_cube(x_cand_up, self.bound_ex[:, 0], self.bound_ex[:, 1])
                elif self.cma_enc == 1:
                    temp_enc = self.enc.transform(x_cand_up)
                    temp = to_unit_cube(temp_enc, self.encoded_lb, self.encoded_ub)
                mask = self._is_in_ellipse_custom(self.es_mean, new_cov, temp)
                x_cand_up = x_cand_up[mask]
            ...
            if len(x_cand_up) == 0: # in case no candidates are valid
                logging.info("No candidates, use all candidates")
                x_cand_up = x_cand_up_backup

            # sanity check
            max_hamming_dist = max(np.count_nonzero(x_cand_up[:,self.cat_idx] != mean_cat_up_round,axis=1))
            assert L_H >= max_hamming_dist, f'max_hamming_dist exceeds threshold ({max_hamming_dist} > {L_H})'

            # remove already observed data points
            x_cand_up = self.remove_observed_x(x_cand_up)
            if len(x_cand_up) < n_cand:
                logging.info(f'n_unique_cand={len(x_cand_up)}')
            return x_cand_up   

        
        cma_opts = dict()
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

        kwargs["create_candidates"] = create_candidates

        problem_type = 'mixed' if self.obj_func.discrete_dim_m + self.cont_dim > 0 else 'categorical'
        ard = True 
        noise_variance = None
        batch_size = 1
        kernel_type = 'mixed' if problem_type == 'mixed' else 'transformed_overlap'

        config = np.array(self.cat_vertices)
        lb_cont = self.obj_func.bounds_m[self.cont_idx][:, 0]
        ub_cont = self.obj_func.bounds_m[self.cont_idx][:, 1]
        # x_casmo = np.empty((0, self.obj_func.dim_m))
        # fx_casmo = np.empty((0, 1))
        max_iters = self.max_evals - self.total_eval

        if problem_type == 'mixed':
            optim = MixedOptimizer(config, lb_cont, ub_cont, self.cont_idx, self.cat_idx,
                                n_init=self.n_init, use_ard=ard,
                                kernel_type=kernel_type,
                                noise_variance=noise_variance,
                                **kwargs)
        else:
            optim = Optimizer(config, n_init=self.n_init, use_ard=ard,
                            kernel_type=kernel_type,
                            noise_variance=noise_variance, **kwargs)

        for i in range(max_iters):
            x_next_unwrap = optim.suggest(batch_size)
            # self._save_extra_data(self.total_eval + len(x_casmo), {'iter_time': end-start})
            x_next = to_unit_cube(x_next_unwrap, self.bound_ex[:, 0], self.bound_ex[:, 1])
            x_next_compute = self._convert_to_x_eval(x_next)
            y_next = self.obj_func.func(x_next_compute.squeeze()).reshape(batch_size)

            assert x_next.min() <= 1 and x_next.max() >= 0
            self.observed_x = np.vstack((self.observed_x, deepcopy(x_next)))
            self.observed_fx = np.vstack((self.observed_fx, deepcopy(y_next)))

            optim.observe(x_next_unwrap, y_next)
            # x_casmo = np.vstack((x_casmo, x_next))
            # fx_casmo = np.vstack((fx_casmo, y_next))
            Y = np.array(optim.casmopolitan.fX)

            if self.n_init != optim.casmopolitan.n_init:
                self.n_init = optim.casmopolitan.n_init
                logging.info(f'self.n_init is changed to {self.n_init}')

            try:
                logging.info('Iter %d, fX: %.4f, fX_best: %.4f. L: %.2f (max=%.2f); L_H: %d/%d (max=%d); Fail: %d/%d; Suc: %d/%d, X: %s'
                    % ( i, float(y_next), float(optim.casmopolitan._fX.min()) if len(optim.casmopolitan._fX) else np.inf,
                        optim.casmopolitan.length, optim.casmopolitan.length_max,
                        optim.casmopolitan.length_discrete, optim.casmopolitan.length_init_discrete, optim.casmopolitan.length_max_discrete, 
                        optim.casmopolitan.failcount,
                        optim.casmopolitan.failtol,
                        optim.casmopolitan.succcount,
                        optim.casmopolitan.succtol,
                        self.str_x(x_next_unwrap, decode=False),
                    )
                    )
            except Exception as e:
                logging.info(e)
            
            if len(optim.casmopolitan._X) == len(optim.casmopolitan._fX) == optim.casmopolitan.n_init:
                X = to_unit_cube(optim.casmopolitan._X, self.bound_ex[:, 0], self.bound_ex[:, 1])
                self.init_cma(X, optim.casmopolitan._fX, cma_opts)
                logging.info(f'MOCA-HESP-Casmo init at iter = {self.total_eval}')
            elif len(optim.casmopolitan._X) == len(optim.casmopolitan._fX) < optim.casmopolitan.n_init:
                if self.es is not None:
                    self.history_x = np.vstack((self.history_x, deepcopy(self.observed_x)))
                    self.history_fx = np.vstack((self.history_fx, deepcopy(self.observed_fx)))
                    logging.info(f'MOCA-HESP-Casmo restarts at iter = {self.total_eval}')
                    self.observed_x = np.zeros((0, self.dim))
                    self.observed_fx = np.zeros((0, 1)) 
                    self.pending = []
                    self.es = None
                else:
                    # initializing phase, do nothing
                    ...
            else:
                stopdict = self.update_cma(x_next, y_next, cma_opts)
                if stopdict != {}:
                    logging.info(f"HESP is stopped, needs to derive this case:\n{stopdict}")
                    assert False, "HESP is stopped, needs to derive this case"
                ...
            assert len(self.history_x) + len(self.observed_x) == len(self.history_fx) + len(self.observed_fx) == len(optim.casmopolitan.fX) == len(optim.casmopolitan.X)
            ...
        return self.get_optimization_results()

        
