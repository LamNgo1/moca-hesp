'''Code implementation for MOCAHESP-Bounce'''

import collections
import logging
import math
import os
import pickle
import sys
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

from bounce_mocahesp.bounce.benchmarks import *
from bounce_mocahesp.bounce.bounce import Bounce
from bounce_mocahesp.bounce.util.data_handling import from_1_around_origin
from bounce_mocahesp.bounce.util.data_handling import \
    from_unit_cube as from_unit_cube_bounce
from bounce_mocahesp.bounce.util.data_handling import to_1_around_origin
from bounce_mocahesp.bounce.util.data_handling import \
    to_unit_cube as to_unit_cube_bounce
from mocahesp.mocahesp_meta import MOCAHESPMeta

from .gp import train_gp
from .utils import from_unit_cube, latin_hypercube, to_unit_cube

warnings.simplefilter("ignore", BotorchWarning)


class MOCAHESPBounce(MOCAHESPMeta):
    def __init__(
            self, 
            obj_func, 
            max_evals, 
            func_name='',
            n_init=20,
            **kwargs,
    ):
        super().__init__(obj_func=obj_func, max_evals=max_evals, n_init=n_init, func_name=func_name)
        self.solver = 'bounce'
        self.bound_expansion = 0.5 - 1e-6 # to prevent rounding error

    def str_x(self, x_, decode=True):
        if decode:
            x = self.decode_input(x_.flatten())
        else:
            x = deepcopy(x_.flatten())
        x_out = x[self.cat_idx].astype(int).tolist() + np.around(x[self.cont_idx], 2).tolist()
        return x_out
    
    @property
    def tbound_ex(self):
        """
        Return torch tensor of bound_ex
        """
        return torch.tensor(self.bound_ex, dtype=torch.float64)

    def _generate_required_init_points(self, X_available):
        # Find initial mean by initial sampling
        assert X_available.ndim == 2
        assert X_available[:, self.cat_idx].shape[1] == len(self.cat_vertices)
        X_available_up = from_unit_cube(X_available, self.bound_ex[:, 0], self.bound_ex[:, 1])
        X_available_up = np.round(X_available_up).astype(int)
        required_vertices_list = []
        for i, n_vertex in enumerate(self.cat_vertices):
            X_available_i = X_available_up[:, i]
            required_vertex_i = np.arange(n_vertex)
            for v in X_available_i:
                required_vertex_i = required_vertex_i[~(required_vertex_i==v)]
                ...
            required_vertices_list.append(required_vertex_i.tolist())
        ...
        n_required_points = np.max([len(rvl) for rvl in required_vertices_list])
        X_required = np.zeros((n_required_points, self.cat_dim), dtype=int)
        X_required_cont = np.zeros((n_required_points, self.cont_dim))
        # fX_required = np.zeros((0, 1))
        logging.info(f'n_required_points = {n_required_points}')
        if n_required_points > 0:
            for i in self.cat_idx:
                required_vertices = required_vertices_list[i]
                n_extra = n_required_points - len(required_vertices)
                extra_vertices = []
                if n_extra > 0:
                    extra_vertices = np.random.choice(self.cat_vertices[i], n_extra).tolist()
                all_vertices = np.array(required_vertices + extra_vertices)
                np.random.shuffle(all_vertices)
                X_required[:, i] = all_vertices
            X_required = to_unit_cube(X_required.astype(float), self.bound_ex[self.cat_idx, 0], self.bound_ex[self.cat_idx, 1])

            if self.cont_dim > 0:
                X_required_cont = latin_hypercube(n_required_points, self.cont_dim)
                X_required_cont = from_unit_cube(X_required_cont, self.lb[self.cont_idx], self.ub[self.cont_idx])
        X_required = np.hstack((X_required, X_required_cont))
        # if len(X_required) > 0:
        #     assert X_required.min() <= 1 and X_required.max() >= 0
        #     X_required_compute = self._convert_to_x_eval(X_required)
        #     fX_required = np.array([self.f(x) for x in X_required_compute]).reshape(-1, 1)
        assert all([len(np.unique(np.vstack((X_available, X_required))[:,i]))==self.cat_vertices[i] for i in range(self.cat_dim)])
        return X_required
    
    def _project_encoder_to_target_space(self, random_embedding, P):
        A = random_embedding._A.float()
        b = 0#random_embedding._b.float()
        # c = random_embedding._c.float()

        n_required_points_Y = 0
        for bin in random_embedding.bins:
            if bin.parameter_type == ParameterType.CATEGORICAL:
                required = bin.dims_required
            elif bin.parameter_type == ParameterType.BINARY:
                required = 2
            else:
                required = 0
            n_required_points_Y = max(n_required_points_Y, required)

        Y_required = -1.0 * torch.ones((n_required_points_Y, len(random_embedding.bins)))
        for i, bin in enumerate(random_embedding.bins):
            if bin.parameter_type == ParameterType.CATEGORICAL:
                required_vertices = np.arange(bin.dims_required).tolist()
                n_extra = n_required_points_Y - len(required_vertices)
                extra_vertices = []
                if n_extra:
                    extra_vertices = np.random.choice(bin.dims_required, n_extra).tolist()
                all_vertices = np.array(required_vertices + extra_vertices)
                np.random.shuffle(all_vertices)
                Y_required[:, i] = torch.tensor(all_vertices)
            elif bin.parameter_type == ParameterType.BINARY:
                required_vertices = [-1, 1]
                n_extra = n_required_points_Y - len(required_vertices)
                extra_vertices = []
                if n_extra:
                    extra_vertices = np.random.choice([-1, 1], n_extra).tolist()
                all_vertices = np.array(required_vertices + extra_vertices)
                np.random.shuffle(all_vertices)
                Y_required[:, i] = torch.tensor(all_vertices)

        X_required = (A @ Y_required.t() + b)
        # apply_shift = torch.where(c > 1)[0]
        # X_required[apply_shift, :] %= c[apply_shift]
        X_required = X_required.t()
        X_required[X_required == -1] = 0
        Y_required[Y_required == -1] = 0

        X_required_encoded = self.enc.transform(X_required.numpy())
        Y_required_encoded = (P @ X_required_encoded.T).T
        enc_Y_mapping = dict()
        encoded_Y_lb = torch.zeros(len(random_embedding.bins)) # default range for continuous
        encoded_Y_ub = torch.ones(len(random_embedding.bins))
        for i, bin in enumerate(random_embedding.bins):
            if bin.parameter_type != ParameterType.CONTINUOUS:
                y, idx = np.unique(Y_required[:,i].numpy(), return_index=True)
                y_encoded = Y_required_encoded[:,i].numpy()[idx]
                if len(np.unique(y_encoded)) < len(y_encoded):
                    y_encoded += (np.random.rand(len(y_encoded)) * 1e-3)
                enc_Y_mapping.update({i: dict(zip(y.tolist(), y_encoded.tolist()))})

                y_encoded_sort = np.sort(y_encoded)
                encoded_Y_lb[i] = y_encoded_sort[0] - abs(y_encoded_sort[1] - y_encoded_sort[0]) / 2.0
                encoded_Y_ub[i] = y_encoded_sort[-1] + abs(y_encoded_sort[-1] - y_encoded_sort[-2]) / 2.0
            else:
                enc_Y_mapping.update({i: None})
    
        return enc_Y_mapping, encoded_Y_lb, encoded_Y_ub      
    
    def _decode_to_nearest_value_target_space(self, enc_mapping: dict, Y_encoded_):
        def find_nearest_idx(array, value):
            idx = (np.abs(array.reshape(1, -1) - value.reshape(-1, 1))).argmin(axis=1)
            return idx
        
        if Y_encoded_.ndim == 1:
            Y_encoded = Y_encoded_.reshape(1, -1)
        else:
            Y_encoded = deepcopy(Y_encoded_)
        
        Y_decoded = np.zeros(Y_encoded.shape, dtype='float64')
        for i, enc_mapping_i in enc_mapping.items():
            if enc_mapping_i is not None:
                nearest_idx = find_nearest_idx(np.array(list(enc_mapping_i.values())), Y_encoded[:, i])
                nearest_pd_idx = np.array(list(enc_mapping_i.keys()))[None, :][:, nearest_idx]
                Y_decoded[:, i] = nearest_pd_idx
            else:
                Y_decoded[:, i] = Y_encoded_[:, i]
            ...
        
        return Y_decoded
    
    def _encode_target_space(self, enc_mapping: dict, Y_decoded_):
        if Y_decoded_.ndim == 1:
            Y_decoded = Y_decoded_.reshape(1, -1)
        else:
            Y_decoded = deepcopy(Y_decoded_)

        Y_encoded = np.zeros(Y_decoded.shape, dtype='float64')
        for i, enc_mapping_i in enc_mapping.items():
            if enc_mapping_i is not None:
                Y_encoded[:, i] = np.vectorize(enc_mapping_i.get)(Y_decoded[:,i])
            else:
                Y_encoded[:, i] = Y_decoded_[:, i]
            ...
        
        return Y_encoded

    def _remove_random_sign_categorical(self, x, random_embedding):
        """Remove random sign assigned to categorical variables (see Bounce for details)"""
        b = random_embedding._b.int().numpy()
        c = random_embedding._c.int().numpy()
        x_ = from_unit_cube(x , self.bound_ex[:, 0],self.bound_ex[:, 1]).T
        x_ = np.round(x_).astype(int)
        x_ = x_ - b
        apply_shift = np.where(c > 1)[0]
        x_[apply_shift, :] %= c[apply_shift]
        x_ = to_unit_cube(x_.T.astype(float) , self.bound_ex[:, 0],self.bound_ex[:, 1])
        x_[:, self.cont_idx] = x[:, self.cont_idx]
        return x_
    
    def _add_random_sign_categorical(self, x, random_embedding):
        """Add back the random sign assigned to categorical variables (see Bounce for details)"""
        b = random_embedding._b.int().numpy()
        c = random_embedding._c.int().numpy()
        x_ = from_unit_cube(x , self.bound_ex[:, 0],self.bound_ex[:, 1]).T
        x_ = np.round(x_).astype(int)
        x_ = x_ + b
        apply_shift = np.where(c > 1)[0]
        x_[apply_shift, :] %= c[apply_shift]
        x_ = to_unit_cube(x_.T.astype(float) , self.bound_ex[:, 0],self.bound_ex[:, 1])
        x_[:, self.cont_idx] = x[:, self.cont_idx]
        return x_

    def _convert_to_cma_bounds(self, X: np.ndarray):
        X = to_unit_cube( X, self.bound_ex[:, 0],self.bound_ex[:, 1])
        X = from_unit_cube(X, np.array(self.obj_func.bounds)[:, 0], np.array(self.obj_func.bounds)[:, 1])
        return X

    def _transform_one_hot(self, problem, X: np.ndarray):
        _transform_one_hot = getattr(problem, "_transform_one_hot", None)
        if callable(_transform_one_hot):
            X_transformed = np.array([problem._transform_one_hot(torch.tensor(x)) for x in X], dtype=float)
        else:
            X_transformed = X.astype(float)
        return X_transformed

    def optimize(self):
        self.local_fbest = np.inf
        self.es = None # reset the cma region
        self.x_pending = None
        self.fx_pending = None

        self.cma_opts = dict()
        minstd = np.zeros(self.dim)
        minstd[self.cat_idx] = 0.1
        minstd[self.obj_func.discrete_idx_m] = 0.1
        self.cma_opts = {
            'popsize': self.popsize, 
            'bounds': [0, 1],
            'seed': np.nan,
            'integer_variables': list(self.cat_idx) + list(self.obj_func.discrete_idx_m),
            'minstd': minstd.tolist(),
        }

        def cma_init(_X_init: np.ndarray, fX_init: np.ndarray, **kwargs):
            random_embedding = deepcopy(kwargs.get('random_embedding', None))
            trust_region = deepcopy(kwargs.get('trust_region', None))
            X_init_compute = self._transform_one_hot(problem, _X_init)
            dim = X_init_compute.shape[1]
            assert self.popsize == 4 + math.floor(3*np.log(dim))       
            X_init = self._convert_to_cma_bounds(X_init_compute)
            X_init_remove_shift = self._remove_random_sign_categorical(X_init, random_embedding)
            
            X_required_remove_shift = self._generate_required_init_points(X_init_remove_shift)
            X_required = self._add_random_sign_categorical(X_required_remove_shift, random_embedding)
            fX_required = np.zeros((X_required.shape[0], 1))
            if len(X_required) > 0:
                assert X_required.min() <= 1 and X_required.max() >= 0
                X_required_compute = self._convert_to_x_eval(X_required)
                fX_required = np.array([self.f(x) for x in X_required_compute]).reshape(-1, 1)
            X_init_remove_shift = np.vstack((deepcopy(X_required_remove_shift), deepcopy(X_init_remove_shift)))
            X_init = np.vstack((deepcopy(X_required), deepcopy(X_init)))
            fX_init = np.vstack((deepcopy(fX_required), deepcopy(fX_init)))
            self.n_init = len(fX_init)
            logging.info(f'self.n_init is changed to {self.n_init}')
            assert all([len(np.unique((X_init_remove_shift)[:,i]))==self.cat_vertices[i] for i in range(self.cat_dim)])

            self.history_x = np.vstack((self.history_x, self.observed_x))
            self.history_fx = np.vstack((self.history_fx, self.observed_fx))

            self.observed_x = deepcopy(X_init_remove_shift)
            self.observed_fx = deepcopy(fX_init)

            self.exp3_mean, self.exp3_std = fX_init.mean(), fX_init.std()
            T = np.round(self.max_evals/self.popsize)
            self.exp3_weights = [1, 1]
            self.exp3_count = [0, 0]
            self.exp3_K = len(self.exp3_weights)
            self.exp3_eta = min(1 , np.sqrt((self.exp3_K * np.log(self.exp3_K)) / ((np.e - 1) * T)))
            self.cma_enc = self.choose_cma_enc()

            if self.cma_enc == 0: # ordinal
                encoded_x_init_unit = deepcopy(X_init_remove_shift)
            elif self.cma_enc == 1: # target
                # transform to target
                cols = (np.arange(self.cat_dim)).tolist()
                X_init_up = from_unit_cube(X_init_remove_shift, self.bound_ex[:, 0], self.bound_ex[:, 1])
                X_init_up[:, self.cat_idx] = np.round(X_init_up[:, self.cat_idx])     
                self.enc = TargetEncoder(cols=cols, handle_unknown='error', handle_missing='error', return_df=False).fit(X=X_init_up, y=fX_init)
                encoded_x_init = self.enc.transform(X_init_up, y=fX_init)
                self.encoded_lb = np.zeros(self.dim)
                self.encoded_ub = np.zeros(self.dim)
                self.encoded_lb[self.cat_idx], self.encoded_ub[self.cat_idx] = self.get_bound_enc()
                self.encoded_lb[self.cont_idx] = self.bound_ex[self.cont_idx, 0]
                self.encoded_ub[self.cont_idx] = self.bound_ex[self.cont_idx, 1]
                encoded_x_init_unit = to_unit_cube(encoded_x_init, self.encoded_lb, self.encoded_ub)
            else:
                raise NotImplementedError()
            mean0 = encoded_x_init_unit[fX_init.argmin()]
            sigma0 = 0.3
            logging.info(f'{self.total_eval}/{self.max_evals} - fbest: {self.observed_fx.min():.4f}')

            mean0_ = self.bound_tf.repair(mean0)      
            self.es = cma.CMAEvolutionStrategy(mean0_.flatten(), sigma0, self.cma_opts)
            self.x_pending = None
            self.fx_pending = None
            self.random_embedding = random_embedding
            self.n_prev_dim = 0
            return self.es

        self.random_embedding = None
        self.n_prev_dim = 0

        def cma_update(_x_fevals: np.ndarray, fx_fevals: np.ndarray, **kwargs):
            random_embedding = deepcopy(kwargs.get('random_embedding', None))
            trust_region = deepcopy(kwargs.get('trust_region', None))

            target_dim_increase = False
            
            if self.random_embedding is None:
                self.random_embedding = random_embedding
            if len(self.random_embedding.bins) != len(random_embedding.bins):
                target_dim_increase = True
                self.n_prev_dim = len(self.observed_fx) - self.n_init
                self.x_pending = None
                self.fx_pending = None
                self.random_embedding = deepcopy(random_embedding)
            x_fevals = self._transform_one_hot(problem, _x_fevals)
            x_fevals = self._convert_to_cma_bounds(x_fevals)
            if self.x_pending is None:
                self.x_pending = x_fevals
                self.fx_pending = fx_fevals
            else:
                self.x_pending = np.vstack((self.x_pending, x_fevals))
                self.fx_pending = np.vstack((self.fx_pending, fx_fevals))
            self.observed_x = np.vstack((self.observed_x, self._remove_random_sign_categorical(x_fevals, random_embedding)))
            self.observed_fx = np.vstack((self.observed_fx, deepcopy(fx_fevals)))
            logging.debug(f'n_unique={len(np.unique(self.observed_x, axis=0))}')
            logging.info(f'fx={fx_fevals.squeeze():.2f}; x={np.around(self.decode_input(x_fevals.squeeze()), 2)}')

            popsize = self.es.popsize
            if target_dim_increase:                   
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
                encoded_x_init, encoded_x_prev = np.vsplit(encoded_x_unit, [self.n_init + self.n_prev_dim])
                fx_init, fx_prev = np.vsplit(self.observed_fx, [self.n_init + self.n_prev_dim])
                
                mean0 = encoded_x_init[fx_init.argmin()]
                sigma0 = 0.3
                mean0_ = self.bound_tf.repair(mean0)   
                self.es = cma.CMAEvolutionStrategy(mean0_.flatten(), sigma0, self.cma_opts)
            elif len(self.fx_pending) == popsize:
                current_arm = self.cma_enc
                fx_lambda = np.array([ffx for ffx in self.fx_pending])
                fx_reward = fx_lambda.min()
                exp3_lb, exp3_ub = (-self.observed_fx).min(), (-self.observed_fx).max()
                reward = (-fx_reward - exp3_lb) / (exp3_ub - exp3_lb)
                prob = self.calculate_exp3_probabilities()[current_arm]
                estimated_reward = reward/prob
                self.exp3_weights[current_arm] *= np.exp(self.exp3_eta* estimated_reward / self.exp3_K)           
                self.cma_enc = self.choose_cma_enc()

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
                encoded_x_init, encoded_x_prev = np.vsplit(encoded_x_unit, [self.n_init + self.n_prev_dim])
                fx_init, fx_prev = np.vsplit(self.observed_fx, [self.n_init + self.n_prev_dim])
                
                mean0 = encoded_x_init[fx_init.argmin()]
                sigma0 = 0.3
                mean0_ = self.bound_tf.repair(mean0)   
                self.es = cma.CMAEvolutionStrategy(mean0_.flatten(), sigma0, self.cma_opts)
                if len(encoded_x_prev):
                    self.es.feed_for_resume(encoded_x_prev, fx_prev.flatten())
                # logging.debug(f'DEBUG: sigma_vec.scaling = {np.around(self.es.sigma_vec.scaling, 2)}')
                if len(fx_prev):
                    if self.local_fbest > fx_prev.min():
                        self.local_fbest = fx_prev.min()
                     
                logging.info(f'{self.total_eval}/{self.max_evals}({self.func_name})-' +\
                    f' fbest: {np.vstack((self.history_fx, self.observed_fx)).min():.4f}' +\
                    f'; fbest_local: {self.observed_fx.min():.4f}' +\
                    f'; sigma/sigma0: {self.es.sigma/(self.es.sigma0):.4f}' +\
                    f'; sigma_vec.scaling: {np.around(self.es.sigma_vec.scaling, 2)}' +\
                    f'; x_local: {np.around(self.decode_input(self._add_random_sign_categorical(self.observed_x[np.argmin(self.observed_fx), :][None, :], random_embedding).flatten()), 2)}'
                )  
                if self.cma_enc == 0:
                    updated_mean = self._add_random_sign_categorical(self.es_mean[None, :], random_embedding)
                    updated_mean = from_unit_cube(updated_mean, self.bound_ex[:, 0], self.bound_ex[:, 1]).flatten()
                    updated_mean[self.cat_idx] = np.round(updated_mean[self.cat_idx])
                elif self.cma_enc == 1:
                    updated_mean = self._add_random_sign_categorical(self.es_mean[None, :], random_embedding)
                    updated_mean = from_unit_cube(self.es_mean[None, :], self.encoded_lb, self.encoded_ub)
                    updated_mean = self._decode_to_nearest_value(self.enc, updated_mean).flatten()
                else:
                    updated_mean = 0
                self.x_pending = None
                self.fx_pending = None
            return self.es

        def create_candidates_discrete(n_cand: int, **kwargs):
            """
            Create candidates, for purely discrete case (only ordinal and categorical parameters)
            """
            trust_region = kwargs['trust_region']
            random_embedding = kwargs['random_embedding']

            A = random_embedding._A.double()
            # b = 0 random_embedding._b.double()
            # c = random_embedding._c.double()
            ATA = torch.mm(A.t(), A) 
            P = torch.mm(torch.linalg.inv(ATA), A.t())

            tr_length_continuous = trust_region.length_continuous
            tr_length_discrete_continuous = trust_region.length_discrete_continuous
            tr_length_init_discrete = trust_region.length_init_discrete
            tr_length_max_discrete = trust_region.length_max_discrete

            non_cont_ids = [idx for idx, bin in enumerate(random_embedding.bins) if bin.parameter_type != ParameterType.CONTINUOUS]
            cont_ids = [idx for idx, bin in enumerate(random_embedding.bins) if bin.parameter_type == ParameterType.CONTINUOUS]
            bin_ids = [idx for idx, bin in enumerate(random_embedding.bins) if bin.parameter_type == ParameterType.BINARY]
            cat_ids = [idx for idx, bin in enumerate(random_embedding.bins) if bin.parameter_type == ParameterType.CATEGORICAL]

            length_continuous = tr_length_continuous
            target_dim = len(random_embedding.bins)
            length_discrete = max(1, round(target_dim * tr_length_discrete_continuous/tr_length_max_discrete))
            
            if len(bin_ids) == 0 and len(cat_ids) != 0:
                length_discrete = max(1, round(tr_length_discrete_continuous/2))
            elif len(bin_ids) != 0 and len(cat_ids) == 0:
                length_discrete = max(1, round(tr_length_discrete_continuous))
            else:
                raise NotImplementedError()
            # logging.info(f'cma-length_discrete={length_discrete}')

            temp_mean_X = self.bound_tf.repair(self.es_mean)
            mean_X = torch.tensor(temp_mean_X, dtype=torch.float64)
            cov_X = torch.tensor(self.es_sigma**2 * self.es_cov, dtype=torch.float64)

            # scale the mvn from [self.lb, self.ub]^D to [-1, 1]^D
            self_lb_torch = torch.tensor(self.lb, dtype=torch.float64)
            self_ub_torch = torch.tensor(self.ub, dtype=torch.float64)
            scale_mtrx = torch.diag((2/(self_ub_torch - self_lb_torch)))
            mean_X_scaled = to_1_around_origin(mean_X.reshape(1, -1), self_lb_torch, self_ub_torch).squeeze()

            cov_X_scaled = scale_mtrx @ cov_X @ scale_mtrx.T
            eival, eivec = torch.linalg.eigh(cov_X_scaled)

            new_eigval = torch.sqrt(eival)
            # trust region
            if self.cat_dim:
                new_eigval[self.cat_idx] *= 1
            if self.obj_func.discrete_dim_m:
                new_eigval[self.obj_func.discrete_idx_m] *= length_continuous
            if self.cont_dim:
                new_eigval[self.cont_idx] *= length_continuous
            new_eigval = torch.square(new_eigval) 

            cov_X_scaled_tr = eivec @ torch.diag(new_eigval) @ torch.linalg.inv(eivec)

            # Remove random shift for categorical
            ub_X = torch.zeros(len(random_embedding.parameters))
            lb_X = torch.zeros(len(random_embedding.parameters))
            for i, param in enumerate(random_embedding.parameters):
                if param.type == ParameterType.CATEGORICAL:
                    ub_X[i] = param.dims_required - 1
                    lb_X[i] = 0
                else:
                    ub_X[i] = param.upper_bound
                    lb_X[i] = param.lower_bound
            assert np.allclose(lb_X, self.obj_func.bounds_m[:,0]) and np.allclose(ub_X, self.obj_func.bounds_m[:,1])
            
            delta_m = mean_X_scaled.detach().clone()

            # project mvn to Y: [-1, 1]^d from [-1, 1]^D
            mean_Y = torch.matmul(P, delta_m.reshape(-1, 1)).t()
            cov_Y = P @ cov_X_scaled_tr @ P.T
                            
            # sample in [-1, 1]^d
            mean_Y_np = mean_Y.numpy().squeeze()
            cov_Y_np = cov_Y.numpy()
            target_dim_cat = len(non_cont_ids)

            X_cand = np.random.multivariate_normal(mean_Y_np, cov_Y_np, size=int(1.2*n_cand))
            X_cand = X_cand[self._is_in_ellipse_custom(mean_Y_np, cov_Y_np, X_cand)][:n_cand,:]
            bound_tf = cma.BoundTransform([-1., 1.])
            X_cand_1around = np.array([bound_tf.repair(x) for x in X_cand])

            # scale to Y: [0, ...]^d
            if self.cma_enc == 0:
                ub_Y = torch.zeros(len(random_embedding.bins))
                lb_Y = torch.zeros(len(random_embedding.bins))
                for i, bin in enumerate(random_embedding.bins):
                    if bin.parameter_type == ParameterType.CATEGORICAL:
                        ub_Y[i] = bin.dims_required - 1 + self.bound_expansion
                        lb_Y[i] = 0 - self.bound_expansion
                    elif bin.parameter_type == ParameterType.BINARY:
                        ub_Y[i] = 1. + self.bound_expansion
                        lb_Y[i] = 0. - self.bound_expansion
                    else:
                        ub_Y[i] = 1
                        lb_Y[i] = 0
                X_cand_up = from_1_around_origin(torch.tensor(X_cand_1around), lb_Y, ub_Y) 
                X_cand_up[:, non_cont_ids] = torch.round(X_cand_up[:, non_cont_ids])
            elif self.cma_enc == 1:
                enc_Y, encoded_lb_Y, encoded_ub_Y = self._project_encoder_to_target_space(random_embedding, P)
                X_cand_encoded = from_1_around_origin(torch.tensor(X_cand_1around), encoded_lb_Y, encoded_ub_Y) 
                X_cand_up = torch.tensor(self._decode_to_nearest_value_target_space(enc_Y, X_cand_encoded.numpy()))
            else:
                raise NotImplementedError()
            
            X_cand_up = torch.unique(X_cand_up, dim=0)
            X_cand_up_back_up = X_cand_up.detach().clone()

            # constrain some dims to the mean
            if self.cma_enc == 0:
                mean_Y_ = torch.clip(mean_Y, -1, 1)
                mean_Y_up = from_1_around_origin(mean_Y_, lb_Y, ub_Y) 
            elif self.cma_enc == 1:
                mean_Y_up_encoded = from_1_around_origin(mean_Y, encoded_lb_Y, encoded_ub_Y) 
                mean_Y_up = torch.tensor(self._decode_to_nearest_value_target_space(enc_Y, mean_Y_up_encoded.numpy()))
            else:
                raise NotImplementedError()
            mean_Y_up_round = deepcopy(mean_Y_up)
            mean_Y_up_round[:, non_cont_ids] = torch.round(mean_Y_up_round[:, non_cont_ids])
            mean_Y_cat_up_round = mean_Y_up_round[:, non_cont_ids].squeeze()
            X_cand_up = torch.vstack((X_cand_up, mean_Y_up_round))

            if length_discrete < len(non_cont_ids):
                count = 0
                for i, x in enumerate(X_cand_up):
                    x_temp_cat = X_cand_up[i, non_cont_ids]
                    d = np.count_nonzero(x_temp_cat != mean_Y_cat_up_round)
                    if d > length_discrete:
                        count += 1 
                        non_identical_idx = torch.where(x_temp_cat != mean_Y_cat_up_round)[0]
                        n_min_fix, n_max_fix = d - length_discrete, len(non_identical_idx) - 1

                        n_fix = np.random.choice(np.arange(n_min_fix, n_max_fix)) if n_max_fix > n_min_fix else n_min_fix
                        keep_axis = np.random.choice(non_identical_idx, n_fix, replace=False)
                        x_temp_cat[keep_axis] = mean_Y_cat_up_round[keep_axis]
                            
                        assert np.count_nonzero(x_temp_cat != mean_Y_cat_up_round) == d - n_fix <= length_discrete
                        X_cand_up[i, non_cont_ids] = x_temp_cat
                        ...
                    ...
                X_cand_up_back_up = X_cand_up.detach().clone()
                if 1:
                    if self.cma_enc == 0:
                        X_cand_up_1a = to_1_around_origin(X_cand_up, lb_Y, ub_Y)
                    elif self.cma_enc == 1:
                        X_cand_up_enc = torch.from_numpy(self._encode_target_space(enc_Y, X_cand_up.numpy()))
                        X_cand_up_1a = to_1_around_origin(X_cand_up_enc, encoded_lb_Y, encoded_ub_Y)
                mask = self._is_in_ellipse_custom(mean_Y_np, cov_Y_np, X_cand_up_1a.numpy())
                X_cand_up = X_cand_up[mask]
                ...
            if len(X_cand_up) == 0:
                logging.info("No candidates, use all candidates")
                X_cand_up = X_cand_up_back_up
            max_hamming_dist = max(np.count_nonzero(X_cand_up[:, non_cont_ids] != mean_Y_cat_up_round, axis=1))
            if length_discrete < max_hamming_dist:
                logging.info(f'max_hamming_dist exceeds threshold ({max_hamming_dist} > {length_discrete})')

            # make unique
            # if len(cont_ids) == 0:
            X_cand_up = torch.unique(X_cand_up, dim=0)


            # convert from categorical to onehot
            temp = torch.vstack((mean_Y_up_round, X_cand_up))
            temp_onehot = torch.zeros((temp.shape[0], random_embedding.target_dim), dtype=temp.dtype)
            start = 0
            for i, bin in enumerate(random_embedding.bins):
                end = start + bin.dims_required
                if bin.parameter_type == ParameterType.CATEGORICAL:
                    temp_onehot[:, start:end][np.arange(temp.shape[0]), temp[:, i].long()] = 1
                else:
                    temp_onehot[:, start:end] = temp[:, i:i+1]
                start = end
            mean_Y_up_one_hot, X_cand_up_onehot = torch.vsplit(temp_onehot, [1])

            assert len(np.unique(X_cand_up_onehot) == len(X_cand_up))
            
            if self.cma_enc == 0:
                lower, upper = lb_Y, ub_Y
            elif self.cma_enc == 1:
                lower, upper = encoded_lb_Y, encoded_ub_Y
            else:
                raise NotImplementedError()
            
            def is_in_Y_ellipse_cat(u_):
                # keep continuous part the same as mean_Y
                u_full = torch.repeat_interleave(mean_Y, u_.shape[0], dim=0)
                if self.cma_enc == 0:
                    u_full[:, non_cont_ids] = u_[:, non_cont_ids].double()
                elif self.cma_enc == 1:
                    temp_encoded = torch.tensor(self._encode_target_space(enc_Y, u_.double().numpy()))
                    u_full[:, non_cont_ids] = temp_encoded[:, non_cont_ids]
                else:
                    raise NotImplementedError()
                u_full[:, non_cont_ids] = to_1_around_origin(u_full[:, non_cont_ids], lower[non_cont_ids], upper[non_cont_ids])
                d = self._mahalanobis_v_torch(u_full, mean_Y.flatten(), cov_Y)
                std = self._get_es_threshold(len(mean_Y.flatten()))
                return d < std
                
            return X_cand_up_onehot, mean_Y_up_one_hot, is_in_Y_ellipse_cat
        
        def create_candidates_mixed_cat_component(n_cand: int, **kwargs):
            """
            Create candidates, for mixed case, categorical component
            """
            trust_region = kwargs['trust_region']
            random_embedding = kwargs['random_embedding']
            x_best_ordinal = kwargs['x_best_ordinal'] # continuous variables should be in [0, 1]

            A = random_embedding._A.double()
            # b = 0 random_embedding._b.double()
            # c = random_embedding._c.double()
            ATA = torch.mm(A.t(), A) 
            P = torch.mm(torch.linalg.inv(ATA), A.t())

            tr_length_continuous = trust_region.length_continuous
            tr_length_discrete_continuous = trust_region.length_discrete_continuous
            tr_length_init_discrete = trust_region.length_init_discrete
            tr_length_max_discrete = trust_region.length_max_discrete

            non_cont_ids = [idx for idx, bin in enumerate(random_embedding.bins) if bin.parameter_type != ParameterType.CONTINUOUS]
            cont_ids = [idx for idx, bin in enumerate(random_embedding.bins) if bin.parameter_type == ParameterType.CONTINUOUS]
            bin_ids = [idx for idx, bin in enumerate(random_embedding.bins) if bin.parameter_type == ParameterType.BINARY]
            cat_ids = [idx for idx, bin in enumerate(random_embedding.bins) if bin.parameter_type == ParameterType.CATEGORICAL]
            if x_best_ordinal is not None:
                assert torch.all(x_best_ordinal[:, cont_ids] >= 0) and torch.all(x_best_ordinal[:, cont_ids] <= 1)

            length_continuous = tr_length_continuous
            target_dim = len(random_embedding.bins)
            length_discrete = max(1, round(target_dim * tr_length_discrete_continuous/tr_length_max_discrete))
            
            if len(bin_ids) == 0 and len(cat_ids) != 0:
                length_discrete = max(1, round(tr_length_discrete_continuous/2))
            elif len(bin_ids) != 0 and len(cat_ids) == 0:
                length_discrete = max(1, round(tr_length_discrete_continuous))
            else:
                raise NotImplementedError()
            # logging.info(f'cma-length_discrete={length_discrete}')

            temp_mean_X = self.bound_tf.repair(self.es_mean)
            mean_X = torch.tensor(temp_mean_X, dtype=torch.float64)
            cov_X = torch.tensor(self.es_sigma**2 * self.es_cov, dtype=torch.float64)

            # scale the mvn from [self.lb, self.ub]^D to [-1, 1]^D
            self_lb_torch = torch.tensor(self.lb, dtype=torch.float64)
            self_ub_torch = torch.tensor(self.ub, dtype=torch.float64)
            scale_mtrx = torch.diag((2/(self_ub_torch - self_lb_torch)))
            mean_X_scaled = to_1_around_origin(mean_X.reshape(1, -1), self_lb_torch, self_ub_torch).squeeze()

            cov_X_scaled = scale_mtrx @ cov_X @ scale_mtrx.T
            eival, eivec = torch.linalg.eigh(cov_X_scaled)

            new_eigval = torch.sqrt(eival)
            # trust region
            dims_to_scale = []
            for i in range(len(eival)):
                eivec_i = eivec[:, i]
                if any(eivec_i[self.cont_idx] != 0):
                    dims_to_scale.append(i)
            new_eigval[dims_to_scale] *= length_continuous
            new_eigval = torch.square(new_eigval) 

            cov_X_scaled_tr = eivec @ torch.diag(new_eigval) @ torch.linalg.inv(eivec)

            # Remove random shift for categorical
            ub_X = torch.zeros(len(random_embedding.parameters))
            lb_X = torch.zeros(len(random_embedding.parameters))
            for i, param in enumerate(random_embedding.parameters):
                if param.type == ParameterType.CATEGORICAL:
                    ub_X[i] = param.dims_required - 1
                    lb_X[i] = 0
                else:
                    ub_X[i] = param.upper_bound
                    lb_X[i] = param.lower_bound
            assert np.allclose(lb_X, self.obj_func.bounds_m[:,0]) and np.allclose(ub_X, self.obj_func.bounds_m[:,1])
            
            delta_m = mean_X_scaled.detach().clone()

            # project mvn to Y: [-1, 1]^d from [-1, 1]^D
            mean_Y = torch.matmul(P, delta_m.reshape(-1, 1)).t()
            cov_Y = P @ cov_X_scaled_tr @ P.T
                            
            # sample in [-1, 1]^d
            if x_best_ordinal is not None:
                fixed_continuous_1around = x_best_ordinal[:, cont_ids] * 2 - 1
                mean_Y[:, cont_ids] = fixed_continuous_1around
            else:
                fixed_continuous_1around = mean_Y[:, cont_ids]
            mean_Y_np = mean_Y.numpy().squeeze()
            cov_Y_np = cov_Y.numpy()
            target_dim_cat = len(non_cont_ids)

            X_cand = np.random.multivariate_normal(mean_Y_np, cov_Y_np, size=int(1.2*n_cand))
            X_cand = X_cand[self._is_in_ellipse_custom(mean_Y_np, cov_Y_np, X_cand)][:n_cand,:]
            bound_tf = cma.BoundTransform([-1., 1.])
            X_cand_1around = np.array([bound_tf.repair(x) for x in X_cand])
            X_cand_1around[:, cont_ids] = fixed_continuous_1around

            # scale to Y: [0, ...]^d
            if self.cma_enc == 0:
                ub_Y = torch.zeros(len(random_embedding.bins))
                lb_Y = torch.zeros(len(random_embedding.bins))
                for i, bin in enumerate(random_embedding.bins):
                    if bin.parameter_type == ParameterType.CATEGORICAL:
                        ub_Y[i] = bin.dims_required - 1 + self.bound_expansion
                        lb_Y[i] = 0 - self.bound_expansion
                    elif bin.parameter_type == ParameterType.BINARY:
                        ub_Y[i] = 1. + self.bound_expansion
                        lb_Y[i] = 0. - self.bound_expansion
                    else:
                        ub_Y[i] = 1
                        lb_Y[i] = 0
                X_cand_up = from_1_around_origin(torch.tensor(X_cand_1around), lb_Y, ub_Y) 
                X_cand_up[:, non_cont_ids] = torch.round(X_cand_up[:, non_cont_ids])
            elif self.cma_enc == 1:
                enc_Y, encoded_lb_Y, encoded_ub_Y = self._project_encoder_to_target_space(random_embedding, P)
                X_cand_encoded = from_1_around_origin(torch.tensor(X_cand_1around), encoded_lb_Y, encoded_ub_Y) 
                X_cand_up = torch.tensor(self._decode_to_nearest_value_target_space(enc_Y, X_cand_encoded.numpy()))
            else:
                raise NotImplementedError()
            
            X_cand_up = torch.unique(X_cand_up, dim=0)
            X_cand_up_back_up = X_cand_up.detach().clone()

            # constrain some dims to the mean
            if self.cma_enc == 0:
                mean_Y_ = torch.clip(mean_Y, -1, 1)
                mean_Y_up = from_1_around_origin(mean_Y_, lb_Y, ub_Y) 
            elif self.cma_enc == 1:
                mean_Y_up_encoded = from_1_around_origin(mean_Y, encoded_lb_Y, encoded_ub_Y) 
                mean_Y_up = torch.tensor(self._decode_to_nearest_value_target_space(enc_Y, mean_Y_up_encoded.numpy()))
            else:
                raise NotImplementedError()
            mean_Y_up_round = deepcopy(mean_Y_up)
            mean_Y_up_round[:, non_cont_ids] = torch.round(mean_Y_up_round[:, non_cont_ids])
            mean_Y_cat_up_round = mean_Y_up_round[:, non_cont_ids].squeeze()
            X_cand_up = torch.vstack((X_cand_up, mean_Y_up_round))
            if length_discrete < len(non_cont_ids):
                count = 0
                for i, x in enumerate(X_cand_up):
                    x_temp_cat = X_cand_up[i, non_cont_ids]
                    d = np.count_nonzero(x_temp_cat != mean_Y_cat_up_round)
                    if d > length_discrete:
                        count += 1 
                        non_identical_idx = torch.where(x_temp_cat != mean_Y_cat_up_round)[0]
                        n_min_fix, n_max_fix = d - length_discrete, len(non_identical_idx) - 1

                        n_fix = np.random.choice(np.arange(n_min_fix, n_max_fix)) if n_max_fix > n_min_fix else n_min_fix
                        keep_axis = np.random.choice(non_identical_idx, n_fix, replace=False)
                        x_temp_cat[keep_axis] = mean_Y_cat_up_round[keep_axis]
                            
                        assert np.count_nonzero(x_temp_cat != mean_Y_cat_up_round) == d - n_fix <= length_discrete
                        X_cand_up[i, non_cont_ids] = x_temp_cat
                        ...
                    ...      
                X_cand_up_back_up = X_cand_up.detach().clone()
                if 1:
                    if self.cma_enc == 0:
                        X_cand_up_1a = to_1_around_origin(X_cand_up, lb_Y, ub_Y)
                    elif self.cma_enc == 1:
                        X_cand_up_enc = torch.from_numpy(self._encode_target_space(enc_Y, X_cand_up.numpy()))
                        X_cand_up_1a = to_1_around_origin(X_cand_up_enc, encoded_lb_Y, encoded_ub_Y)
                mask = self._is_in_ellipse_custom(mean_Y_np, cov_Y_np, X_cand_up_1a.numpy())
                X_cand_up = X_cand_up[mask]
                ...
            if len(X_cand_up) == 0:
                logging.info("No candidates, use all candidates")
                X_cand_up = X_cand_up_back_up

            max_hamming_dist = max(np.count_nonzero(X_cand_up[:, non_cont_ids] != mean_Y_cat_up_round, axis=1))
            if length_discrete < max_hamming_dist:
                logging.info(f'max_hamming_dist exceeds threshold ({max_hamming_dist} > {length_discrete})')

            # make unique
            X_cand_up = torch.unique(X_cand_up, dim=0)

            # convert from categorical to onehot
            temp = torch.vstack((mean_Y_up_round, X_cand_up))
            temp_onehot = torch.zeros((temp.shape[0], random_embedding.target_dim), dtype=temp.dtype)
            start = 0
            for i, bin in enumerate(random_embedding.bins):
                end = start + bin.dims_required
                if bin.parameter_type == ParameterType.CATEGORICAL:
                    temp_onehot[:, start:end][np.arange(temp.shape[0]), temp[:, i].long()] = 1
                else:
                    temp_onehot[:, start:end] = temp[:, i:i+1]
                start = end
            mean_Y_up_one_hot, X_cand_up_onehot = torch.vsplit(temp_onehot, [1])

            assert len(np.unique(X_cand_up_onehot) == len(X_cand_up))
            
            if self.cma_enc == 0:
                lower, upper = lb_Y, ub_Y
            elif self.cma_enc == 1:
                lower, upper = encoded_lb_Y, encoded_ub_Y
            else:
                raise NotImplementedError()
            
            def is_in_Y_ellipse_cat(u_):
                # keep continuous part the same as mean_Y
                u_full = torch.repeat_interleave(mean_Y, u_.shape[0], dim=0)
                if self.cma_enc == 0:
                    u_full[:, non_cont_ids] = u_[:, non_cont_ids].double()
                elif self.cma_enc == 1:
                    temp_encoded = torch.tensor(self._encode_target_space(enc_Y, u_.double().numpy()))
                    u_full[:, non_cont_ids] = temp_encoded[:, non_cont_ids]
                else:
                    raise NotImplementedError()
                u_full[:, non_cont_ids] = to_1_around_origin(u_full[:, non_cont_ids], lower[non_cont_ids], upper[non_cont_ids])
                d = self._mahalanobis_v_torch(u_full, mean_Y.flatten(), cov_Y)
                std = self._get_es_threshold(len(mean_Y.flatten()))
                return d < std
      
            return X_cand_up_onehot, mean_Y_up_one_hot, is_in_Y_ellipse_cat
        
        def create_candidates_mixed_cont_component(n_cand: int, **kwargs):
            """
            Create candidates, for mixed case, continuous component
            """
            trust_region = kwargs['trust_region']
            random_embedding = kwargs['random_embedding']
            x_best_ordinal = kwargs['x_best_ordinal'] # categorical variables should be in ordinal form
            A = random_embedding._A.double()
            # b = 0 random_embedding._b.double()
            # c = random_embedding._c.double()
            ATA = torch.mm(A.t(), A) 
            P = torch.mm(torch.linalg.inv(ATA), A.t())

            tr_length_continuous = trust_region.length_continuous
            tr_length_discrete_continuous = trust_region.length_discrete_continuous
            tr_length_init_discrete = trust_region.length_init_discrete
            tr_length_max_discrete = trust_region.length_max_discrete

            non_cont_ids = [idx for idx, bin in enumerate(random_embedding.bins) if bin.parameter_type != ParameterType.CONTINUOUS]
            cont_ids = [idx for idx, bin in enumerate(random_embedding.bins) if bin.parameter_type == ParameterType.CONTINUOUS]
            bin_ids = [idx for idx, bin in enumerate(random_embedding.bins) if bin.parameter_type == ParameterType.BINARY]
            cat_ids = [idx for idx, bin in enumerate(random_embedding.bins) if bin.parameter_type == ParameterType.CATEGORICAL]

            length_continuous = tr_length_continuous
            target_dim = len(random_embedding.bins)
            length_discrete = max(1, round(target_dim * tr_length_discrete_continuous/tr_length_max_discrete))
            
            if len(bin_ids) == 0 and len(cat_ids) != 0:
                length_discrete = max(1, round(tr_length_discrete_continuous/2))
            elif len(bin_ids) != 0 and len(cat_ids) == 0:
                length_discrete = max(1, round(tr_length_discrete_continuous))
            else:
                raise NotImplementedError()
            # logging.info(f'cma-length_discrete={length_discrete}')

            temp_mean_X = self.bound_tf.repair(self.es_mean)
            mean_X = torch.tensor(temp_mean_X, dtype=torch.float64)
            cov_X = torch.tensor(self.es_sigma**2 * self.es_cov, dtype=torch.float64)

            # scale the mvn from [self.lb, self.ub]^D to [-1, 1]^D
            self_lb_torch = torch.tensor(self.lb, dtype=torch.float64)
            self_ub_torch = torch.tensor(self.ub, dtype=torch.float64)
            scale_mtrx = torch.diag((2/(self_ub_torch - self_lb_torch)))
            mean_X_scaled = to_1_around_origin(mean_X.reshape(1, -1), self_lb_torch, self_ub_torch).squeeze()

            cov_X_scaled = scale_mtrx @ cov_X @ scale_mtrx.T
            eival, eivec = torch.linalg.eigh(cov_X_scaled)

            new_eigval = torch.sqrt(eival)
            # trust region
            dims_to_scale = []
            for i in range(len(eival)):
                eivec_i = eivec[:, i]
                if any(eivec_i[self.cont_idx] != 0):
                    dims_to_scale.append(i)
            new_eigval[dims_to_scale] *= length_continuous
            new_eigval = torch.square(new_eigval) 

            cov_X_scaled_tr = eivec @ torch.diag(new_eigval) @ torch.linalg.inv(eivec)

            # Remove random shift for categorical
            ub_X = torch.zeros(len(random_embedding.parameters))
            lb_X = torch.zeros(len(random_embedding.parameters))
            for i, param in enumerate(random_embedding.parameters):
                if param.type == ParameterType.CATEGORICAL:
                    ub_X[i] = param.dims_required - 1
                    lb_X[i] = 0
                else:
                    ub_X[i] = param.upper_bound
                    lb_X[i] = param.lower_bound
            assert np.allclose(lb_X, self.obj_func.bounds_m[:,0]) and np.allclose(ub_X, self.obj_func.bounds_m[:,1])
            
            if 1:
                ub_Y = torch.zeros(len(random_embedding.bins))
                lb_Y = torch.zeros(len(random_embedding.bins))
                for i, bin in enumerate(random_embedding.bins):
                    if bin.parameter_type == ParameterType.CATEGORICAL:
                        ub_Y[i] = bin.dims_required - 1 + self.bound_expansion
                        lb_Y[i] = 0 - self.bound_expansion
                    elif bin.parameter_type == ParameterType.BINARY:
                        ub_Y[i] = 1. + self.bound_expansion
                        lb_Y[i] = 0. - self.bound_expansion
                    else:
                        ub_Y[i] = 1
                        lb_Y[i] = 0
            
            if self.cma_enc == 1:
                enc_Y, encoded_lb_Y, encoded_ub_Y = self._project_encoder_to_target_space(random_embedding, P)

            delta_m = mean_X_scaled.detach().clone()

            # project mvn to Y: [-1, 1]^d from [-1, 1]^D
            mean_Y = torch.matmul(P, delta_m.reshape(-1, 1)).t()
            cov_Y = P @ cov_X_scaled_tr @ P.T
                            
            # sample in [-1, 1]^d
            if x_best_ordinal is not None:
                if self.cma_enc == 0:
                    fixed_categorical = x_best_ordinal[:, non_cont_ids]
                    fixed_categorical_1around = to_1_around_origin(fixed_categorical.double(), lb_Y[non_cont_ids], ub_Y[non_cont_ids]) 
                elif self.cma_enc == 1:
                    fixed_categorical_enc = torch.from_numpy(self._encode_target_space(enc_Y, x_best_ordinal.numpy())).double()
                    fixed_categorical_1around = to_1_around_origin(fixed_categorical_enc[:, non_cont_ids], encoded_lb_Y[non_cont_ids], encoded_ub_Y[non_cont_ids]) 
                mean_Y[:, non_cont_ids] = fixed_categorical_1around
            mean_Y_np = mean_Y.numpy().squeeze()
            cov_Y_np = cov_Y.numpy()

            X_cand = np.random.multivariate_normal(mean_Y_np, cov_Y_np, size=int(1.2*n_cand))
            X_cand = X_cand[self._is_in_ellipse_custom(mean_Y_np, cov_Y_np, X_cand)][:n_cand,:]
            bound_tf = cma.BoundTransform([-1., 1.])
            X_cand_1around = np.array([bound_tf.repair(x) for x in X_cand])
            X_cand_1around[:, non_cont_ids] = fixed_categorical_1around

            # scale to Y: [0, ...]^d
            if self.cma_enc == 0:
                X_cand_up = from_1_around_origin(torch.tensor(X_cand_1around), lb_Y, ub_Y) 
                X_cand_up[:, non_cont_ids] = torch.round(X_cand_up[:, non_cont_ids])
            elif self.cma_enc == 1:
                X_cand_encoded = from_1_around_origin(torch.tensor(X_cand_1around), encoded_lb_Y, encoded_ub_Y) 
                X_cand_up = torch.tensor(self._decode_to_nearest_value_target_space(enc_Y, X_cand_encoded.numpy()))
            else:
                raise NotImplementedError()

            # constrain some dims to the mean
            if self.cma_enc == 0:
                mean_Y_ = torch.clip(mean_Y, -1, 1)
                mean_Y_up = from_1_around_origin(mean_Y_, lb_Y, ub_Y) 
            elif self.cma_enc == 1:
                mean_Y_up_encoded = from_1_around_origin(mean_Y, encoded_lb_Y, encoded_ub_Y) 
                mean_Y_up = torch.tensor(self._decode_to_nearest_value_target_space(enc_Y, mean_Y_up_encoded.numpy()))
            else:
                raise NotImplementedError()
            mean_Y_up_round = deepcopy(mean_Y_up)
            mean_Y_up_round[:, non_cont_ids] = torch.round(mean_Y_up_round[:, non_cont_ids])
            mean_Y_cat_up_round = mean_Y_up_round[:, non_cont_ids].squeeze()

            # convert from categorical to onehot
            temp = torch.vstack((mean_Y_up_round, X_cand_up))
            temp_onehot = torch.zeros((temp.shape[0], random_embedding.target_dim), dtype=temp.dtype)
            start = 0
            for i, bin in enumerate(random_embedding.bins):
                end = start + bin.dims_required
                if bin.parameter_type == ParameterType.CATEGORICAL:
                    temp_onehot[:, start:end][np.arange(temp.shape[0]), temp[:, i].long()] = 1
                else:
                    temp_onehot[:, start:end] = temp[:, i:i+1]
                start = end
            mean_Y_up_one_hot, X_cand_up_onehot = torch.vsplit(temp_onehot, [1])

            assert len(np.unique(X_cand_up_onehot) == len(X_cand_up))
            
            if self.cma_enc == 0:
                lower, upper = lb_Y, ub_Y
            elif self.cma_enc == 1:
                lower, upper = encoded_lb_Y, encoded_ub_Y
            else:
                raise NotImplementedError()
             
            def is_in_Y_ellipse(u, current_best):
                # keep categorical part the same as mean_Y
                if u.ndim == 1:
                    u_ = u[None, :]                    
                elif u.ndim == 3:
                    u_ = u.squeeze(dim=1)
                else:
                    u_ = u
                if self.cma_enc == 1:
                    current_best = torch.tensor(self._encode_target_space(enc_Y, current_best[None, :].double().numpy())).squeeze()
                u_full = torch.repeat_interleave(mean_Y, u_.shape[0], dim=0)
                u_full[:, cont_ids] = u_
                u_full[:, cont_ids] = to_1_around_origin(u_full[:, cont_ids], lower[cont_ids], upper[cont_ids])
                d = self._mahalanobis_v_torch(u_full, mean_Y.flatten(), cov_Y)
                std = self._get_es_threshold(len(mean_Y.flatten()))
                return std - d
            
            return X_cand_up_onehot, mean_Y_up_one_hot, is_in_Y_ellipse
        
        maximum_number_evaluations_until_input_dim = int(self.max_evals/2)
        logging.debug(f'maximum_number_evaluations_until_input_dim={maximum_number_evaluations_until_input_dim}')
        func_core = self.obj_func.func_core if hasattr(self.obj_func, 'func_core') else None
        assert func_core, 'Need to define func_core in the objective function'
        # Convert to Bounce API (see Bounce code for details)
        if self.func_name == 'pest':
            problem = BouncePestControl(func=func_core)
        elif self.func_name == 'antibody':
            problem = BounceAntibodyDesgin(func=func_core) 
        elif self.func_name == 'labs':
            problem = BounceLabs(func=func_core)
        elif self.func_name == 'maxsat28':
            problem = BounceMaxSAT28(func=func_core)
        elif self.func_name == 'ackley53m':
            problem = BounceAckley53m(func=func_core) 
        elif self.func_name == 'svm':
            problem = BounceSVMMixed(func=func_core)
        elif self.func_name == 'cco':
            problem = BounceCellNetOpt(func=func_core)
        elif self.func_name == 'ackley20c':
            problem = BounceAckley20c(func=func_core)
        else:
            raise NotImplementedError()
        
        assert np.allclose(self.obj_func.bounds_m[:, 0], problem._lb_vec.numpy())
        assert np.allclose(self.obj_func.bounds_m[:, 1], problem._ub_vec.numpy())

        bounce = Bounce(
            benchmark=problem,
            number_initial_points=self.n_init,
            initial_target_dimensionality=5, # see Bounce code for details
            number_new_bins_on_split=2,      # see Bounce code for details
            maximum_number_evaluations=self.max_evals,
            batch_size=1,
            results_dir=f"bounce_results/{self.obj_func.name}",
            device="cpu",
            maximum_number_evaluations_until_input_dim=maximum_number_evaluations_until_input_dim,
            cma_init=cma_init,
            cma_update=cma_update,
            sample_acq_points=create_candidates_discrete,
            sample_acq_points_cat=create_candidates_mixed_cat_component,
            sample_acq_points_cont=create_candidates_mixed_cont_component,
        )
        bounce.run()

        return self.get_optimization_results()
        
    def _mahalanobis_v_torch(self, d, mean, Sigma):
        """
        Torch version
        """
        Sigma_inv = torch.linalg.inv(Sigma)
        xdiff = d - mean
        return torch.sqrt(torch.einsum('ij,im,mj->i', xdiff, xdiff, Sigma_inv))
            