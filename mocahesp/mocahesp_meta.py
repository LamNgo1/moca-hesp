import logging
import math
from copy import deepcopy

import cma
import numpy as np
import scipy
from category_encoders import TargetEncoder

from _benchmark_problems.utils import get_bound

from .utils import from_unit_cube, latin_hypercube, mahalanobis_v, to_unit_cube


class MOCAHESPMeta:
    """
    Meta class for MOCA-HESP.
    This class is used to store the meta information of the MOCA-HESP algorithm.
    """
    def __init__(self, 
                 obj_func, 
                 max_evals, 
                 n_init=20,
                 func_name='',
    ):
        self.obj_func = obj_func
        self.max_evals = max_evals
        self.n_init = n_init
        self.dim = obj_func.input_dim
        self.f = obj_func.func
        self.decode_input = obj_func.decode_input
        self.func_name = func_name
        bounds = get_bound(self.obj_func.bounds)
        self.lb = bounds[:, 0]
        self.ub = bounds[:, 1]

        self.history_x = np.zeros((0, self.dim))    # dataset from past restarts
        self.history_fx = np.zeros((0, 1))          # dataset from past restarts
        self.observed_x = np.zeros((0, self.dim))   # dataset from current restart
        self.observed_fx = np.zeros((0, 1))         # dataset from current restart

        self.bound_tf = cma.BoundTransform([self.lb, self.ub])
        self.popsize = 4 + math.floor(3*np.log(self.dim)) # population
        self.es: cma.CMAEvolutionStrategy = None # current cma class
        self.max_cholesky_size = 2000

        self.encoded_lb = None
        self.encoded_ub = None
        self.enc = None
        self.cma_enc = -1 # use EXP3 to decide encoding

        self.exp3_weights = [1, 1]
        self.exp3_eta = None
        self.exp3_K = 2 # 0 - ordinal, 1 - target
        self.exp3_count = np.zeros(self.exp3_K, dtype=int)  

        uniformBound = all([0==_lb for _lb in self.lb]) and all([1==_ub for _ub in self.ub])
        assert uniformBound, 'please use unit cube bounds [0, 1] for the function'
        assert self.n_init < self.max_evals, 'Number of initial points must be smaller than max # evaluations'

    @property
    def total_eval(self):
        """
        Total number of evaluations.
        """
        return len(self.history_fx) + len(self.observed_fx)
    
    @property
    def cat_idx(self):
        return self.obj_func.categorical_idx_m
    
    @property
    def cat_dim(self):
        return self.obj_func.categorical_dim_m
    
    @property
    def cat_vertices(self):
        return self.obj_func.categorical_vertices_m
    
    @property
    def cont_idx(self):
        return self.obj_func.continuous_idx_m
    
    @property
    def cont_dim(self):
        return self.obj_func.continuous_dim_m
    
    @property
    def bound_ex(self):
        """
        Return bounds extended by 0.5 on both sides to improve sampling probability.
        """
        bound_ret = deepcopy(self.obj_func.bounds_m).astype(float)
        bound_ret[self.cat_idx, 0] -= 0.5
        bound_ret[self.cat_idx, 1] += 0.5
        return bound_ret
    
    @property
    def es_mean(self):
        return self.es.mean
    
    @property
    def es_sigma(self):
        return self.es.sigma
    
    @property
    def es_sigma0(self):
        return self.es.sigma0

    @property
    def es_cov(self):
        sigma_vec = self.es.sigma_vec.scaling * np.ones((1, len(self.es.mean)))
        return sigma_vec * self.es.C * sigma_vec.T
    
    def _get_es_threshold(self, dim):
        std = np.sqrt(scipy.stats.chi2.ppf(q=0.9973,df=dim))
        return std
    
    def _convert_to_x_eval(self, x):
        """
        Convert x from extended bounds to original bounds, for function evaluation.
        """
        X_eval = deepcopy(x)
        if X_eval.ndim == 1:
            X_eval = X_eval.reshape(1, -1)
        X_eval[:, self.cat_idx] = from_unit_cube(X_eval[:, self.cat_idx], self.bound_ex[self.cat_idx, 0], self.bound_ex[self.cat_idx, 1])
        X_eval[:, self.cat_idx] = to_unit_cube(X_eval[:, self.cat_idx], self.obj_func.bounds_m[self.cat_idx, 0], self.obj_func.bounds_m[self.cat_idx, 1])
        return X_eval
    
    def _is_in_ellipse(self, u):
        '''Check whether points is inside a confidence ellipsoid'''
        cov = self.es_sigma**2 *np.array(self.es_cov)
        d = mahalanobis_v(u, self.es_mean, cov)
        mask = d < self._get_es_threshold(len(self.es_mean))
        return mask
    
    def _is_in_ellipse_custom(self, mean, cov, u):
        '''Check whether points is inside a confidence ellipsoid'''
        d = mahalanobis_v(u, mean, cov)
        mask = d < self._get_es_threshold(len(mean))
        return mask
    
    def _decode_to_nearest_value(self, enc: TargetEncoder, X_encoded_):
        def find_nearest_idx(array, value):
            idx = (np.abs(array.reshape(1, -1) - value.reshape(-1, 1))).argmin(axis=1)
            return idx
        
        if X_encoded_.ndim == 1:
            X_encoded = X_encoded_.reshape(1, -1)
        else:
            X_encoded = deepcopy(X_encoded_)
        
        X_decoded_ = np.zeros(X_encoded.shape, dtype='float64')
        for i in range(self.cat_dim):
            nearest_idx = find_nearest_idx(enc.mapping[i].values, X_encoded[:, i])
            nearest_pd_idx = enc.mapping[i].index.values[None, :][:, nearest_idx]
            X_decoded_[:, i] = nearest_pd_idx
            ...
        if 0: # old style, slow
            def find_nearest_idx2(array, value):
                idx = (np.abs(array - value)).argmin()
                return idx
            X_decoded2_ = np.zeros(X_encoded.shape, dtype='float32')
            for j, xx in enumerate(X_encoded):
                for i in range(self.cat_dim):
                    nearest_pd_idx = enc.mapping[i].index[find_nearest_idx2(enc.mapping[i].values, xx[i])]
                    X_decoded2_[j][i] = nearest_pd_idx
                    ...
                ...
            assert np.allclose(X_decoded_, X_decoded2_)
        X_decoded = enc.ordinal_encoder.inverse_transform(X_decoded_).values.reshape(X_encoded_.shape)
        X_decoded[:, self.cont_idx] = X_encoded_[:, self.cont_idx]
        
        return X_decoded
    
    def get_bound_enc(self):
        lb_ret = np.zeros(self.cat_dim)
        ub_ret = np.zeros(self.cat_dim)
        n_max_vertices = max(self.cat_vertices)
        X = np.zeros((n_max_vertices, self.cat_dim))
        for i, vertices in enumerate(self.cat_vertices):
            X[:vertices, i] = np.arange(vertices)
        if self.cont_dim > 0:
            X_cat = deepcopy(X)
            X = np.zeros((n_max_vertices, self.dim))
            X[:, self.cat_idx] = X_cat
        X_tar = self.enc.transform(X)
        ...
        for i in range(self.cat_dim):
            a = np.unique(X_tar[:, i]).tolist()
            lb_ret[i] = a[0] - abs(a[1] - a[0]) / 2.0
            ub_ret[i] = a[-1] + abs(a[-1] - a[-2]) / 2.0

        return lb_ret, ub_ret
    
    def choose_cma_enc(self):
        prob = self.calculate_exp3_probabilities()
        encoding_type = np.random.choice(self.exp3_K, p=prob)
        self.exp3_count[encoding_type] += 1
        logging.info(f'Encoder={"ordinal" if encoding_type == 0 else "target"}; Prob={np.around(prob, 4)}; Counter={self.exp3_count}')
        return encoding_type
    
    def calculate_exp3_probabilities(self):
        weights = self.exp3_weights
        eta = self.exp3_eta
        prob = np.array([1/self.exp3_K for _ in range(self.exp3_K)])
        sum_weight = np.sum(weights)
        for idx, weight_i in enumerate(weights):
            prob[idx] = (1 - eta) * (weight_i / sum_weight) + (eta/self.exp3_K)
        return prob

    def optimize(self):
        """
        Optimize the objective function using the specified solver.
        This method should be implemented in the subclasses.
        """
        raise NotImplementedError("This method should be implemented in the subclasses.")
    
    def get_optimization_results(self):
        X = np.vstack((self.history_x, self.observed_x))
        fx = np.vstack((self.history_fx, self.observed_fx))
        return X, fx
