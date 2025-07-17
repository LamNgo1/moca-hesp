import math
import sys
from copy import deepcopy
from casmopolitan_mocahesp.bo.localbo_cat import CASMOPOLITANCat

import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from casmopolitan_mocahesp.bo.localbo_utils import train_gp
from casmopolitan_mocahesp.bo.localbo_utils import from_unit_cube, latin_hypercube, to_unit_cube
from casmopolitan_mocahesp.bo.localbo_utils import onehot2ordinal
from casmopolitan_mocahesp.bo.localbo_utils import random_sample_within_discrete_tr_ordinal


class CASMOPOLITANMixed(CASMOPOLITANCat):
    """

    Parameters
    ----------
    config: the configuration for the categorical dimensions
    cat_dims: the list of indices of dimensions that are categorical
    cont_dims: the list of indices of dimensions that are continuous
    *Note*: in general, you should have the first d_cat dimensions as cat_dims, and the rest as the continuous dims.
    lb : Lower variable bounds of the continuous dimensions, numpy.array, shape (d_cont,).
    ub : Upper variable bounds of the continuous dimensions, numpy.array, shape (d_cont,).
    n_init : Number of initial points (2*dim is recommended), int.
    max_evals : Total evaluation budget, int.
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    max_cholesky_size : Largest number of training points where we use Cholesky, int
    n_training_steps : Number of training steps for learning the GP hypers, int
    min_cuda : We use float64 on the CPU if we have this or fewer datapoints
    device : Device to use for GP fitting ("cpu" or "cuda")
    dtype : Dtype to use for GP fitting ("float32" or "float64")
    """

    def __init__(
            self,
            config,
            cat_dim,
            cont_dim,
            lb,
            ub,
            n_init,
            max_evals,
            batch_size=1,
            int_constrained_dims=None,
            verbose=True,
            use_ard=True,
            max_cholesky_size=2000,
            n_training_steps=50,
            min_cuda=1024,
            device="cpu",
            dtype="float32",
            kernel_type='mixed',
            **kwargs
    ):
        super(CASMOPOLITANMixed, self).__init__(len(cat_dim) + len(cont_dim),
                                                n_init, max_evals, config, batch_size, verbose, use_ard,
                                                max_cholesky_size, n_training_steps, min_cuda,
                                                device, dtype, kernel_type, **kwargs)
        # Very basic input checks
        assert lb.ndim == 1 and ub.ndim == 1
        assert len(lb) == len(ub)
        assert np.all(ub > lb)

        self.kwargs = kwargs
        # Save function information for both the continuous and categorical parts.
        self.cat_dims, self.cont_dims = cat_dim, cont_dim
        self.int_constrained_dims = int_constrained_dims
        # self.n_categories = n_cats
        self.lb = lb
        self.ub = ub

    def _adjust_length(self, fX_next):
        if np.min(fX_next) <= np.min(self._fX) - 1e-3 * math.fabs(np.min(self._fX)):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += 1

        if self.succcount == self.succtol:  # Expand trust region
            # self.length = min([self.tr_multiplier * self.length, self.length_max])
            # For the Hamming distance-bounded trust region, we additively (instead of multiplicatively) adjust.
            self.length_discrete = int(min(self.length_discrete * self.tr_multiplier, self.length_max_discrete))
            self.length = min(self.length * self.tr_multiplier, self.length_max)
            self.succcount = 0
            print("expand", self.length, self.length_discrete)
        elif self.failcount == self.failtol:  # Shrink trust region
            self.failcount = 0
            # Ditto for shrinking.
            self.length_discrete = int(self.length_discrete / self.tr_multiplier)
            self.length = max(self.length / self.tr_multiplier, self.length_min)
            print("Shrink", self.length, self.length_discrete)

    def _create_and_select_candidates(self, X, fX, length, n_training_steps, hypers):
        # assert X.min() >= 0.0 and X.max() <= 1.0
        # Figure out what device we are running on
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float32
        else:
            device, dtype = self.device, self.dtype
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            gp = train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers,
                kern=self.kernel_type,
                cat_dims=self.cat_dims, cont_dims=self.cont_dims,
                int_constrained_dims=self.int_constrained_dims,
                noise_variance=self.kwargs['noise_variance'] if 'noise_variance' in self.kwargs else None
            )
            # Save state dict
            hypers = gp.state_dict()

        kwargs = {}
        kwargs["length_discrete"] = self.length_discrete
        kwargs["length_init_discrete"] = self.length_init_discrete
        kwargs["tr"] = {
            'length': self.length,
            'length_discrete': self.length_discrete,
            'failcount': self.failcount,
            'succcount': self.succcount,
        }
        X_cand = self.create_candidates(n_cand=5000, length=self.length, **kwargs) 
        # Generate n_cand candidates for the continuous variables, in their trust region
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand, dtype=torch.float32)
            y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()

        # Select the best candidates
        X_next = np.ones((self.batch_size, self.dim))
        y_next = np.ones((self.batch_size, 1))
        for i in range(self.batch_size):
            indbest = np.argmin(y_cand[:, i])
            X_next[i, :] = deepcopy(X_cand[indbest, :])
            y_next[i, :] = deepcopy(y_cand[indbest, i])
            y_cand[indbest, :] = np.inf

        X_next = np.array(X_next)
        return X_next
