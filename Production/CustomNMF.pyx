# cython: language_level=3
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

cimport numpy as np  # noqa
import numpy as np
from six.moves import range
from surprise.utils import get_rng
from surprise.prediction_algorithms.algo_base import AlgoBase
from surprise.prediction_algorithms.predictions import PredictionImpossible

class NMF(AlgoBase):
    """A collaborative filtering algorithm based on Non-negative Matrix
    Factorization.
    This algorithm is very similar to :class:`SVD`. The prediction
    :math:`\\hat{r}_{ui}` is set as:
    .. math::
        \hat{r}_{ui} = q_i^Tp_u,
    where user and item factors are kept **positive**. Our implementation
    follows that suggested in :cite:`NMF:2014`, which is equivalent to
    :cite:`Zhang96` in its non-regularized form. Both are direct applications
    of NMF for dense matrices :cite:`NMF_algo`.
    The optimization procedure is a (regularized) stochastic gradient descent
    with a specific choice of step size that ensures non-negativity of factors,
    provided that their initial values are also positive.
    At each step of the SGD procedure, the factors :math:`f` or user :math:`u`
    and item :math:`i` are updated as follows:
    .. math::
        p_{uf} &\\leftarrow p_{uf} &\cdot \\frac{\\sum_{i \in I_u} q_{if}
        \\cdot r_{ui}}{\\sum_{i \in I_u} q_{if} \\cdot \\hat{r_{ui}} +
        \\lambda_u |I_u| p_{uf}}\\\\
        q_{if} &\\leftarrow q_{if} &\cdot \\frac{\\sum_{u \in U_i} p_{uf}
        \\cdot r_{ui}}{\\sum_{u \in U_i} p_{uf} \\cdot \\hat{r_{ui}} +
        \lambda_i |U_i| q_{if}}\\\\
    where :math:`\lambda_u` and :math:`\lambda_i` are regularization
    parameters.
    This algorithm is highly dependent on initial values. User and item factors
    are uniformly initialized between ``init_low`` and ``init_high``. Change
    them at your own risks!
    A biased version is available by setting the ``biased`` parameter to
    ``True``. In this case, the prediction is set as
    .. math::
        \hat{r}_{ui} = \mu + b_u + b_i + q_i^Tp_u,
    still ensuring positive factors. Baselines are optimized in the same way as
    in the :class:`SVD` algorithm. While yielding better accuracy, the biased
    version seems highly prone to overfitting so you may want to reduce the
    number of factors (or increase regularization).
    Args:
        n_factors: The number of factors. Default is ``15``.
        n_epochs: The number of iteration of the SGD procedure. Default is
            ``50``.
        biased(bool): Whether to use baselines (or biases). Default is
            ``False``.
        reg_pu: The regularization term for users :math:`\lambda_u`. Default is
            ``0.06``.
        reg_qi: The regularization term for items :math:`\lambda_i`. Default is
            ``0.06``.
        reg_bu: The regularization term for :math:`b_u`. Only relevant for
            biased version. Default is ``0.02``.
        reg_bi: The regularization term for :math:`b_i`. Only relevant for
            biased version. Default is ``0.02``.
        lr_bu: The learning rate for :math:`b_u`. Only relevant for biased
            version. Default is ``0.005``.
        lr_bi: The learning rate for :math:`b_i`. Only relevant for biased
            version. Default is ``0.005``.
        init_low: Lower bound for random initialization of factors. Must be
            greater than ``0`` to ensure non-negative factors. Default is
            ``0``.
        init_high: Higher bound for random initialization of factors. Default
            is ``1``.
        random_state(int, RandomState instance from numpy, or ``None``):
            Determines the RNG that will be used for initialization. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same initialization over multiple calls to
            ``fit()``.  If RandomState instance, this same instance is used as
            RNG. If ``None``, the current RNG from numpy is used.  Default is
            ``None``.
        verbose: If ``True``, prints the current epoch. Default is ``False``.
    Attributes:
        pu(numpy array of size (n_users, n_factors)): The user factors (only
            exists if ``fit()`` has been called)
        qi(numpy array of size (n_items, n_factors)): The item factors (only
            exists if ``fit()`` has been called)
        bu(numpy array of size (n_users)): The user biases (only
            exists if ``fit()`` has been called)
        bi(numpy array of size (n_items)): The item biases (only
            exists if ``fit()`` has been called)
    """

    def __init__(self, n_factors=15, n_epochs=50, biased=False, reg_pu=.06,
                 reg_qi=.06, reg_bu=.02, reg_bi=.02, lr_bu=.005, lr_bi=.005,
                 init_low=0, init_high=1, random_state=None, verbose=False):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.lr_bu = lr_bu
        self.lr_bi = lr_bi
        self.reg_bu = reg_bu
        self.reg_bi = reg_bi
        self.init_low = init_low
        self.init_high = init_high
        self.random_state = random_state
        self.verbose = verbose

        if self.init_low < 0:
            raise ValueError('init_low should be greater than zero')

        AlgoBase.__init__(self)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)
        self.sgd(trainset)

        return self

    def sgd(self, trainset):

        # user and item factors
        cdef np.ndarray[np.double_t, ndim=2] pu
        cdef np.ndarray[np.double_t, ndim=2] qi

        # user and item biases
        cdef np.ndarray[np.double_t] bu
        cdef np.ndarray[np.double_t] bi

        # auxiliary matrices used in optimization process
        cdef np.ndarray[np.double_t, ndim=2] user_num
        cdef np.ndarray[np.double_t, ndim=2] user_denom
        cdef np.ndarray[np.double_t, ndim=2] item_num
        cdef np.ndarray[np.double_t, ndim=2] item_denom

        cdef int u, i, f
        cdef double r, est, l, dot, err
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi
        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double global_mean = self.trainset.global_mean

        # Randomly initialize user and item factors
        rng = get_rng(self.random_state)
        pu = rng.uniform(self.init_low, self.init_high,
                         size=(trainset.n_users, self.n_factors))
        qi = rng.uniform(self.init_low, self.init_high,
                         size=(trainset.n_items, self.n_factors))

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)

        if not self.biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):

            if self.verbose:
                print("Processing epoch {}".format(current_epoch))

            # (re)initialize nums and denoms to zero
            user_num = np.zeros((trainset.n_users, self.n_factors))
            user_denom = np.zeros((trainset.n_users, self.n_factors))
            item_num = np.zeros((trainset.n_items, self.n_factors))
            item_denom = np.zeros((trainset.n_items, self.n_factors))

            # Compute numerators and denominators for users and items factors
            for u, i, r in trainset.all_ratings():

                # compute current estimation and error
                dot = 0  # <q_i, p_u>
                for f in range(self.n_factors):
                    dot += qi[i, f] * pu[u, f]
                est = global_mean + bu[u] + bi[i] + dot
                err = r - est

                # update biases
                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])

                # compute numerators and denominators
                for f in range(self.n_factors):
                    user_num[u, f] += qi[i, f] * r
                    user_denom[u, f] += qi[i, f] * est
                    item_num[i, f] += pu[u, f] * r
                    item_denom[i, f] += pu[u, f] * est

            # Update user factors
            for u in trainset.all_users():
                n_ratings = len(trainset.ur[u])
                for f in range(self.n_factors):
                    user_denom[u, f] += n_ratings * reg_pu * pu[u, f]
                    if user_denom[u,f] != 0: # check if not 0 to prevent div by 0
                        pu[u, f] *= user_num[u, f] / user_denom[u, f]

            # Update item factors
            for i in trainset.all_items():
                n_ratings = len(trainset.ir[i])
                for f in range(self.n_factors):
                    item_denom[i, f] += n_ratings * reg_qi * qi[i, f]
                    if item_denom[i,f] != 0: # check if not 0 to prevent div by 0
                        qi[i, f] *= item_num[i, f] / item_denom[i, f]

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi

    def estimate(self, u, i):
        # Should we cythonize this as well?

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean

            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])

        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unknown.')

        return est