import jax.numpy as jnp
from jax import random, vmap
from jax.scipy.stats import norm as jax_normal
from jax.scipy.special import gammaln
from dibs_func_utils import _slogdet_jax

class BGe:
    """
    Linear Gaussian BN model corresponding to linear structural equation model (SEM) with additive Gaussian noise.
    Uses Normal-Wishart conjugate parameter prior to allow for closed-form marginal likelihood
    :math:`\\log p(D | G)` and thus allows inference of the marginal posterior :math:`p(G | D)`

    For details on the closed-form expression, refer to

    - Geiger et al. (2002):  https://projecteuclid.org/download/pdf_1/euclid.aos/1035844981
    - Kuipers et al. (2014): https://projecteuclid.org/download/suppdf_1/euclid.aos/1407420013

    The default arguments imply commonly-used default hyperparameters for mean and precision
    of the Normal-Wishart and assume a diagonal parameter matrix :math:`T`.
    Inspiration for the implementation was drawn from
    https://bitbucket.org/jamescussens/pygobnilp/src/master/pygobnilp/scoring.py

    This implementation uses properties of the determinant to make the computation of the marginal likelihood
    ``jax.jit``-compilable and ``jax.grad``-differentiable by remaining well-defined for soft relaxations of the graph.

    Args:
        n_vars (int): number of variables (nodes in the graph)
        mean_obs (ndarray, optional): mean parameter of Normal
        alpha_mu (float, optional): precision parameter of Normal
        alpha_lambd (float, optional): degrees of freedom parameter of Wishart

    """

    def __init__(self, *,
                 n_vars,
                 mean_obs=None,
                 alpha_mu=None,
                 alpha_lambd=None,
                 ):
        self.n_vars = n_vars

        self.mean_obs = mean_obs or jnp.zeros(self.n_vars)
        self.alpha_mu = alpha_mu or 1.0
        self.alpha_lambd = alpha_lambd or (self.n_vars + 2)

        assert self.alpha_lambd > self.n_vars + 1
        self.no_interv_targets = jnp.zeros(self.n_vars).astype(bool)

    def get_theta_shape(self, *, n_vars):
        raise NotImplementedError("Not available for BGe score; use `LinearGaussian` model instead.")

    def sample_parameters(self, *, key, n_vars, n_particles=0, batch_size=0):
        raise NotImplementedError("Not available for BGe score; use `LinearGaussian` model instead.")

    def sample_obs(self, *, key, n_samples, g, theta, toporder=None, interv=None):
        raise NotImplementedError("Not available for BGe score; use `LinearGaussian` model instead.")

    """
    The following functions need to be functionally pure and jax.jit-compilable
    """

    def _log_marginal_likelihood_single(self, j, n_parents, g, x, interv_targets):
        """
        Computes node-specific score of BGe marginal likelihood. ``jax.jit``-compilable

        Args:
            j (int): node index for score
            n_parents (int): number of parents of node ``j``
            g (ndarray): adjacency matrix of shape ``[d, d]
            x (ndarray): observations matrix of shape ``[N, d]``
            interv_targets (ndarray): intervention indicator matrix of shape ``[N, d]``

        Returns:
            BGe score for node ``j``
        """

        d = x.shape[-1]
        small_t = (self.alpha_mu * (self.alpha_lambd - d - 1)) / (self.alpha_mu + 1)
        T = small_t * jnp.eye(d)

        # mask rows of `x` where j is intervened upon to 0.0 and compute (remaining) number of observations `N`
        x = x * (1 - interv_targets[..., j, None])
        N = (1 - interv_targets[..., j]).sum()

        # covariance matrix of non-intervened rows
        x_bar = jnp.where(jnp.isclose(N, 0), jnp.zeros((1, d)), x.sum(axis=0, keepdims=True) / N)
        x_center = (x - x_bar) * (1 - interv_targets[..., j, None])
        s_N = x_center.T @ x_center  # [d, d]

        # Kuipers et al. (2014) state `R` wrongly in the paper, using `alpha_lambd` rather than `alpha_mu`
        # their supplementary contains the correct term
        R = T + s_N + ((N * self.alpha_mu) / (N + self.alpha_mu)) * \
            ((x_bar - self.mean_obs).T @ (x_bar - self.mean_obs))  # [d, d]

        parents = g[:, j]
        parents_and_j = (g + jnp.eye(d))[:, j]

        log_gamma_term = (
                0.5 * (jnp.log(self.alpha_mu) - jnp.log(N + self.alpha_mu))
                + gammaln(0.5 * (N + self.alpha_lambd - d + n_parents + 1))
                - gammaln(0.5 * (self.alpha_lambd - d + n_parents + 1))
                - 0.5 * N * jnp.log(jnp.pi)
                # log det(T_JJ)^(..) / det(T_II)^(..) for default T
                + 0.5 * (self.alpha_lambd - d + 2 * n_parents + 1) *
                jnp.log(small_t)
        )

        log_term_r = (
            # log det(R_II)^(..) / det(R_JJ)^(..)
                0.5 * (N + self.alpha_lambd - d + n_parents) *
                _slogdet_jax(R, parents)
                - 0.5 * (N + self.alpha_lambd - d + n_parents + 1) *
                _slogdet_jax(R, parents_and_j)
        )

        # return neutral sum element (0) if no observations (N=0)
        return jnp.where(jnp.isclose(N, 0), 0.0, log_gamma_term + log_term_r)

    def log_marginal_likelihood(self, *, g, x, interv_targets):
        """Computes BGe marginal likelihood :math:`\\log p(D | G)`` in closed-form;
        ``jax.jit``-compatible

        Args:
           g (ndarray): adjacency matrix of shape ``[d, d]``
           x (ndarray): observations of shape ``[N, d]``
           interv_targets (ndarray): boolean mask of shape ``[N, d]`` of whether or not
               a node was intervened upon in a given sample. Intervened nodes are ignored in likelihood computation

        Returns:
           BGe Score
        """
        # indices
        _, d = x.shape
        nodes_idx = jnp.arange(d)

        # number of parents for each node
        n_parents_all = g.sum(axis=0)

        # sum scores for all nodes [d,]
        scores = vmap(self._log_marginal_likelihood_single,
                      (0, 0, None, None, None), 0)(nodes_idx, n_parents_all, g, x, interv_targets)

        return scores.sum(0)

    """
    Distributions used by DiBS for inference:  prior and marginal likelihood 
    """

    def interventional_log_marginal_prob(self, g, _, x, interv_targets, rng):
        """Computes interventional marginal likelihood :math:`\\log p(D | G)`` in closed-form;
        ``jax.jit``-compatible

        To unify the function signatures for the marginal and joint inference classes
        :class:`~dibs.inference.MarginalDiBS` and :class:`~dibs.inference.JointDiBS`,
        this marginal likelihood is defined with dummy ``theta`` inputs as ``_``,
        i.e., like a joint likelihood

        Arguments:
            g (ndarray): graph adjacency matrix of shape ``[n_vars, n_vars]``.
                Entries must be binary and of type ``jnp.int32``
            _:
            x (ndarray): observational data of shape ``[n_observations, n_vars]``
            interv_targets (ndarray): indicator mask of interventions of shape ``[n_observations, n_vars]``
            rng (ndarray): rng; skeleton for minibatching (TBD)

        Returns:
            BGe score of shape ``[1,]``
        """
        return self.log_marginal_likelihood(g=g, x=x, interv_targets=interv_targets)
