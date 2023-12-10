import jax.numpy as jnp
import haiku as hk
import jax
import math

from collections import namedtuple
from functools import partial
from jax.scipy import linalg

from dag_gflownet.models.base import LinearGaussianModel, NormalParams, _LOG_2PI
from dag_gflownet.models.priors import UniformPrior

NormalFullParams = namedtuple('NormalFullParams', ['loc', 'precision_triu'])

def to_masked_triu(values, mask, prior_scale):
    num_variables = mask.shape[0]
    rows, cols = jnp.triu_indices(num_variables, k=1)  # 从次对角线开始，获取d=num_variables维矩阵上三角部分元素的行列指标

    # Split the values into diagonal & off-diagonal
    diag, non_diag = values[:num_variables], values[num_variables:]  # num_vars个对角线元素

    # Mask the diagonal element (masked elements have scale given by prior)
    # Diagonal elements have to be positive for the matrix to be PD
    diag = jnp.where(mask, jax.nn.softplus(diag), prior_scale)

    # Create off-diagonal matrix
    triu = jnp.zeros((num_variables, num_variables), dtype=values.dtype)
    triu = triu.at[rows, cols].set(mask[rows] * mask[cols] * non_diag)

    # Add the diagonal elements
    return triu + jnp.diag(diag)


def mvn_sample(key, params, num_samples):
    epsilon = jax.random.normal(key, shape=params.loc.shape + (num_samples,)) # shape:(params.loc.shape, num_samples)
    # 求解线性方程组params.precision_triu * X = epsilon，lower=False表示params.precision_triu为上三角矩阵，否则为下三角矩阵
    lhs = linalg.solve_triangular(params.precision_triu, epsilon, lower=False)
    samples = params.loc + jnp.rollaxis(lhs, -1)
    return jnp.swapaxes(samples, 0, 1)  # 交换samples的第一维和第二维


def mvn_log_prob(params, samples, mask):
    diff = jnp.expand_dims((samples - params.loc) * mask, axis=-2)
    normalized = jnp.sum(params.precision_triu * diff, axis=-1)

    # Log-determinant
    diags = jnp.diagonal(params.precision_triu, axis1=-2, axis2=-1)
    logdet = -2. * jnp.sum(jnp.log(diags), axis=-1)

    # Normalization
    norm = jnp.sum(mask, axis=-1) * _LOG_2PI

    return -0.5 * (norm + logdet + jnp.sum(normalized ** 2, axis=-1))


class LingaussFullModel(LinearGaussianModel):
    def __init__(
            self,
            num_variables,
            hidden_sizes,
            prior=NormalParams(loc=0., scale=1.),
            obs_scale=math.sqrt(0.1),
            prior_graph=UniformPrior()
        ):
        super().__init__(
            num_variables,
            prior=prior,
            obs_scale=obs_scale,
            prior_graph=prior_graph
        )
        self.hidden_sizes = tuple(hidden_sizes)

    # 返回cpd参数后验估计模型的参数（这里采用LingaussFullModel作为后验估计模型，返回其均值和协方差矩阵）
    def posterior_parameters(self, features, adjacencies):
        num_values = (self.num_variables
            + self.num_variables * (self.num_variables + 1) // 2)  # 是啥？计算参数所需值的总个数，是什么值？
        hidden_sizes = self.hidden_sizes + (num_values,)
        params = hk.nets.MLP(hidden_sizes, name='edges')(features.nodes) # 用MLP产生的cpd参数
        params = params.reshape(-1, self.num_variables, num_values)

        @partial(jax.vmap, in_axes=(0, 1))
        def _parameters(values, mask):
            loc = jnp.where(mask, values[:self.num_variables], 0.)
            precision_triu = to_masked_triu(
                values[self.num_variables:], mask, self.prior.scale)
            return NormalFullParams(loc=loc, precision_triu=precision_triu)

        return jax.vmap(_parameters)(params, adjacencies)

    # 根据cpd参数后验分布模型，计算后验概率 log P(theta|G)
    def log_prob(self, theta, masked_dist):
        return mvn_log_prob(masked_dist.dist, theta, masked_dist.mask)

    def sample(self, key, masked_dists, num_samples):
        thetas = mvn_sample(key, masked_dists.dist, num_samples)
        return thetas * jnp.expand_dims(masked_dists.mask, axis=1)

