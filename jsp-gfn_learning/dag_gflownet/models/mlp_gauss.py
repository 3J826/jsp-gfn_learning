import jax.numpy as jnp
import jax
import haiku as hk
import math

from jax.flatten_util import ravel_pytree

from dag_gflownet.models.base import BaseModel, GaussianModel, NormalParams, _LOG_2PI
from dag_gflownet.utils.haiku_utils import create_mlp, get_first_weights
from dag_gflownet.models.priors import UniformPrior


class MLPModel(BaseModel):
    def __init__(
            self,
            num_variables,
            hidden_sizes,  # (128,)-只有一层
            model,               # mlp
            mask_filter_fn=None,
            min_scale=0.
        ):
        if mask_filter_fn is None:
            mask_filter_fn = get_first_weights  # 从mlp模型中提取第一个线性层的权重

        self.hidden_sizes = tuple(hidden_sizes)
        self._model = model
        self.min_scale = min_scale

        # Get the number of parameters and the function to unflatten parameters
        # from a 1D vector.
        key = jax.random.PRNGKey(0)  # Dummy random key
        inputs = jnp.zeros((num_variables,))
        params = model.init(key, inputs)  # mlp模型的初始化参数

        # 扁平化初始化参数维1维矩阵params, 并返回一个可调用函数_unflatten_fn，用于恢复原始的层次结构
        params, self._unflatten_fn = ravel_pytree(params)
        self._num_parameters = params.size

        # Find the indices for masking the parameters, depending on the adjacency matrix.
        indices = jnp.arange(self._num_parameters) # [0,1,2,...,num_parameters-1]
        self._mask_indices = mask_filter_fn(self._unflatten_fn(indices))

    def posterior_parameters(self, features, adjacencies):
        # 在features.globals里沿第二个维度将元素重复nun_vars次。features.globals[:, None]表示在features.globals第一个维度后添加一个维度
        global_features = jnp.repeat(features.globals[:, None], self.num_variables, axis=1)
        inputs = jnp.concatenate((features.nodes, global_features), axis=2) # 沿第三个维度拼接features.nodes和global_features
        # 用mlp输出cpd参数后验分布模型的参数
        params = hk.nets.MLP(
            self.hidden_sizes,
            activate_final=True,
            name='edges'
        )(inputs)
        # 对参数做了一次线性层变换
        params = hk.Linear(
            2 * self._num_parameters,  # output_size
            w_init=hk.initializers.Constant(0.),
            b_init=hk.initializers.Constant(0.)
        )(params)
        params = params.reshape(-1, self.num_variables, 2 * self._num_parameters)

        # Split the parameters into loc and scale
        locs, scales = jnp.split(params, 2, axis=-1)
        return NormalParams(loc=locs, scale=self.min_scale + jax.nn.softplus(scales))

    def mask_parameters(self, adjacencies):
        def _mask_parameters(parents):
            ones = jnp.ones((self._num_parameters,), dtype=parents.dtype)
            return ones.at[self._mask_indices].set(parents[:, None])

        v_mask_parameters = jax.vmap(_mask_parameters, in_axes=1)  # vmapping over variables
        v_mask_parameters = jax.vmap(v_mask_parameters, in_axes=0)  # vmapping over batch
        return v_mask_parameters(adjacencies)

    def model(self, theta, data, mask):
        # Unflatten theta into a pytree of parameters for the model
        params = self._unflatten_fn(theta * mask)
        return self._model.apply(params, data)

    def log_prob(self, theta, masked_dist):
        diff = (theta - masked_dist.dist.loc) / masked_dist.dist.scale
        return -0.5 * jnp.sum(masked_dist.mask * (diff ** 2 + _LOG_2PI
            + 2 * jnp.log(masked_dist.dist.scale)), axis=-1)

    def sample(self, key, masked_dists, num_samples):
        # Sample from a standard Normal distribution
        shape = masked_dists.dist.loc.shape
        epsilons = jax.random.normal(key, shape=(num_samples,) + shape)

        # Sample from a Normal distribution
        samples = masked_dists.dist.loc + masked_dists.dist.scale * epsilons
        samples = samples * masked_dists.mask  # Mask the samples
        samples = jnp.swapaxes(samples, 0, 1)

        return samples


class MLPGaussianModel(MLPModel, GaussianModel):
    def __init__(
            self,
            num_variables,
            hidden_sizes,
            model=None,
            prior=NormalParams(loc=0., scale=1.),
            obs_scale=math.sqrt(0.1),
            prior_graph=UniformPrior(),
            mask_filter_fn=None,
            min_scale=0.
        ):
        if model is None:
            # Following the architecture from DiBS
            # https://arxiv.org/pdf/2105.11839.pdf (Section 6.3)
            model = create_mlp((5, 1), activation='relu')

        MLPModel.__init__(
            self,
            num_variables,
            hidden_sizes,
            model,
            mask_filter_fn=mask_filter_fn,
            min_scale=min_scale
        )
        GaussianModel.__init__(
            self,
            num_variables,
            prior=prior,
            obs_scale=obs_scale,
            prior_graph=prior_graph
        )
