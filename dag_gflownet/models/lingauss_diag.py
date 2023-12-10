import jax.numpy as jnp
import haiku as hk
import jax
import math

from functools import partial

from dag_gflownet.models.base import LinearGaussianModel, NormalParams, _LOG_2PI
from dag_gflownet.models.priors import UniformPrior


class LingaussDiagModel(LinearGaussianModel):
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
        self.hidden_sizes = tuple(hidden_sizes)  # hidden_sizes=(128,)

    # 计算cpd参数后验分布的参数
    def posterior_parameters(self, features, adjacencies):
        # 重复全局特征
        global_features = jnp.repeat(features.globals[:, None], self.num_variables, axis=1)
        # 节点特征和重复的全局特征沿着最后一个维度拼接起来，形成MLP的输入
        inputs = jnp.concatenate((features.nodes, global_features), axis=2)
        # 用haiku构造神经网络
        # MLP层
        params = hk.nets.MLP(
            self.hidden_sizes,   # (128,)-MLP只有一个隐藏层，包含128个神经元
            activate_final=True, # 最后一层需要应用激活函数，中间隐藏层默认为Relu激活，最后一层为线性激活
            name='edges'         # 网络名称
        )(inputs)
        # haiku线性层
        params = hk.Linear(
            2 * self.num_variables,               # 输出维度
            w_init=hk.initializers.Constant(0.),  # 初始化网络权重
            b_init=hk.initializers.Constant(0.)   # 初始化网络偏置
        )(params)
        params = params.reshape(-1, self.num_variables, 2 * self.num_variables)  # 将参数reshape成(128, 10, 20)，其中第一个维度是自动计算的

        # Split the parameters into loc and scale
        locs, scales = jnp.split(params, 2, axis=-1)

        # Mask the parameters, based on the adjacencies
        masks = jnp.swapaxes(adjacencies, -1, -2)
        return NormalParams(
            loc=locs * masks,
            scale=jax.nn.softplus(scales * masks)  # 确保scale为正
        )

    def log_prob(self, theta, masked_dist):
        diff = (theta - masked_dist.dist.loc) / masked_dist.dist.scale  # 计算标准化差异
        return -0.5 * jnp.sum(masked_dist.mask * (diff ** 2 + _LOG_2PI
            + 2 * jnp.log(masked_dist.dist.scale)), axis=-1)

    def sample(self, key, masked_dists, num_samples):
        shape = masked_dists.mask.shape
        epsilons = jax.random.normal(key, shape=(num_samples,) + shape) # 确保所生成随机数的形状与mask的形状匹配

        @partial(jax.vmap, in_axes=(1, 0))
        def _sample(epsilon, masked_dist):
            # Sample from a Normal distribution
            samples = masked_dist.dist.loc + epsilon * masked_dist.dist.scale
            return samples * masked_dist.mask  # Mask the samples

        return _sample(epsilons, masked_dists)
