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
            num_variables,  # 变量个数-图中节点数
            hidden_sizes,  # (128,)
            prior=NormalParams(loc=0., scale=1.),  # 命名元组
            obs_scale=math.sqrt(0.1),  # 噪声
            prior_graph=UniformPrior()  # LingaussDiag模型中使用均匀先验P(G)
        ):
        super().__init__(  # 初始化父类的父类-GaussianModel
            num_variables,
            prior=prior,
            obs_scale=obs_scale,
            prior_graph=prior_graph
        )
        self.hidden_sizes = tuple(hidden_sizes)  # 将hidden_sizes转化为元组形式，但hidden_sizes=(128,)已经是一个元组形式

    # 计算theta后验分布模型的参数——均值，方差
    def posterior_parameters(self, features, adjacencies):  # features是G中节点X经GNN+self-attention嵌入和的特征
       
        """features: GraphsTuple, 包含节点、边、全局特征。
               节点特征形状为(batch_size, num_variables, feature_size_nodes)
               全局特征形状为(batch_size, feature_size_globals)
               
           adjacencies: 图G的邻接矩阵(batch_size, n_vars, n_vars)
           
           Returns: Batch中每张图G的theta后验分布近似模型的参数, 包括均值和方差
               均值形状为(batch_size, num_variables, num_variables), 其中第一个维度表示哪张图G, 第二个维度表示图G中哪个节点X_i, 第三个维度表示该节点cpd参数theta_i的均值向量
               (LingaussFullModel)方差形状为(batch_size, num_variables, num_variables, nun_variables), 其中第一个维度表示哪张图G, 第二个维度表示图G中哪个节点X_i, 第三个和四个维度表示该节点cpd参数theta_i的协方差矩阵   
               (LingaussDiagModel)方差形状为(batch_size, num_variables, num_variables), 其中第一个维度表示哪张图G, 第二个维度表示图G中哪个节点X_i, 第三个维度表示协方差矩阵对角线元素
        """
        
        # 获得G中所有节点的全局特征,形状为(batch_size, num_vars, feature_size_globals)
        # features.globals[:, None]表示沿第二个维度为features.globals增加一个维度
        # 每个节点的全局特征相同（因为是重复features.globals[:, None]）得到
        global_features = jnp.repeat(features.globals[:, None], self.num_variables, axis=1)
        
        # G中所有的节点特征和全局特征沿着最后一个维度拼接起来，形成MLP的输入
        # inputs形状为(batch_size, num_variables, feature_size_nodes+feature_size_globals)
        inputs = jnp.concatenate((features.nodes, global_features), axis=2)
        
        # 用haiku构造神经网络
        # MLP层
        params = hk.nets.MLP(    # 创建一个MLP
            self.hidden_sizes,   # (128,)-MLP只有一个包含128个神经元的隐藏层
            activate_final=True, # 最后一层需要应用激活函数，中间隐藏层默认为Relu激活，最后一层默认为线性激活
            name='edges'         # 网络名称
        )(inputs)                # 输入inputs到构建的MLP，得到的输出即为params
        
        # haiku线性层
        params = hk.Linear(                       # 创建一个线性层
            2 * self.num_variables,               # 输出维度-2d——相当于把输入的feature_size_nodes+feature_size_globals变为2d
            w_init=hk.initializers.Constant(0.),  # 初始化网络权重为0
            b_init=hk.initializers.Constant(0.)   # 初始化网络偏置为0
        )(params)                                 # 将MLP得到的参数输入该网络，对其进行一个线性变换得到新的params
        params = params.reshape(-1, self.num_variables, 2 * self.num_variables)  

        # Split the parameters into loc and scale
        locs, scales = jnp.split(params, 2, axis=-1)  # locs, scales的形状都为(batch, n_vars, n_vars)

        # Mask the parameters, based on the adjacencies
        # 去掉locs和scales中在G中已有的以及会成环的节点
        masks = jnp.swapaxes(adjacencies, -1, -2)  # 交换adjacencies矩阵的最后两个维度，即交换行和列
        return NormalParams(
            loc=locs * masks,  # 经掩码过滤的均值
            scale=jax.nn.softplus(scales * masks)  # 经掩码过滤的方差, jax.nn.softplus确保scale为正
        )

    # 这一步算的什么？？
    def log_prob(self, theta, masked_dist):  # 继承GaussianModel的抽象方法
        
        """theta形状为(num_samples, batch_size, num_vars, num_vars), 其中num_samples=1
           Return: 形状为(num_samples, batch_size, num_vars)
        """
        
        # 对theta做标准化-标准化后均值为0方差为1
        # dist.loc与dist.scale形状都为(batch_size, num_vars, num_vars)
        diff = (theta - masked_dist.dist.loc) / masked_dist.dist.scale  
        
        return -0.5 * jnp.sum(masked_dist.mask * (diff ** 2 + _LOG_2PI
            + 2 * jnp.log(masked_dist.dist.scale)), axis=-1)  

    def sample(self, key, masked_dists, num_samples):  # 继承BaseModel的抽象方法
        """从theta的掩码分布中采样theta

        Args:
            key (jax.random.PRNGKey): The random key to sample the parameters \theta
            masked_dists (_type_): `MaskedDistribution` instance(MaskedDistribution为命名元组,包含dists和masks两个属性)
                theta的后验分布近似模型的参数, 代表后验分布近似模型本身
            num_samples (int): 每张图G采样的theta个数

        Returns:
            来自theta后验分布近似模型的样本thetas. 
            形状为(num_samples, batch_size, num_vars, num_vars)
        """
        shape = masked_dists.mask.shape  # (batch_size, n_vars, n_vars)
        epsilons = jax.random.normal(key, shape=(num_samples,) + shape) # (num_samples, batch_size, num_vars, num_vars)

        @partial(jax.vmap, in_axes=(1, 0))  # 对_sample的第一维度输入进行向量化
        def _sample(epsilon, masked_dist):
            
            # Sample from a Normal distribution
            # lingaussdiag下dist.loc与dist.scale形状都为(batch_size, num_vars, num_vars)
            samples = masked_dist.dist.loc + epsilon * masked_dist.dist.scale  
            return samples * masked_dist.mask  # Mask the samples, 形状为(num_samples, batch_size, num_vars, num_vars)

        return _sample(epsilons, masked_dists)
