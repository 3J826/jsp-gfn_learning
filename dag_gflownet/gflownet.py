import jax.numpy as jnp
import optax
import jax

from functools import partial
from collections import namedtuple
from jax import grad, random, jit, tree_util, lax

from dag_gflownet.nets.gnn.gflownet import gflownet
from dag_gflownet.utils.gflownet import uniform_log_policy, sub_trajectory_balance_loss
from dag_gflownet.utils.jnp_utils import batch_random_choice


DAGGFlowNetParameters = namedtuple('DAGGFlowNetParameters', ['online', 'target'])
DAGGFlowNetState = namedtuple('DAGGFlowNetState', ['optimizer', 'key', 'steps'])
GFNState = namedtuple('GFNState', ['thetas', 'log_pi', 'log_p_theta', 'scores', 'diffs'])


class DAGGFlowNet:
    def __init__(self, model, network=None, delta=1., num_samples=1, update_target_every=0, dataset_size=1, batch_size=None):
        if network is None:
            network = gflownet(model.masked_posterior_parameters)

        self.model = model  # cpd参数的后验分布模型如lingauss模型
        self.network = network
        self.delta = delta
        self.num_samples = num_samples
        self.update_target_every = update_target_every
        self.dataset_size = dataset_size
        self.batch_size = batch_size

        self._optimizer = None  # _optimizer是一个私有变量，只能在类内部访问

    def get_state(self, params, key, graphs, masks, dataset, num_samples, norm1, norm2):
        """params: gflownet网络的online参数
           graphs: 来自replaybuffer里32个经验数据的图(共8*32张图), 为GraphsTuple对象
           masks: 来自replaybuffer里32个经验数据的掩码(共8*32个掩码)
           dataset: batch,即数据集D
           num_samples: 1
           norm1: |D|
           norm2: 1
        """
        
        # Compute the forward transition probability
        # 输入params, graphs, masks, norm1到gflownet里，获得网络输出
        # 其中log_pi为前向转移策略，dist为theta后验分布近似模型的参数，代表后验分布本身
        log_pi, dist = self.network.apply(params, graphs, masks, norm1)  

        # Sample the parameters of the model
        # 从theta后验分布近似模型中采样一个theta, 形状为(num_samples, batch_size, num_variables, num_variables)                                                                                                                                   
        thetas = self.model.sample(key, dist, num_samples)  # num_samples=1

        # 
        def diff_grads(theta, dist, dataset):
            
            # log_joint是BaseModel的抽象方法, 用于计算log P(D,G,theta)
            score = self.model.log_joint(dist.mask, theta, dataset, norm2)  

            log_p_theta = self.model.log_prob(theta, dist)
            log_p_theta = jnp.sum(log_p_theta)  # Sum over variables, 形状变为(num_samples, batch_size)

            return (score - log_p_theta, score)

        v_diff_grads = jax.grad(diff_grads, has_aux=True)  #  diff_grads的梯度函数
        v_diff_grads = jax.vmap(v_diff_grads, in_axes=(0, None, None))  # vmapping over thetas
        v_diff_grads = jax.vmap(v_diff_grads, in_axes=(0, 0, None))  # vmapping over batch
        diffs, scores = v_diff_grads(thetas, dist, dataset)  # diffs为损失函数关于theta的梯度，scores为损失函数关于G的梯度

        # Compute the log-probabilities of the parameters. Use stop-gradient
        # to incorporate information about sub-trajectories of length 2.
        v_log_prob = jax.vmap(self.model.log_prob, in_axes=(0, None))  # vmapping over thetas
        v_log_prob = jax.vmap(v_log_prob, in_axes=(0, 0))  # vmapping over batch
        log_p_theta = v_log_prob(lax.stop_gradient(thetas), dist)  # 在梯度计算中将theta视作常数，不对其计算梯度

        return GFNState(
            thetas=thetas,  # (num_samples, batch_size, num_variables, num_variables)
            log_pi=log_pi,
            log_p_theta=jnp.sum(log_p_theta, axis=2),  # Sum the log-probabilities over the variables
            scores=scores,
            diffs=jnp.sum(diffs ** 2, axis=(1, 2, 3)) / (
                self.num_samples * jnp.sum(dist.mask, axis=(1, 2))),  # 平均差异
        )

    # 一个batch中的平均子轨迹平衡损失
    def loss(self, params, target_params, key, samples, dataset, normalization):
        """params: gflownet的online参数
           target_params: gflownet的target参数
           samples: 来自replaybuffer的32条经验数据, 每条数据代表gflownet环境在某一步的状态, 包括图的邻接矩阵等
           dataset: 数据集D
           normalization: 归一化参数, 即数据集D的大小
        """
        
        # 定义损失函数的计算公式
        @partial(jax.vmap, in_axes=0)  # 修饰器，将_loss函数的第一个参数向量化
        def _loss(state_t, state_tp1, actions, num_edges):
            # Compute the delta-scores. Use stop-gradient to incorporate
            # information about sub-trajectories of length 2. This comes
            # from the derivative of log P_F(theta | G) wrt. theta is the
            # same as the derivative of log R(G, theta) wrt. theta.
            
            # 计算下一状态与当前状态之间的得分差(scores=log P(D,G,theta) = log R(G,theta)))
            delta_scores = state_tp1.scores[:, None] - state_t.scores 
            # 在梯度计算中将delta_scores视作常数，不对其计算梯度，等号左边的值可以理解为被“冻住”的值
            delta_scores = lax.stop_gradient(delta_scores)  

            # Compute the sub-trajectory balance loss over a batch(32个) of samples 
            loss, logs = sub_trajectory_balance_loss(
                state_t.log_pi, state_tp1.log_pi,  # 当前状态和下一状态的前向转移概率(对应前向行动为加边)
                state_t.log_p_theta, state_tp1.log_p_theta,  # 当前状态和下一状态的前向转移概率(对应前向行动为采样theta)
                actions,  # s_t转移到s_{t+1}的行动
                delta_scores,  # log R(s_{t+1}) - log R(s_t)
                num_edges,  # 当前状态s_t的边数
                normalization=self.dataset_size,
                delta=self.delta
            )

            # Add penalty for sub-trajectories of length 2 (in differential form)
            loss = loss + 0.5 * (state_t.diffs + state_tp1.diffs)

            return (loss, logs)

        subkey1, subkey2, subkey3 = random.split(key, 3)

        # Sample a batch of data
        if self.batch_size is not None:  # batch_size默认为None
            indices = jax.random.choice(subkey1, self.dataset_size,  # self.dataset_size=1
                shape=(self.batch_size,), replace=False)
            batch = jax.tree_util.tree_map(lambda x: x[indices], dataset)
        else:
            batch = dataset  # 数据集D
        
        norm = (normalization / self.batch_size  
            if self.batch_size is not None else jnp.array(1.))  # norm=1

        # Compute the states
        state_t = self.get_state(params, subkey2,  # gflownet的当前环境状态
            samples['graph'], samples['mask'], batch, self.num_samples, normalization, norm)
        if self.update_target_every == 0:  # 该参数默认为0 
            target_params = params
        state_tp1 = self.get_state(target_params, subkey3,  # 执行行动后得到的gflownet的环境状态
            samples['next_graph'], samples['next_mask'], batch, self.num_samples, normalization, norm)

        outputs = _loss(state_t, state_tp1,  # outputs为损失和日志
            samples['actions'], samples['num_edges'])
        
        # 对batch(32个)中并行环境下的损失和日志求均值
        loss, logs = tree_util.tree_map(partial(jnp.mean, axis=0), outputs)
        logs.update({
            'post_theta/log_prob': state_t.log_p_theta,
            'error': outputs[1]['error'],  # Leave "error" unchanged
        })
        return (loss, logs)

    @partial(jit, static_argnums=(0,))  # 函数修饰器，将jit函数部分应用于act函数，加快act函数的运行速度；static_argnums=(0,)表示act的第一个参数params是静态参数，其值改变后不用重新编译
    def act(self, params, key, observations, epsilon, normalization):
        """params为gflownet网络参数, 包括online参数和target参数"""
        masks = observations['mask'].astype(jnp.float32)  # 将mask中的元素变为float32类型.注意这里的mask是3D的，(num_envs,1,1),第一维度储存每个环境下的mask矩阵
        graphs = observations['graph']   # graphstuple对象
        batch_size = masks.shape[0]  # num_envs
        key, subkey1, subkey2 = random.split(key, 3)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        # Get the GFlowNet policy
        # 输入params, graphs, masks, normalization到gflownet里，获得网络输出
        # 输出为(log_policy(logits * normalization, stop * normalization, masks), post_parameters)
        # 输出表示当前所有环境下，为图继续加边和停止加边的概率的对数
        log_pi, _ = self.network.apply(params, graphs, masks, normalization)  # (8, 26)

        # Get uniform policy
        log_uniform = uniform_log_policy(masks)  # (8, 26)

        # Mixture of GFlowNet policy and uniform policy
        is_exploration = random.bernoulli(  # 随机获取每个环境下是否探索的标志-1表示探索，0表示不探索
            subkey1, p=1. - epsilon, shape=(batch_size, 1))  # batch_size=num_envs，采样到1的概率为1. - epsilon
        
        # 将 要探索的环境下的策略换成uniform的策略，否则采用gflownet策略
        log_pi = jnp.where(is_exploration, log_uniform, log_pi)  # 若某环境下的is_exploration为1，则返回log_uniform，否则返回log_pi

        # Sample actions
        # log_pi的形状为(batch_size, num_vars*num_vars)
        # actions中每行的元素为一个取值于[0,25]的整数，表示在该环境下采用的策略(继续加边或停止加边)
        actions = batch_random_choice(subkey2, jnp.exp(log_pi), masks)  # (batch_size, )

        # 探索日志
        logs = {
            'is_exploration': is_exploration.astype(jnp.int32),
        }
        return (actions, key, logs)

    @partial(jit, static_argnums=(0, 5))
    def act_and_params(self, params, key, observations, dataset, num_samples):
        masks = observations['mask'].astype(jnp.float32)
        graphs = observations['graph']
        key, subkey1, subkey2 = random.split(key, 3)

        normalization = jnp.array(dataset.data.shape[0])
        norm = jnp.array(1.)
        state = self.get_state(params, subkey1, graphs, masks, dataset, num_samples, normalization, norm)

        # Sample actions
        actions = batch_random_choice(subkey2, jnp.exp(state.log_pi), masks)

        logs = {}
        return (actions, key, state, logs)

    @partial(jit, static_argnums=(0,))  # 修饰器，加快step函数的运行速度；static_argnums=(0,)表示step的第一个输入参数params是静态参数
    def step(self, params, state, samples, dataset, normalization):  # params为网络参数，state为网络训练状态，samples为采样数据，dataset为训练数据集，normalization为归一化参数
        """state-gflownnet网络训练状态, 包括optimizer, key, steps
           samples-来自replaybuffer的经验数据,共32条
           dataset-数据集D
           normalization-正则化参数, 即数据集D的大小
        """
        key, subkey = random.split(state.key)
        
        # 计算损失函数关于网络参数的梯度。has_aux=True表示损失函数返回值中包含辅助数据（除梯度外）
        grads, logs = grad(self.loss, has_aux=True)(params.online, params.target, subkey, samples, dataset, normalization)

        # Update the online params
        # 优化器本身也会得到更新(包括优化步长和方向等)
        updates, opt_state = self.optimizer.update(grads, state.optimizer, params.online)  #  根据grads用update函数更新online参数
        # 更新gflownet的训练状态
        state = DAGGFlowNetState(optimizer=opt_state, key=key, steps=state.steps + 1)
        
        # 更新online参数
        online_params = optax.apply_updates(params.online, updates) 
        
        # 更新target参数
        if self.update_target_every > 0:  # 若目标参数更新频率大于0
            target_params = optax.periodic_update(  # 则按照频率定期更新
                online_params,
                params.target,
                state.steps,
                self.update_target_every
            )
        else:
            target_params = params.target  # 否则不更新
            
        # 归纳更新后的gflownet参数
        params = DAGGFlowNetParameters(online=online_params, target=target_params)

        return (params, state, logs)

    def init(self, key, optimizer, graph, mask):
        key, subkey = random.split(key)
        # Set the optimizer-optax.zero_nans()是一个转换器，能将梯度中的nan转换为0,从而避免计算错误
        self._optimizer = optax.chain(optimizer, optax.zero_nans())
        # 初始化gflownet网络参数params
        online_params = self.network.init(subkey, graph, mask, jnp.array(1.))  # subkey为随机密钥；graph为图，mask为掩码，jnp.array(1.)为归一化参数，三者均为初始化过程中网络所需的参数
        target_params = online_params if (self.update_target_every > 0) else None
        params = DAGGFlowNetParameters(online=online_params, target=target_params)
        # 初始化gflownet网络训练状态state
        state = DAGGFlowNetState(
            optimizer=self.optimizer.init(online_params),
            key=key,
            steps=jnp.array(0),  # 已完成的迭代步数
        )
        return (params, state)  # 返回DAGGFlowNet网络参数与训练状态

    @property
    def optimizer(self):  # 定义一个optimizer属性
        if self._optimizer is None:
            raise RuntimeError('The optimizer is not defined. To train the '
                               'GFlowNet, you must call `DAGGFlowNet.init` first.')
        return self._optimizer
