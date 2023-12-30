import jax.numpy as jnp
import numpy as np
import optax
import jax
import wandb # 一种轻量级可视化工具，可跟踪实验，记录运行中的超参数和输出指标，可视化并共享结果
import os

from pathlib import Path
from tqdm import trange
from numpy.random import default_rng

from dag_gflownet.env import GFlowNetDAGEnv
from dag_gflownet.gflownet import DAGGFlowNet
from dag_gflownet.utils.replay_buffer import ReplayBuffer
from dag_gflownet.utils.factories import get_model, get_model_prior
from dag_gflownet.utils.gflownet import posterior_estimate
from dag_gflownet.utils.jraph_utils import to_graphs_tuple
from dag_gflownet.utils.data import load_artifact_continuous

# 自己加的-在wandb中创建该项目
# wandb.init(
#     project="jsp-gfn-fuxian",
#     config={
#     "learning_rate":0.02,
#     "architecture":"CNN",
#     "dataset":"CIFAR-100",
#     "epochs":10,   
#     }
# )

wandb.login(key = '6558815ff993312c6b39be2c636f38401fbeb6d3') # 登陆自己的wandb账号
def main(args):
    
    api = wandb.Api()  # 初始化wandb接口-自己的wandb账号链接，代码运行产生的结果会放到该账号下

    # 初始化PRNG key用于随机数生成   随机数的生成器，随机种子为args.seed
    key = jax.random.PRNGKey(args.seed)  # 创建一个伪随机数key，一个二维矩阵[0,args.seed]
    key, subkey = jax.random.split(key)  # 把key随机分成两部分,所得结果可作为随机数的生成种子

    # Get the artifact from wandb, 之后可以对artifact中的文件进行操作
    artifact = api.artifact(args.artifact)  # 创建artifact(作用类似于文件夹)，args.artifact=(名称,type=数据集/模型/结果)
    artifact_dir = Path(artifact.download()) / f'{args.seed:02d}'  # 从W&B下载artifact并创建一个文件夹路径，该路径包含特定的种子值
    # 保证arg.seed符合artifact的设置，从而确保用户提供的有效的种子值以保证可复现性
    if args.seed not in artifact.metadata['seeds']:  # metadata指与artifict相关的附加信息或性质，包括artifact的细节信息如创建时间、作者、种子信息等
        raise ValueError(f'The seed `{args.seed}` is not in the list of seeds '
            f'for artifact `{args.artifact}`: {artifact.metadata["seeds"]}')

    # Load data & graph
    train, valid, graph = load_artifact_continuous(artifact_dir)  
    train_jnp = jax.tree_util.tree_map(jnp.asarray, train) # 树形结构，每个节点数据都为jax array

    # Create the environment
    env = GFlowNetDAGEnv(
        num_envs=args.num_envs,  # 并行环境数
        num_variables=train.data.shape[1],  # 变量数-取决于训练数据
        max_parents=args.max_parents  # 最大父节点数
    )

    # Create the replay buffer
    replay = ReplayBuffer(
        args.replay_capacity,
        num_variables=env.num_variables,
    )

    # Create the model
    # 设置obs_cale，用于表征数据不确定性的一种度量
    if 'obs_noise' not in artifact.metadata['cpd_kwargs']:
        if args.obs_scale is None:
            raise ValueError('The obs_noise is not defined in the artifact, '
                'therefore is must be set as a command argument `--obs_scale`.')
        obs_scale = args.obs_scale
    else:
        obs_scale = artifact.metadata['cpd_kwargs']['obs_noise']
    # 获取图先验和cpd参数的后验分布模型
    prior_graph = get_model_prior(args.prior, artifact.metadata, args)
    model = get_model(args.model, prior_graph, train_jnp, obs_scale)

    # Create the GFlowNet & initialize parameters
    gflownet = DAGGFlowNet(
        model=model,
        delta=args.delta,
        num_samples=args.params_num_samples,  # 用于计算损失的模型参数的样本数
        update_target_every=args.update_target_every,  # 目标网络的更新频率（默认为0，表示无目标网络）
        dataset_size=train_jnp.data.shape[0],
        batch_size=args.batch_size_data,
    )

    optimizer = optax.adam(args.lr)  # 设置优化器adam
    params, state = gflownet.init(
        subkey,
        optimizer,
        replay.dummy['graph'],  # dummy-虚拟的，这里指的是replay中的虚拟数据-图：一个GraphsTuple对象-初始化的
        replay.dummy['mask']  # replay中的虚拟数据-掩码：一个ndarray对象，形状为(1,n_vars, n_vars)-初始化的全为零
    )

    # 构造一个线性探索时间表-探索率从初始值经特定步数到达终止值
    exploration_schedule = jax.jit(optax.linear_schedule(  # jax.jit-编译函数，加快函数的运行速度
        init_value=jnp.array(0.),
        end_value=jnp.array(1. - args.min_exploration),  # min_exploration默认为0.1
        transition_steps=args.num_iterations // 2,  # num_iterations默认为100000
        transition_begin=args.prefill,  # prefill默认为1000
    ))

    # Training loop
    indices = None
    observations = env.reset()  # 重置gflownet环境的状态 
    # observations = {
    #    'adjacency': np.zeros(shape, dtype=np.int_), # shape=(num_envs, num_variables, num_variables)
    #    'mask': 1 - self._closure_T,
    #    'num_edges': np.zeros((self.num_envs,), dtype=np.int_),
    #    'score': np.zeros((self.num_envs,), dtype=np.float_),
    #    'order': np.full(shape, -1, dtype=np.int_)
    # }

    normalization = jnp.array(train_jnp.data.shape[0])  # 数据集D中的样本量

    with trange(args.prefill + args.num_iterations, desc='Training') as pbar:
        for iteration in pbar:
            # Sample actions, execute them, and save transitions in the replay buffer
            # 更新探索率
            epsilon = exploration_schedule(iteration)  # 探索率随训练次数变化
            observations['graph'] = to_graphs_tuple(observations['adjacency'])  # 在observations中添加一个键值对，键为'graph'
            actions, key, logs = gflownet.act(params.online, key, observations, epsilon, normalization)
            next_observations, delta_scores, dones, _ = env.step(np.asarray(actions))
            indices = replay.add(
                observations,
                actions,
                logs['is_exploration'],
                next_observations,
                delta_scores,
                dones,
                prev_indices=indices
            )
            observations = next_observations

            if iteration >= args.prefill:
                # Update the parameters of the GFlowNet
                samples = replay.sample(batch_size=args.batch_size, rng=default_rng())
                params, state, logs = gflownet.step(params, state, samples, train_jnp, normalization)

                pbar.set_postfix(loss=f"{logs['loss']:.2f}")

    # Evaluate the posterior estimate
    posterior, logs = posterior_estimate(
        gflownet,
        params.online,
        env,
        key,
        train_jnp,
        num_samples=args.num_samples_posterior,
        desc='Sampling from posterior'
    )

  
if __name__ == '__main__':
    from argparse import ArgumentParser
    import json
    import math

    parser = ArgumentParser(description='JSP-GFN for Strucure Learning.')  # 创建ArgumentParser对象，用于传参

    # Environment-环境参数设置
    environment = parser.add_argument_group('Environment')
    environment.add_argument('--num_envs', type=int, default=8,  # num_envs
        help='Number of parallel environments (default: %(default)s)')
    environment.add_argument('--prior', type=str, default='uniform',  # prior
        choices=['uniform', 'erdos_renyi', 'edge', 'fair'],
        help='Prior over graphs (default: %(default)s)')
    environment.add_argument('--max_parents', type=int, default=None,  # max_parents
        help='Maximum number of parents')
    # 使用示例-在命令行输入：python xxx.py --num_envs 10 --prior erdos_renyi --max_parents 5

    # Data
    data = parser.add_argument_group('Data')
    data.add_argument('--artifact', type=str, required=True,  # artifact-输入数据存放在里面
        help='Path to the artifact for input data in Wandb')
    data.add_argument('--obs_scale', type=float, default=math.sqrt(0.1),  # obs_scale-用于表征数据不确定性的一种度量
        help='Scale of the observation noise (default: %(default)s)')

    # Model
    model = parser.add_argument_group('Model')
    model.add_argument('--model', type=str, default='lingauss_diag',  # model
        choices=['lingauss_diag', 'lingauss_full', 'mlp_gauss'],
        help='Type of model (default: %(default)s)')

    # Optimization
    optimization = parser.add_argument_group('Optimization')
    optimization.add_argument('--lr', type=float, default=1e-5,  # learning rate
        help='Learning rate (default: %(default)s)')
    optimization.add_argument('--delta', type=float, default=1.,  # delta
        help='Value of delta for Huber loss (default: %(default)s)')
    optimization.add_argument('--batch_size', type=int, default=32,  # batch_size
        help='Batch size (default: %(default)s)')
    optimization.add_argument('--num_iterations', type=int, default=100_000,  # num_iterations
        help='Number of iterations (default: %(default)s)')
    optimization.add_argument('--params_num_samples', type=int, default=1,  # params_num_samples
        help='Number of samples of model parameters to compute the loss (default: %(default)s)')   # update_target_every 
    optimization.add_argument('--update_target_every', type=int, default=0,
        help='Frequency of update for the target network (0 = no target network)')
    optimization.add_argument('--batch_size_data', type=int, default=None,  # batch_size_data
        help='Batch size for the data (default: %(default)s)')

    # Replay buffer
    replay = parser.add_argument_group('Replay Buffer')
    replay.add_argument('--replay_capacity', type=int, default=100_000,  # replay_capacity
        help='Capacity of the replay buffer (default: %(default)s)')
    replay.add_argument('--prefill', type=int, default=1000,  # prefill
        help='Number of iterations with a random policy to prefill '  # 这一行与下一行连成一句话，为prefill的描述
             'the replay buffer (default: %(default)s)')
    
    # Exploration
    exploration = parser.add_argument_group('Exploration')
    exploration.add_argument('--min_exploration', type=float, default=0.1,  # min_exploration
        help='Minimum value of epsilon-exploration (default: %(default)s)')
    
    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--num_samples_posterior', type=int, default=1000,  # num_samples_posterior
        help='Number of samples for the posterior estimate (default: %(default)s)')
    misc.add_argument('--seed', type=int, default=0,  # seed
        help='Random seed (default: %(default)s)')

    args = parser.parse_args() # 终端命令行输入的参数通过该命令传到main函数中

    main(args) 
    