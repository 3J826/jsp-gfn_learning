from dag_gflownet.models import LingaussDiagModel, LingaussFullModel, MLPGaussianModel
from dag_gflownet.models import priors as model_priors


def get_model_prior(name, metadata, args):
    if name == 'uniform':
        prior = model_priors.UniformPrior()  # 返回未标准化的先验概率的对数log P(G)=0，即P(G)=1
    elif name == 'erdos_renyi':
        prior = model_priors.EdgesPrior(
            num_variables=metadata['num_variables'],
            num_edges_per_node=metadata['num_edges_per_node']
        )
    else:
        raise NotImplementedError(f'Unknown prior: {name}')

    return prior

# cpd参数后验分布模型
def get_model(name, prior_graph, dataset, obs_scale):
    """获取theta的后验分布近似模型,如 LingaussDiagModel.

    Args:
        name (str): 模型名称
        prior_graph (jnp.array): log P(G)
        dataset (ndarray): N * d维的观察数据矩阵-D
        obs_scale (float): 向后验模型里添加的噪声

    Returns:
        model (haiku.transform): 后验分布近似模型(一个类)
    """
    num_variables  = dataset.data.shape[1]  # 变量个数-图中节点数

    if name == 'lingauss_diag':
        model = LingaussDiagModel(  # 多元正态分布模型，均值和方差都被神经网络参数化
            num_variables=num_variables,
            hidden_sizes=(128,),
            obs_scale=obs_scale,
            prior_graph=prior_graph
        )
    elif name == 'lingauss_full':
        model = LingaussFullModel(
            num_variables=num_variables,
            hidden_sizes=(128,),
            obs_scale=obs_scale,
            prior_graph=prior_graph
        )
    elif name == 'mlp_gauss':
        model = MLPGaussianModel(
            num_variables=num_variables,
            hidden_sizes=(128,),
            obs_scale=obs_scale,
            prior_graph=prior_graph,
            min_scale=0.
        )
    else:
        raise NotImplementedError(f'Unknown model: {name}')

    return model
