import jax.numpy as jnp
import haiku as hk
import jraph
import math

from jax import lax, nn
from collections import namedtuple

from dag_gflownet.utils.gflownet import log_policy
from dag_gflownet.utils.jraph_utils import edge_features_to_dense


def gflownet(posterior_parameters_model):
    @hk.without_apply_rng
    # 将gflownet转换为haiku可识别版本，意味着该函数可用于定义一个具有可学习参数的神经网络模块
    @hk.transform  # 一种haiku修饰器，本质上是对函数进行包装，使其使用方式符合haiku的模块化和可组合原则
    
    # gflownet网络模块 
    def _gflownet(graphs, masks, normalization):  # normalization为数据量N
        
        # 获取batch_size(=num_envs)和num_variables
        batch_size, num_variables = masks.shape[:2]
       
        # 用1初始化边掩码-将graphs中edges的元素全部置为1
        # 一维向量，每个分量为1，长度为当前环境下所有图的总边数
        edge_masks = jnp.ones(graphs.edges.shape, dtype=jnp.float32)

        # Embedding of the nodes & edges
        # 每个节点被嵌入为一个128维的向量
        node_embeddings = hk.Embed(num_variables, embed_dim=128)  # hk.Embed会创建一个参数化的嵌入层，用于节点嵌入
        # 边嵌入就是'edge_embed'参数
        edge_embedding = hk.get_parameter('edge_embed', shape=(1, 128),  # 获取可学习参数'edge_embed'，其形状为(1,128)
            init=hk.initializers.TruncatedNormal())  # 用截断正态分布初始化参数'edge_embed'

        # 将graphs中的节点、边、全局特征都换成embedding形式，以便后续的GNN处理
        graphs = graphs._replace(
            nodes=node_embeddings(graphs.nodes),  # graphs.nodes=(0,1,2,3,4, 0,1,2,3,4, 0,1,2,3,4,...)
            edges=jnp.repeat(edge_embedding, graphs.edges.shape[0], axis=0),  # graphs.edges.shape[0]为所有环境下图的总边数
            globals=jnp.zeros((graphs.n_node.shape[0], 1)),  # 形状为(num_graphs, 1)的零矩阵，一张图对应一个全局特征
        )
        
        # Define graph network updates
        # 用带2个隐藏层(每层128个节点)的MLP更新节点、边、全局特征
        @jraph.concatenated_args  # 函数修饰器，用于拼接多个输入参数(例如features里包含多张图的节点特征)，便于函数处理
        def update_node_fn(features):
            return hk.nets.MLP([128, 128], name='node')(features)  # 用一个MLP(2个隐藏层，每层128个节点)更新节点特征
        @jraph.concatenated_args
        def update_edge_fn(features):
            return hk.nets.MLP([128, 128], name='edge')(features)
        @jraph.concatenated_args
        def update_global_fn(features):
            return hk.nets.MLP([128, 128], name='global')(features)
        
        # 创建一个图神经网络实例，里面需要包含节点、边、全局特征的更新函数
        graph_net = jraph.GraphNetwork(
            update_edge_fn=update_edge_fn,
            update_node_fn=update_node_fn,
            update_global_fn=update_global_fn,
        )
        
        # 用创建的图神经网络更新图的节点、边、全局特征
        features = graph_net(graphs)  

        # Reshape the node features, and project into keys, queries & values
        node_features = features.nodes[:batch_size * num_variables]  # batch_size=num_envs. 把batch中的所有节点特征提取出来(batch中的一个元素就是一张图，一张图包含nun-variables个节点)
        node_features = node_features.reshape(batch_size, num_variables, -1)  # 每个节点特征都是128维向量，所以reshape后的形状为(batch_size, num_variables, 128)
        node_features = hk.Linear(128 * 3, name='projection')(node_features)  # 对node_features做线性变换，将每个节点特征由原来的128维向量变为128*3维的向量
        queries, keys, values = jnp.split(node_features, 3, axis=2)  #  根据线性变换后的节点特征维度据均分为queries, keys, values三部分，用于下面的注意力机制

        # Self-attention layer
        # 将queries, keys, values输入到多头注意力层中，得到更新后的node_features
        # queries——代表序列中用于查询其他元素的元素，每个查询将“关注”或“寻求”其他元素的信息；
        # keys——代表序列中用于为查询提供信息的元素，每个键将被查询以计算注意力分数；
        # values——表示与每个元素相关的值，这些值将根据注意力分数进行组合以产生最终输出结果；
        # 若输入的形状为(batch_size, num_variables, feature_dim),则输出形状为(batch_size, num_variables, num_heads*key_size)
        node_features = hk.MultiHeadAttention(
            num_heads=4,  # 独立头数
            key_size=32,  # key的维度，通常小于输入key的维度，且输入key会被投射到key_size维度
            w_init_scale=2.  # 初始化权重的缩放因子-方差缩放初始化技术，即根据输入和输出维度大小来调整权重大小，以设置初始权重使得不同层的激活方差大致相同
        )(queries, keys, values)  # queries, keys, values的形状均为(batch_size, num_variables, 128)

        # Replace the node & global features with normalized versions
        features = features._replace(
            nodes=hk.LayerNorm(  # 标准化一个层的激活函数，使其均值为0，方差为1.有助于稳健训练和提高模型泛化性能
                axis=-1,  # 标准化的轴
                create_scale=True,  # 将可学习的缩放参数纳入归一化层中-模型将在训练过程中学习最佳比例
                create_offset=True  # 将可学习偏置参数纳入归一化层中-模型将在训练过程中学习最佳偏移值
            )(node_features),  # 输出的形状为(batch_size, num_variables, 128)
            globals=hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True
            )(features.globals[:batch_size])
        )
 
        # 用两个结构相同的MLP获得graphs中的senders和receivers
        # 咋确定为senders和receivers的？？？
        senders = hk.nets.MLP([128, 128], name='senders')(features.nodes)  # 输入和输出的形状都为(batch_size, num_variables, 128)
        receivers = hk.nets.MLP([128, 128], name='receivers')(features.nodes)

        logits = lax.batch_matmul(senders, receivers.transpose(0, 2, 1))  # 先将receivers的后两个维度置换，再做批次矩阵乘法，所得形状为(batch_size, num_variables, num_variables)
        logits = logits.reshape(batch_size, -1)  # 形状为(batch_size, num_variables*num_variables)
        stop = hk.nets.MLP([128, 1], name='stop')(features.globals)  # 获得终止行动。输入和输出的形状都为(num_graphs, 1), num_graphs=batch_szie

        with hk.experimental.name_scope('posterior_parameters'):  # 创建上下文，将其中的操作归类为posterior_parameters，可以更好的组织和计算图
            adjacencies = edge_features_to_dense(features, edge_masks, num_variables)  # 将边特征转换为稠密数组
            adjacencies = adjacencies[:batch_size]  # 提取前batch_size个图的邻接矩阵
            post_parameters = posterior_parameters_model(features, adjacencies)  # 计算所有环境下图的theta后验分布近似模型的参数-均值和方差

        # 返回gflownet的对数策略-继续加边的概率和停止加边的概率，以及后验分布模型的参数
        return (log_policy(logits * normalization, stop * normalization, masks), post_parameters)

    return _gflownet  # 返回一个transformed function
