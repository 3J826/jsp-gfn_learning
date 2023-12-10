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
    @hk.transform  # 一种haiku修饰器，本质上是对函数进行包装，使其使用方式符合haiku的模块化和可组合原则
    # 用于将函数转换Haiku transformation，意味着该函数可用于定义一个具有可学习参数的神经网络模块
    def _gflownet(graphs, masks, normalization):  # graph与masks通过init方法传入
        # 从mask矩阵中提取batch_size和num_variables
        batch_size, num_variables = masks.shape[:2]
        # 用1初始化边掩码-将graphs中edges的元素全部置为1
        edge_masks = jnp.ones(graphs.edges.shape, dtype=jnp.float32)

        # Embedding of the nodes & edges
        node_embeddings = hk.Embed(num_variables, embed_dim=128)
        edge_embedding = hk.get_parameter('edge_embed', shape=(1, 128),
            init=hk.initializers.TruncatedNormal())

        # 用GNN更新graphs中的节点、边和全局特征
        graphs = graphs._replace(
            nodes=node_embeddings(graphs.nodes),
            edges=jnp.repeat(edge_embedding, graphs.edges.shape[0], axis=0),
            globals=jnp.zeros((graphs.n_node.shape[0], 1)),
        )
        # Define graph network updates
        @jraph.concatenated_args
        def update_node_fn(features):
            return hk.nets.MLP([128, 128], name='node')(features)
        @jraph.concatenated_args
        def update_edge_fn(features):
            return hk.nets.MLP([128, 128], name='edge')(features)
        @jraph.concatenated_args
        def update_global_fn(features):
            return hk.nets.MLP([128, 128], name='global')(features)
        # 创建一个图神经网络实例
        graph_net = jraph.GraphNetwork(
            update_edge_fn=update_edge_fn,
            update_node_fn=update_node_fn,
            update_global_fn=update_global_fn,
        )
        features = graph_net(graphs)  # 用图神经网络更新图的节点、边、全局特征 

        # Reshape the node features, and project into keys, queries & values
        
        node_features = features.nodes[:batch_size * num_variables]
        node_features = node_features.reshape(batch_size, num_variables, -1)
        node_features = hk.Linear(128 * 3, name='projection')(node_features)  # 对node_features做线性变换，输出维度为128*3
        queries, keys, values = jnp.split(node_features, 3, axis=2)

        # Self-attention layer
        node_features = hk.MultiHeadAttention(
            num_heads=4,  # 独立头数
            key_size=32,  # 用作attention的keys和queries的维度，每个head都有相应的key和query向量
            w_init_scale=2.  # 初始化线性映射权重的缩放因子
        )(queries, keys, values)  # values是每个key对应的值

        # Replace the node & global features with normalized versions
        features = features._replace(
            nodes=hk.LayerNorm(  # 标准化一个层的激活函数，使其均值为0，方差为1.有助于稳健训练和提高模型泛化性能
                axis=-1,  # 标准化的轴
                create_scale=True,  # 将可学习的缩放参数纳入归一化层中-模型将在训练过程中学习最佳比例
                create_offset=True  # 将可学习偏置参数纳入归一化层中-模型将在训练过程中学习最佳偏移值
            )(node_features),
            globals=hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True
            )(features.globals[:batch_size])
        )
 
        # 用MLP更新graphs中的senders和receivers
        senders = hk.nets.MLP([128, 128], name='senders')(features.nodes)
        receivers = hk.nets.MLP([128, 128], name='receivers')(features.nodes)

        logits = lax.batch_matmul(senders, receivers.transpose(0, 2, 1))  # 先将receivers的后两个维度交换，再做批次矩阵乘法
        logits = logits.reshape(batch_size, -1)
        stop = hk.nets.MLP([128, 1], name='stop')(features.globals)  # 终止信号

        with hk.experimental.name_scope('posterior_parameters'):  # 创建上下文，将其中的操作归类为posterior_parameters，可以更好的组织和计算图
            adjacencies = edge_features_to_dense(features, edge_masks, num_variables)  # 将边特征转换为稠密数组
            adjacencies = adjacencies[:batch_size]  # 只提取前batch_size个元素
            post_parameters = posterior_parameters_model(features, adjacencies)  # 计算cpd参数的后验分布模型的参数

        return (log_policy(logits * normalization, stop * normalization, masks), post_parameters)

    return _gflownet  # 返回一个transformed function
