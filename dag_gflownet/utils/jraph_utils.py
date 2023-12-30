import numpy as np
import jax.numpy as jnp
import jraph


# 将邻接矩阵转换为graphs_tuple形式
def to_graphs_tuple(adjacencies, pad=True):  
    num_graphs, num_variables = adjacencies.shape[:2]  # adjacencies的形状为(num_envs, num_vars, num_vars)
    n_node = np.full((num_graphs,), num_variables, dtype=np.int_)  # (5,5,5,5,5,5,5,5)并行处理的num_envs张图每张图的节点数都为num_vars

    counts, senders, receivers = np.nonzero(adjacencies) # 返回adjacencies中非零元素的索引
    n_edge = np.bincount(counts, minlength=num_graphs)  # 统计每张图的边数-向量，维度为有边的图数

    # Node features: node indices
    nodes = np.tile(np.arange(num_variables), num_graphs)  # 每张图中节点的索引-(0,1,2,3,4, 0,1,2,3,4, 0,1,2,3,4,...)
    edges = np.ones_like(senders)  # 把senders中的每个元素换成1-作用是啥？

    # 创建图元组
    graphs_tuple =  jraph.GraphsTuple(
        nodes=nodes,  # (0,1,2,3,4, 0,1,2,3,4, 0,1,2,3,4,...)
        edges=edges,  # (1,1,1,...),1的个数为senders的个数
        
        # 把所有num_envs张图中 有边的 那些图抽出来，将这些图的所有senders和receivers的节点索引拉通
        # 例如有三张图有边，则第一张图中5个节点的索引为0-4，第二张图中5个节点的索引为5-9，第三张图中5个节点的索引为10-14
        senders=senders + counts * num_variables,  
        receivers=receivers + counts * num_variables,
        globals=None,
        n_node=n_node,  # num_graphs维向量-每个元素代表每张图的节点数
        n_edge=n_edge,  # num_graphs维向量-每个元素代表每张图的边数
    )
    if pad:   # 如果pad为True，则将图元组的节点数和边数都填充到最近的2的幂次方-为了便于计算机计算及是哟
        graphs_tuple = pad_graph_to_nearest_power_of_two(graphs_tuple)
    return graphs_tuple


def _nearest_bigger_power_of_two(x):
    y = 2
    while y < x:
        y *= 2
    return y


def pad_graph_to_nearest_power_of_two(graphs_tuple):
    # Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py
    # Add 1 since we need at least one padding node for pad_with_graphs.
    pad_nodes_to = _nearest_bigger_power_of_two(np.sum(graphs_tuple.n_node)) + 1
    pad_edges_to = _nearest_bigger_power_of_two(np.sum(graphs_tuple.n_edge))

    # Add 1 since we need at least one padding graph for pad_with_graphs.
    # We do not pad to nearest power of two because the batch size is fixed.
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
    return jraph.pad_with_graphs(
        graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to)


def edge_features_to_dense(graphs, features, num_variables):
    # Get the batch indices
    batch_indices = jnp.arange(graphs.n_edge.shape[0])
    batch_indices = jnp.repeat(batch_indices, graphs.n_edge,
        axis=0, total_repeat_length=graphs.edges.shape[0])  # 不管graphs.n_edge，直接以total_repeat_length作为重复次数

    # Remove the offset to senders & receivers-去除偏移量，使得批处理不会在变量索引时产生冲突
    offset = batch_indices * num_variables
    senders = graphs.senders - offset
    receivers = graphs.receivers - offset

    # Transform the features into a dense array-图的特征可能比较稀疏，所以需要转换为稠密数组
    shape = (graphs.n_node.shape[0], num_variables, num_variables) + features.shape[1:]
    dense_array = jnp.zeros(shape, dtype=features.dtype)
    return dense_array.at[batch_indices, senders, receivers].add(features)
