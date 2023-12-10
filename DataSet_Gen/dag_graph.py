import numpy as np
import networkx as nx
import string

from itertools import chain, product, islice, count

from numpy.random import default_rng
from pgmpy import models
from pgmpy.factors.continuous import LinearGaussianCPD


# 生成随机图-不含cpd参数
def sample_erdos_renyi_graph(
        num_variables,
        p=None,
        num_edges=None,
        nodes=None,
        create_using=models.BayesianNetwork,
        rng=default_rng()
    ):
    
    # 计算图中每条边出现的概率
    if p is None:
        if num_edges is None:
            raise ValueError('One of p or num_edges must be specified.')
        p = num_edges / ((num_variables * (num_variables - 1)) / 2.)
    
    # 生成节点名称，若nodes为空，则使用字母表中的大写字母作为名称，否则认为节点已有名称
    if nodes is None:
        uppercase = string.ascii_uppercase
        iterator = chain.from_iterable(
            product(uppercase, repeat=r) for r in count(1))
        nodes = [''.join(letters) for letters in islice(iterator, num_variables)]

    # Sample the adjacency matrix
    adjacency = rng.binomial(1, p=p, size=(num_variables, num_variables))
    adjacency = np.tril(adjacency, k=-1)  # Only keep the lower triangular part

    # Permute the rows and columns
    perm = rng.permutation(num_variables)
    adjacency = adjacency[perm, :]
    adjacency = adjacency[:, perm]
    
    # Create the graph-使用adjacency矩阵创建图，并用给定的映射为节点做标签
    graph = nx.from_numpy_array(adjacency, create_using=create_using)
    mapping = dict(enumerate(nodes))  # 节点索引(0, 'A'), (1, 'B'), ...
    nx.relabel_nodes(graph, mapping=mapping, copy=False) # 用给定的映射为节点做新标签

    return graph

# 为随机图加上cpd参数
def sample_erdos_renyi_linear_gaussian(
        num_variables,
        p=None,
        num_edges=None,
        nodes=None,
        loc_edges=0.0,
        scale_edges=1.0,
        obs_noise=0.1,
        rng=default_rng()
    ):
    # Create graph structure
    graph = sample_erdos_renyi_graph(
        num_variables,
        p=p,
        num_edges=num_edges,
        nodes=nodes,
        create_using=models.LinearGaussianBayesianNetwork,
        rng=rng
    )

    # Create the model parameters
    factors = []
    for node in graph.nodes:
        parents = list(graph.predecessors(node))

        # Sample random parameters (from Normal distribution)
        # loc_edges-mean, scale_edges-标准差，size-数组形状
        theta = rng.normal(loc_edges, scale_edges, size=(len(parents) + 1,))
        theta[0] = 0.  # There is no bias term

        # Create factor-CPD
        factor = LinearGaussianCPD(node, theta, obs_noise, parents)
        factors.append(factor)

    # 向图中添加cpd参数
    graph.add_cpds(*factors)
    return graph

