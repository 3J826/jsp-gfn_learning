import numpy as np
import math

from numpy.random import default_rng
from jraph import GraphsTuple

from dag_gflownet.utils.jraph_utils import to_graphs_tuple


class ReplayBuffer:
    def __init__(self, capacity, num_variables):
        self.capacity = capacity
        self.num_variables = num_variables

        nbytes = math.ceil((num_variables ** 2) / 8)  # 表达图的邻接矩阵所需的字节数
        # 自定义数据类型，包含了邻接矩阵、边数、动作、是否探索、分数差、分数、掩码、下一步邻接矩阵、下一步掩码
        dtype = np.dtype([
            ('adjacency', np.uint8, (nbytes,)),  # d=5时，邻接矩阵为5*5，nbytes=4
            ('num_edges', np.int_, (1,)),
            ('actions', np.int_, (1,)),
            ('is_exploration', np.bool_, (1,)),
            ('delta_scores', np.float_, (1,)),  # 是什么？
            ('scores', np.float_, (1,)),  # 是什么？
            ('mask', np.uint8, (nbytes,)),
            ('next_adjacency', np.uint8, (nbytes,)),
            ('next_mask', np.uint8, (nbytes,))
        ])
        self._replay = np.zeros((capacity,), dtype=dtype)
        self._index = 0
        self._is_full = False  # ReplayBuffer是否被填满
        self._prev = np.full((capacity,), -1, dtype=np.int_)

    def add(
            self,
            observations,
            actions,  # (8, )
            is_exploration,  # (8,)，每个元素表示相应环境下的动作是否为探索性的
            next_observations,
            delta_scores,
            dones,  # (8,)布尔向量
            prev_indices=None
        ):
        
        indices = np.full((dones.shape[0],), -1, dtype=np.int_)  # (-1,-1,-1,-1,-1,-1,-1,-1)
        
        # 判断是否num_envs个并行图都完成了，若是则直接返回Indices
        if np.all(dones):
            return indices

        # 根据dones更新replaybuffer及各并行环境中图的索引
        # 例如当前已完成0，1，2号图，剩下的4-7号图的索引会更新为0，1，2，3
        num_samples = np.sum(~dones)  # 计数下一步继续加边的图数
        add_idx = np.arange(self._index, self._index + num_samples) % self.capacity  # 计算arange中各个元素除以capacity的余数,(num_samples,)
        self._is_full |= (self._index + num_samples >= self.capacity)
        self._index = (self._index + num_samples) % self.capacity  # 更新self._index
        indices[~dones] = add_idx  # 将未完成的图的索引更新为add_idx

        # 更新data-只保留未完成的图之数据
        data = {
            'adjacency': self.encode(observations['adjacency'][~dones]),  # 添加并行环境中下一步继续加边的图的邻接矩阵
            'num_edges': observations['num_edges'][~dones],  # 添加并行环境中下一步继续加边的图的边数
            'actions': actions[~dones],  #  添加并行环境中下一步继续加边的图的动作
            'delta_scores': delta_scores[~dones],
            'mask': self.encode(observations['mask'][~dones]),
            'next_adjacency': self.encode(next_observations['adjacency'][~dones]),
            'next_mask': self.encode(next_observations['mask'][~dones]),

            # Extra keys for monitoring
            'is_exploration': is_exploration[~dones],
            'scores': observations['score'][~dones],
        }

        # 设置新数据的形状使其匹配replaybuffer中的数据形状
        for name in data:
            shape = self._replay.dtype[name].shape  # 获取data中各个键对应的值的形状
            self._replay[name][add_idx] = np.asarray(data[name].reshape(-1, *shape))
        
        # 保存prev_indices到replaybuffer中
        if prev_indices is not None:
            self._prev[add_idx] = prev_indices[~dones]

        return indices

    def sample(self, batch_size, rng=default_rng()):
        
        """batch_size——采样量, 默认为32"""
        
        # 从replaybuffer中无重复随机采样batch_size个样本——获取样本标号
        # 每个样本代表并行环境中各个图的状态前进一步后的情况
        indices = rng.choice(len(self), size=batch_size, replace=False)
        samples = self._replay[indices]

        adjacency = self.decode(samples['adjacency'], dtype=np.int_)
        next_adjacency = self.decode(samples['next_adjacency'], dtype=np.int_)

        # Convert structured array into dictionary
        return {
            'adjacency': adjacency.astype(np.float32),
            'graph': to_graphs_tuple(adjacency),
            'num_edges': samples['num_edges'],
            'actions': samples['actions'],
            'delta_scores': samples['delta_scores'],
            'mask': self.decode(samples['mask']),
            'next_adjacency': next_adjacency.astype(np.float32),
            'next_graph': to_graphs_tuple(next_adjacency),
            'next_mask': self.decode(samples['next_mask'])
        }

    def __len__(self):
        return self.capacity if self._is_full else self._index

    @property
    def transitions(self):
        return self._replay[:len(self)]

    def save(self, filename):
        data = {
            'version': 3,
            'replay': self.transitions,
            'index': self._index,
            'is_full': self._is_full,
            'prev': self._prev,
            'capacity': self.capacity,
            'num_variables': self.num_variables,
        }
        np.savez_compressed(filename, **data)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            data = np.load(f)
            if data['version'] != 3:
                raise IOError(f'Unknown version: {data["version"]}')
            replay = cls(
                capacity=data['capacity'],
                num_variables=data['num_variables']
            )
            replay._index = data['index']
            replay._is_full = data['is_full']
            replay._prev = data['prev']
            replay._replay[:len(replay)] = data['replay']
        return replay

    def encode(self, decoded):
        encoded = decoded.reshape(-1, self.num_variables ** 2)
        return np.packbits(encoded, axis=1)

    def decode(self, encoded, dtype=np.float32):
        decoded = np.unpackbits(encoded, axis=-1, count=self.num_variables ** 2)
        decoded = decoded.reshape(*encoded.shape[:-1], self.num_variables, self.num_variables)
        return decoded.astype(dtype)

    @property
    def dummy(self):
        shape = (1, self.num_variables, self.num_variables)
        # 一张图
        graph = GraphsTuple(
            nodes=np.arange(self.num_variables),  # (0,1,2,3,4)
            edges=np.zeros((1,), dtype=np.int_),
            senders=np.zeros((1,), dtype=np.int_),
            receivers=np.zeros((1,), dtype=np.int_),
            globals=None,
            n_node=np.full((1,), self.num_variables, dtype=np.int_),
            n_edge=np.ones((1,), dtype=np.int_),
        )
        # 一张图的邻接矩阵
        adjacency = np.zeros(shape, dtype=np.float32)
        return {
            'adjacency': adjacency,
            'graph': graph,
            'num_edges': np.zeros((1,), dtype=np.int_),
            'actions': np.zeros((1,), dtype=np.int_),
            'delta_scores': np.zeros((1,), dtype=np.float_),
            'mask': np.zeros(shape, dtype=np.float32), 
            'next_adjacency': adjacency,  # 下一邻接矩阵和当前的相同，因为此处是对replaybuffer进行初始化
            'next_graph': graph,
            'next_mask': np.zeros(shape, dtype=np.float32)
        }
