import numpy as np
import gym

from copy import deepcopy
from gym.spaces import Dict, Box, Discrete


class GFlowNetDAGEnv(gym.vector.VectorEnv):
    def __init__(
            self,
            num_envs,
            num_variables,
            max_parents=None,
        ):
        """GFlowNet environment for learning a distribution over DAGs.

        Parameters
        ----------
        num_envs : int
            Number of parallel environments, or equivalently the number of
            parallel trajectories to sample.
        
        num_variables : int
            Number of variables in the graphs.

        max_parents : int, optional
            Maximum number of parents for each node in the DAG. If None, then
            there is no constraint on the maximum number of parents.
        """
        self.num_variables = num_variables
        self._state = None
        self.max_parents = max_parents or self.num_variables

        shape = (self.num_variables, self.num_variables)
        max_edges = self.num_variables * (self.num_variables - 1) // 2
        observation_space = Dict({
            'adjacency': Box(low=0., high=1., shape=shape, dtype=np.int_),
            'mask': Box(low=0., high=1., shape=shape, dtype=np.int_),
            'num_edges': Discrete(max_edges),  # 离散空间，其中元素的范围为[0, max_edges)，num_edges为从该空间中采样的一个元素
            'score': Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float_),  # 连续空间，元素范围为[-inf, inf]
            'order': Box(low=-1, high=max_edges, shape=shape, dtype=np.int_)  # 这个order是什么意思？
        })
        action_space = Discrete(self.num_variables ** 2 + 1)  # 行动空间Discrete(26)
        super().__init__(num_envs, observation_space, action_space)  # 初始化父类gym.vector.VectorEnv

    def reset(self):  # 一个环境下一张图
        shape = (self.num_envs, self.num_variables, self.num_variables)  # (8,5,5)
        closure_T = np.eye(self.num_variables, dtype=np.bool_)  # 对角线为True，其他地方均为False的num_vars维方阵
        self._closure_T = np.tile(closure_T, (self.num_envs, 1, 1))  # 沿第一个维度将closure_T重复self.num_envs次
        self._state = {
            'adjacency': np.zeros(shape, dtype=np.int_),  # 每个环境的邻接矩阵
            'mask': 1 - self._closure_T,  # 每个环境的掩码
            'num_edges': np.zeros((self.num_envs,), dtype=np.int_),  # 每个环境中图的边数
            'score': np.zeros((self.num_envs,), dtype=np.float_),  # (0,0,0,0,0,0,0,0),每个环境的得分，什么得分？
            'order': np.full(shape, -1, dtype=np.int_)  # 把shape中每个元素换成-1
        }
        return deepcopy(self._state)

    def step(self, actions):
        
        # 获取每个action对应的起始节点和终止节点——这表示actions的实际意义
        # 起始节点为商，目标节点为余数，二者形状都为(8,)，分别对应每个环境下的图的下一步操作——应该添加哪条边
        sources, targets = divmod(actions, self.num_variables)  
        
        # 创建布尔向量，形状与actions相同——(8,)
        # sources==self.num_variables为True的位置处的元素为True，该suorce对应的actions=25，表示stop行动
        # 其余地方为False，表示相应的图还可以继续加边
        dones = (sources == self.num_variables)  
        
        # 过滤掉下一步行动为stop的图，留下一步待加边的图
        sources, targets = sources[~dones], targets[~dones]  

        # Make sure that all the actions are valid
        # 起点和终点都只有对应mask中的真值才表示相应的行动有效
        if not np.all(self._state['mask'][~dones, sources, targets]):
            raise ValueError('Some actions are invalid: either the edge to be '
                             'added is already in the DAG, or adding this edge '
                             'would lead to a cycle.')

        # Update the adjacency matrices
        # 将actions中有效的行动对应的adjacency矩阵元素置为1,表示向对应图中加一条边
        self._state['adjacency'][~dones, sources, targets] = 1  
        self._state['adjacency'][dones] = 0

        # Update transitive closure of transpose-用于更新mask的
        source_rows = np.expand_dims(self._closure_T[~dones, sources, :], axis=1)
        target_cols = np.expand_dims(self._closure_T[~dones, :, targets], axis=2)
        self._closure_T[~dones] |= np.logical_and(source_rows, target_cols)  # Outer product——先将source_rows和target_cols进行点对点的逻辑与运算，再将运算结果与self._closure_T[~dones]做或运算
        self._closure_T[dones] = np.eye(self.num_variables, dtype=np.bool_)

        # Update the masks
        self._state['mask'] = 1 - (self._state['adjacency'] + self._closure_T)

        # Update the masks (maximum number of parents)
        num_parents = np.sum(self._state['adjacency'], axis=1, keepdims=True)
        self._state['mask'] *= (num_parents < self.max_parents)

        # Update the order——添加边的顺序
        self._state['order'][~dones, sources, targets] = self._state['num_edges'][~dones]
        self._state['order'][dones] = -1

        # Update the number of edges
        self._state['num_edges'] += 1
        self._state['num_edges'][dones] = 0

        delta_scores = np.zeros((self.num_envs,), dtype=np.float_)

        # Update the scores. The scores returned by the environments are scores
        # relative to the empty graph: score(G) - score(G_0).
        self._state['score'] += delta_scores
        self._state['score'][dones] = 0

        return (deepcopy(self._state), delta_scores, dones, {})

    def close_extras(self, **kwargs):
        pass
