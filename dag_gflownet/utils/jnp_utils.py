import jax.numpy as jnp

from jax import random


def batch_random_choice(key, probas, masks):
    """probas-(batch_size, num_vars*num_vars):每个环境下采用相应策略(继续加边或停止加边)的概率
       masks-(batch_size, num_variables, num_variables)
    """
    # Sample from the distribution-为每个batch生成一个（0，1）之间的随机数
    uniform = random.uniform(key, shape=(probas.shape[0], 1))  # （batch_size,1)
    
    # 沿第2个维度(维数为26)计算累积概率
    cum_probas = jnp.cumsum(probas, axis=1)  
    
    # 对cum_probas的每一行，比较该行每个元素与uniform的大小，计算每行的True值数量，得到一个(8,1)的矩阵
    # 每张图的cum_probas中小于该图对应的uniform值的元素个数即为该图采样的行动
    # 一张图采样一个行动
    samples = jnp.sum(cum_probas < uniform, axis=1, keepdims=True)  # cum_probas形状为(8,26),uniform形状为(8,1),samples形状为(8,1)

    # In rare cases, the sampled actions may be invalid, despite having
    # probability 0. In those cases, we select the stop action by default.
    stop_mask = jnp.ones((masks.shape[0], 1), dtype=masks.dtype)  # Stop action is always valid
    masks = masks.reshape(masks.shape[0], -1)  # (batch_size, num_variables*num_variables)
    masks = jnp.concatenate((masks, stop_mask), axis=1)  # (8, 26)
  
    # 将samples经masks过滤一遍判断样本是否有效
    is_valid = jnp.take_along_axis(masks, samples, axis=1)
    stop_action = masks.shape[1]
    # 若样本无效(相应masks处的元素值为0)则用stop_action替换
    samples = jnp.where(is_valid, samples, stop_action)

    return jnp.squeeze(samples, axis=1)
                            