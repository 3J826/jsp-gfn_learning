import jax.numpy as jnp
import haiku as hk
import jax

ACTIVATIONS = {
    'sigmoid': jax.nn.sigmoid,
    'tanh': jax.nn.tanh,
    'relu': jax.nn.relu,
    'leakyrelu': jax.nn.leaky_relu
}

def create_mlp(hidden_sizes, activation='relu', **kwargs):
    @hk.without_apply_rng  # 保证函数_mlp的输出不会随着随机数的变化而变化
    @hk.transform  # haiku修饰器，表示_mlp是一个haiku转换函数
    def _mlp(inputs):
        outputs = hk.nets.MLP(
            hidden_sizes,
            activation=ACTIVATIONS[activation],
            with_bias=True,
            activate_final=False,
            name='mlp',
            **kwargs
        )(inputs)
        if outputs.shape[-1] == 1:
            outputs = jnp.squeeze(outputs, axis=-1)
        return outputs
    return _mlp

def get_first_weights(params):  # 从模型中提取第一个线性层的权重
    params = hk.data_structures.filter(
        lambda module_name, name, _: (module_name == 'mlp/~/linear_0') and (name == 'w'),
        params
    )
    return params['mlp/~/linear_0']['w']
