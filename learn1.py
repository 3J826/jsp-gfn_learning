# from numpy.random import default_rng
# rng = default_rng(0)
# print(rng)

# csv文件读取
# import pandas as pd
# # Define the filenames
# mapping_filename = 'mapping.csv'
# filename = 'perturbations.csv'

# # Read the CSV files into DataFrames
# mapping = pd.read_csv(mapping_filename, index_col=0, header=0)
# print(mapping)
# perturbations = pd.read_csv(filename, index_col=0, header=0)
# print(perturbations)
# print(perturbations.columns)
# print(mapping.reindex(index=perturbations.columns)) # 把perturbations的columns头与mapping的index对应起来，并清空原有数据

# # Perform the dot product and reindexing
# interventions = perturbations.dot(mapping.reindex(index=perturbations.columns))
# print(interventions)

# data = pd.DataFrame(index=['P', 'Q', 'R'], columns=['A', 'B', 'C', 'D', 'E'])
# print(data)
# interventions = interventions.reindex(columns=data.columns)
# print(interventions)

# from numpy.random import default_rng
# rng = default_rng()
# theta = rng.normal(0.1, 1, size=(3 + 1,))
# print(theta)

# args={'num_variables':3,'num_edges':3,'num_samples':5}
# print(args['num_variables'])

#------------------------------ PRNGKey-----------------------------------------------
# import jax
# import jax.numpy as jnp
# # Example 1: Using a specific seed value
# seed1 = 123
# seed2 = 321
# key1 = jax.random.PRNGKey(seed1)
# # key2 = jax.random.PRNGKey(seed2)
# # print(key1,key2)
# # Generate a random number using the key
# # rand1 = jax.random.uniform(key1)
# # rand2 = jax.random.uniform(key2)
# # rand3 = jax.random.uniform(key1, shape=(3,))
# # rand4 = jax.random.uniform(key1, shape=(3,2))
# # rand5 = jax.random.normal(key1, shape=(3,2))
# # print('rand1',rand1,'rand2',rand2,'rand3',rand3,'rand4',rand4,'rand5',rand5)
# # 把key1随机分成两部分
# key, subkey = jax.random.split(key1)
# print(key,subkey)

#--------------------------------------pdDataFrame运算--------------------------------
# import pandas as pd
# import numpy as np
# from pathlib import Path
# perturbations = Path("H:\Papers\JSP-GFN\jax-jsp-gfn-master\perturbations.csv")
# mapping = Path("H:\Papers\JSP-GFN\jax-jsp-gfn-master\mapping.csv")
# perturbations = pd.read_csv(perturbations, index_col=0, header=0)
# mapping = pd.read_csv(mapping, index_col=0, header=0)
# data = pd.DataFrame({
#     'feature1': [1, 2, 3],
#     'feature2': [4, 5, 6],
#     'feature3': [7, 8, 9],
# })
# print('data',data)
# interventions = perturbations.dot(mapping.reindex(index=perturbations.columns))
# interventions1 = interventions.reindex(columns=data.columns)
# interventions2 = pd.DataFrame(False, index=data.index, columns=data.columns)
# print('reindex',mapping.reindex(index=perturbations.columns))
# print('interventions',interventions)
# print('interventions1',interventions1)
# print('interventions2',interventions2)

#--------------------------------------------jax转换嵌套数据结构--------------------
# import jax
# import jax.numpy as jnp
# import numpy as np
# Suppose you have a nested data structure, for example, a dictionary with NumPy arrays.
# train = {
#     'input': np.array([1.0, 2.0, 3.0]),
#     'output': {
#         'values': np.array([4.0, 5.0, 6.0]),
#         'labels': np.array([0, 1, 1])
#     }
# }
# # Use tree_map to convert all NumPy arrays to JAX arrays.
# train_jnp = jax.tree_util.tree_map(jnp.asarray, train)
# # Print the original and converted structures.
# print("Original train structure:")
# print(train)
# print("Converted train_jnp structure:")
# print(train_jnp)
# nested_data = {'a': [1, 2, 3], 'b': {'c': [4, 5, 6], 'd': 7}}
# # Define a function to square each element
# def square(x):
#     return x**2
# # Use tree_map to apply the function element-wise
# result = jax.tree_util.tree_map(square, nested_data)
# print(result)

#-----------------------------------------obs_scale--------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt

# def generate_data(num_points, slope=2.0, intercept=1.0, obs_scale=1.0):
#     x = np.linspace(0, 10, num_points)
#     noise = np.random.normal(0, obs_scale, size=num_points)
#     y_true = slope * x + intercept
#     y_observed = y_true + noise
#     return x, y_observed

# # Varying obs_scale
# obs_scales = [0.1, 1.0, 5.0]

# plt.figure(figsize=(12, 8))

# for obs_scale in obs_scales:
#     x, y_observed = generate_data(100, obs_scale=obs_scale)
#     plt.scatter(x, y_observed, label=f'obs_scale={obs_scale}')

# plt.plot(x, 2 * x + 1, color='black', linestyle='--', label='True Line (slope=2, intercept=1)')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.title('Effect of obs_scale in Linear Regression')
# plt.show()

#-----------------------------------------jax.vmap--------------------------------------
# import jax
# import jax.numpy as jnp
# from functools import partial
# # Original function
# def add_and_square(a, b, c):
#     return (a + b) * c
# # Vectorized function using jax.vmap
# vectorized_add_and_square = partial(jax.vmap, in_axes=(0, 0, 0), out_axes=1)(add_and_square)
# # Example input arrays
# a_values = jnp.array([[1.0,1.1,1.2], [2.0,2.1,2.2], [3.0,3.1,3.2]])
# b_values = jnp.array([[4.0,4.1,4.2], [5.0,5.1,5.2], [6.0,6.1,6.2]])
# c_values = jnp.array([[7.0,7.1,7.2], [8.0,8.1,8.2], [9.0,9.1,9.2]])
# # Applying the vectorized function to arrays
# result = vectorized_add_and_square(a_values, b_values, c_values)
# print(result)


#---------------------------------------hk.net.MLP-------------------------------------
# import haiku as hk
# import jax
# import jax.numpy as jnp
# def forward(x):
#   mlp = hk.nets.MLP([300, 100, 10])
#   return mlp(x)
# forward = hk.transform(forward)
# rng = hk.PRNGSequence(jax.random.PRNGKey(42))
# x = jnp.ones([8, 28 * 28])
# params = forward.init(next(rng), x)
# print(params)
# logits = forward.apply(params, next(rng), x)
# print(logits)
#----------------------------------------------haiku.linear--------------------------------
# import haiku as hk
# import jax
# import jax.numpy as jnp
# # Define a simple neural network using haiku.linear
# def simple_network(x):
#     linear_layer = hk.Linear(output_size=64)
#     return linear_layer(x)
# # Transform the forward function
# transformed_network = hk.transform(simple_network)
# # Initialize parameters
# rng = hk.PRNGSequence(42)
# params = transformed_network.init(next(rng), jnp.ones((32, 128)))
# # Apply the network to some input
# output = transformed_network.apply(params, next(rng), jnp.ones((32, 128)))
# print(output)


#---------------------------------------------------jnp.split--------------------------------
# import jax.numpy as jnp
# # Create a 3D array as an example
# params = jnp.arange(24)
# # print(params)
# # params = params.reshape(2, 3, 4)
# # Split the array along the last axis into two subarrays
# split_params = jnp.split(params, 2, axis=0)
# print('split_params', split_params)
# Print the original and split arrays
# print("Original Array:")
# print(params)
# print("\nSplit Arrays:")
# for i, subarray in enumerate(split_params):
#     print(f"Subarray {i + 1}:")
#     print(subarray)

#-------------------------------------------------------jnp.triu_indices------------------
# rows, cols = jnp.triu_indices(5, k=1)
# print(rows,cols)

# import jax.numpy as jnp
# Create a 3x3 matrix with random values
# key = jax.random.PRNGKey(0)
# matrix = jax.random.normal(key, shape = (3, 3))
# # Get the indices of the upper triangular part
# upper_triangular_indices = jnp.triu_indices(3, 1)  # 3 is the size of the matrix, 0 is the offset
# print('upper_indices:',upper_triangular_indices)
# # Create an upper triangular matrix using the indices
# upper_triangular_matrix = jnp.zeros_like(matrix)
# upper_triangular_matrix = upper_triangular_matrix.at[upper_triangular_indices].add(matrix[upper_triangular_indices])
# print("Original Matrix:")
# print(matrix)
# print("\nUpper Triangular Matrix:")
# print(upper_triangular_matrix)

#-----------------------------------------jnp.expand_dims--------------------------------
# import jax.numpy as jnp
# samples = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# loc = np.array([2.0, 3.0, 4.0])
# mask = np.array([1, 0, 1])
# result = jnp.expand_dims((samples - loc) * mask, axis=-2)
# print('s-l:')
# print(samples - loc)
# print('s-l*m:')
# print(((samples - loc) * mask).shape)
# print('r:')
# print(result.shape)

#------------------------------------jnp.diagonal--------------------------------
# import jax.numpy as jnp
# # Example precision_triu matrix
# precision_triu = jnp.array([[1.0, 2.0, 3.0],
#                             [4.0, 4.0, 5.0],
#                             [6.0, 1.0, 6.0]])
# # Extract diagonal elements along the last two axes
# diags1 = jnp.diagonal(precision_triu, axis1=-2, axis2=-1)
# print(diags1)
# diags2 = jnp.diagonal(precision_triu, axis1=-1, axis2=-2)
# print(diags2)

# a = jnp.array([[1, 2, 3], [4, 5, 6]])
# print(jnp.diagonal(a, axis1=-2, axis2=-1))
# print(jnp.diagonal(a, axis1=-1, axis2=-2))

#-----------------------------------------------------------------------------------
# import jax.numpy as jnps
# Example values
# params_loc = jnp.array([1.0, 2.0, 3.0]) # (3,)
# print(params_loc.shape)
# lhs = jnp.array([[0.1, 0.2],
#                  [0.3, 0.4],
#                  [0.5, 0.6]])  # (3,2)
# # Roll the last axis to the position before the first axis
# rhs = jnp.rollaxis(lhs, -1)
# print('rhs:', rhs)  
# # Calculate samples by adding loc and the rolled lhs
# samples = params_loc + rhs
# print('smaples:', samples)
# print(jnp.swapaxes(samples, 0, 1))

# print(jnp.arange(3))

#------------------------------------------------jnp.repeat()--------------------------
# import jax.numpy as jnp

# # Example global features
# global_features_example = jnp.array([1.0, 2.0, 3.0])  # This might be features.globals

# # Number of times to repeat
# num_variables_example = 4

# # Repeat the global features along axis 1
# global_features_repeated = jnp.repeat(global_features_example[:, None], num_variables_example, axis=1)
# print(global_features_example[:, None])
# print(global_features_repeated)

# print(np.zeros((1,), dtype=np.int_))


#---------------------------------------hk.MultiHeadAttention--------------------------------
# import haiku as hk
# import jax.numpy as jnp

# # Assuming each word is represented as a 3-dimensional vector
# embedding_dim = 3
# num_heads = 2

# # Create a dummy input sequence
# sequence_length = 5

# @hk.transform
# def my_model(input_sequence):
#     # Create a multi-head self-attention layer
#     multihead_attention = hk.MultiHeadAttention(
#         num_heads=num_heads,
#         key_size=embedding_dim // num_heads,
#         w_init_scale=1.0,
#     )

#     # Apply the multi-head attention layer to the input sequence
#     attention_output = multihead_attention(input_sequence, input_sequence, input_sequence)

#     return attention_output

# params = my_model.init(jax.random.PRNGKey(42), jnp.ones((sequence_length, embedding_dim)))

# input_sequence = jax.random.normal(jax.random.PRNGKey(42), shape = (sequence_length, embedding_dim))

# # Apply the multi-head attention layer to the input sequence
# attention_output = my_model.apply(params, jax.random.PRNGKey(42), input_sequence)

# # Print the input sequence and attention output
# print("Input Sequence:")
# print(input_sequence)
# print("\nAttention Output:")
# print(attention_output)

#--------------------------------lax.batch_matmul--------------------------------
# import jax
# import jax.numpy as jnp
# # Create two batches of matrices
# # senders_batch = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# # receivers_batch = jnp.array([[[2, 0], [1, 2]], [[1, 0], [0, 1]]])
# # # Perform batched matrix multiplication
# # result_batch = jax.lax.batch_matmul(senders_batch, receivers_batch.transpose())
# # print(result_batch)
# batch_size = 2
# matrix_size = 3
# senders_batch = jax.random.normal(jax.random.PRNGKey(42), (batch_size, matrix_size, matrix_size))
# receivers_batch = jax.random.normal(jax.random.PRNGKey(24), (batch_size, matrix_size, matrix_size))
# # Perform 3D batched matrix multiplication
# result_batch = jax.lax.batch_matmul(senders_batch, receivers_batch)
# print(result_batch)

#-----------------------------Box--------------------------
# import numpy as np
# from gym.spaces import Box, Discrete
# Define the Box space
# box_space = Box(low=0., high=1., shape=(3, 3), dtype=np.int_)
# print("Box Space:", box_space)
# # Sample a random state from the space
# random_state = box_space.sample()
# # Display the sampled state
# print("Sampled State:")
# print(random_state)
# # Check if the sampled state is within the defined space
# is_valid = box_space.contains(random_state)
# print("\nIs the sampled state valid? ", is_valid)

# print(Discrete(3))
# print(Discrete(3).sample())

#------------------------------------------np.tile---------------------------------
# import numpy as np
# # Assume closure_T is a 2D array for the sake of illustration
# closure_T = np.array([[1, 2], [3, 4]])
# # Tile the array along specified dimensions
# # result1= np.tile(closure_T, (3, 1, 1))
# # result2= np.tile(closure_T, (1, 3, 1))
# # result3= np.tile(closure_T, (3, 2, 1))
# result4 = np.tile(closure_T, (1,1,3))
# # Display the result
# # print('r1:',result1)
# # print('r2:',result2)
# # print('r3:',result3)
# print('r4:',result4)


#---------------------------------------------------------------
# import optax
# from tqdm import trange

# exploration_schedule = jax.jit(optax.linear_schedule(  # jax.jit-编译函数，加快函数的运行速度
#         init_value=jnp.array(0.),
#         end_value=jnp.array(1. - 0.1),  # min_exploration默认为0.1
#         transition_steps=6 // 2,  # num_iterations默认为100000
#         transition_begin=3,  # prefill默认为1000
#     ))


# with trange(3 + 6, desc='Training') as pbar:
#         for iteration in pbar:
#             print(iteration)
#             epsilon = exploration_schedule(iteration)
#             print(epsilon)
            
#-------------------------------np.nonzero()---------------------------------
# adjacency = np.zeros((8,5,5),dtype = np.int_)
# print(np.nonzero(adjacency))

# import numpy as np
# # print(np.tile(np.arange(5),8))

# senders = [1,1,3,4,5]
# counts = [0,1,1,2,3]
# # print(np.ones_like(senders))
# print(senders + counts * 6)

#---------------------------------nearest_bigger_power_of_two---------------------------
# def _nearest_bigger_power_of_two(x):
#     y = 2
#     while y < x:
#         y *= 2
#     return y
# print(_nearest_bigger_power_of_two(3))

#----------------------------------------------------------------
# import numpy as np
# a = np.array([1,2,3])
# print(a.sum())

#---------------------------------jraph.pad_with_graphs---------------------------
import jraph
import numpy as np

num_graphs, num_variables = [4, 3]
n_node = np.full((num_graphs,), num_variables, dtype=np.int_)
print('n_node:',n_node)
adjacencies = np.array([[[0,1,0],[0,0,1],[0,0,0]], [[0,1,1],[0,0,0],[0,1,0]], [[0,1,1],[0,0,1],[0,0,0]], [[0,0,1],[1,0,1],[0,0,0]]])
counts, senders, receivers = np.nonzero(adjacencies)
print('counts:',counts)
print('senders:',senders)
print('receivers:',receivers)
n_edge = np.bincount(counts, minlength=num_graphs)
print('n_edge:',n_edge)

# Node features: node indices
nodes = np.tile(np.arange(num_variables), num_graphs)
# print('nodes:',nodes)
edges = np.ones_like(senders)  # 作用是啥？
# print('edges:',edges)

graphs_tuple =  jraph.GraphsTuple(
    nodes=nodes,
    edges=edges,
    senders=senders + counts * num_variables,  # 为啥这样做？
    receivers=receivers + counts * num_variables,
    globals=None,
    n_node=n_node,
    n_edge=n_edge,
)
print('gt:',graphs_tuple)
print('senders:', graphs_tuple.senders)
print('receivers:', graphs_tuple.receivers)

def _nearest_bigger_power_of_two(x):
    y = 2
    while y < x:
        y *= 2
    return y

def pad_graph_to_nearest_power_of_two(graphs_tuple):
    # Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py
    # Add 1 since we need at least one padding node for pad_with_graphs.
    pad_nodes_to = _nearest_bigger_power_of_two(np.sum(graphs_tuple.n_node)) + 1
    print('pad_nodes_to:',pad_nodes_to)
    pad_edges_to = _nearest_bigger_power_of_two(np.sum(graphs_tuple.n_edge))
    print('pad_edges_to:',pad_edges_to)

    # Add 1 since we need at least one padding graph for pad_with_graphs.
    # We do not pad to nearest power of two because the batch size is fixed.
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
    return jraph.pad_with_graphs(
        graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to)


graphs_tuple = pad_graph_to_nearest_power_of_two(graphs_tuple)
print('gt_pad:',graphs_tuple)





