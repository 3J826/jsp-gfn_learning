import pandas as pd
import urllib.request
import gzip
import json
import pickle
import os

from pathlib import Path
from numpy.random import default_rng
from pgmpy.utils import get_example_model

from dag_graph import sample_erdos_renyi_linear_gaussian
from dag_sampling import sample_from_linear_gaussian

# 定义download函数，用于下载流质细胞仪数据集
def download(url, filename):
    if filename.is_file():
        return filename
    filename.parent.mkdir(exist_ok=True)

    # Download & uncompress archive
    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as uncompressed:
            file_content = uncompressed.read()

    with open(filename, 'wb') as f:
        f.write(file_content)
    
    return filename


def get_data(name, args, rng=default_rng()):
    if name == 'erdos_renyi_lingauss':
        # Create a random DAG(含cpd参数)
        graph = sample_erdos_renyi_linear_gaussian(
            num_variables=args['num_variables'],
            num_edges=args['num_edges'],
            loc_edges=0.0,
            scale_edges=1.0,
            obs_noise=0.1,
            rng=rng
        )
        print('graph',graph)
        # Sample Data from the DAG
        data = sample_from_linear_gaussian(
            graph,
            num_samples=args['num_samples'],
            rng=rng
        )
        # score = 'bge'

    # 流质细胞仪观察数据-从网上下载
    elif name == 'sachs_continuous':  
        graph = get_example_model('sachs')
        filename = download(
            'https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz',
            Path('data/sachs.data.txt')
        )
        data = pd.read_csv(filename, delimiter='\t', dtype=float)
        data = (data - data.mean()) / data.std()  # Standardize data
        # score = 'bge'
    
    # 流质细胞仪干预数据-从网上下载
    elif name =='sachs_interventional':  
        graph = get_example_model('sachs')
        filename = download(
            'https://www.bnlearn.com/book-crc/code/sachs.interventional.txt.gz',
            Path('data/sachs.interventional.txt')
        )
        data = pd.read_csv(filename, delimiter=' ', dtype='category')
        # score = 'bde'
    
    else:
        raise ValueError(f'Unknown graph type: {name}')

    # return graph, data, score
    return graph, data

# 实验一的数据集生成参数
args={'num_variables':5,'num_edges':5,'num_samples':100}

# save data
graph, data = get_data('erdos_renyi_lingauss', args, rng=default_rng())
print('graph', graph)
train_data = data.iloc[:int(args['num_samples']//1.25)]
valid_data = data.iloc[int(args['num_samples']//1.25):]
# print('train_data',train_data, 'valid_data',valid_data)

# 将数据保存到本地
output_folder = Path('H:\Papers\JSP-GFN\jax-jsp-gfn-master\DataSet_Gen\Dataset_ex1')
output_folder.mkdir(exist_ok=True)  # 创建一个文件夹，用于存放输出的数据
train_data.to_csv(output_folder / 'train_data.csv')  # 保存数据到csv文件
valid_data.to_csv(output_folder / 'valid_data.csv')
with open(output_folder / 'graph.pkl', 'wb') as f:
    pickle.dump(graph, f)

# 将数据上传到w&b artifact    
import wandb
wandb.login(key = '6558815ff993312c6b39be2c636f38401fbeb6d3')
wandb.init(project='jsp-gfn-fuxian', name='jsp-gfn-run0')
metadata = {
    'name': 'ex1_inputdata',
    'description': 'A dataset for model learning in experiment1',
    'version': '1.0',
    'creation_date': '2023-11-25',
    'seeds': [0,1,2,3,4,5,6,7,8,9],
    'cpd_kwargs': {
        'obs_noise': 0.06,
    }
}
artifact = wandb.Artifact(name='ex1_inputdata', type='dataset')
artifact.metadata.update(metadata)
artifact.add_file(r"H:\Papers\JSP-GFN\jax-jsp-gfn-master\DataSet_Gen\Dataset_ex1\train_data.csv")
artifact.add_file(r"H:\Papers\JSP-GFN\jax-jsp-gfn-master\DataSet_Gen\Dataset_ex1\valid_data.csv")
artifact.add_file(r"H:\Papers\JSP-GFN\jax-jsp-gfn-master\DataSet_Gen\Dataset_ex1\graph.pkl")
wandb.log_artifact(artifact) # 加载artifact到我的W&B账户
wandb.finish

# wandb.login(key = '6558815ff993312c6b39be2c636f38401fbeb6d3')
# api = wandb.Api()
# artifact = api.artifact('ds-ai1230/jsp-gfn-fuxian/ex1_inputdata:v0')
# print('artifact',artifact)
# artifact_dir = Path(artifact.download()) / f'{0:02d}'
# print('artifact_dir',artifact_dir)
