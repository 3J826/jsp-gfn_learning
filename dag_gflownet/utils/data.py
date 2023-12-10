import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from collections import namedtuple


# 创建一个类似类结构，名为Dataset，其包含两个fileds:data和interventions
Dataset = namedtuple('Dataset', ['data', 'interventions']) 


def load_artifact_dataset(data, artifact_dir, prefix):
    mapping_filename = artifact_dir / 'intervention_mapping.csv' # 定义intervention_mapping.csv文件路径
    filename = artifact_dir / f'{prefix}_interventions.csv' # 创建路径，其中prefix取值为train/valid


    if filename.exists() and mapping_filename.exists():  # 判断数据集是否包含干预数据，是则为realdata，否则为模型生成的data
        mapping = pd.read_csv(mapping_filename, index_col=0, header=0)
        perturbations = pd.read_csv(filename, index_col=0, header=0)
        
        # 提取干预数据的prefix(train或valid)部分
        interventions = perturbations.dot(mapping.reindex(index=perturbations.columns)) # mapping通过reindex将列数设置成与pertuebations相同
        interventions = interventions.reindex(columns=data.columns)
    else:
        interventions = pd.DataFrame(False, index=data.index, columns=data.columns)  # 无干预数据

    return Dataset(data=data, interventions=interventions.astype(np.float32))


def load_artifact_continuous(artifact_dir):  # train_data,valid_data都来自artifact_dir，即下载到本地的artifact数据
    
    # 加载train_data.csv文件为一个pandas数据帧，并创建用于程序的train数据
    train_data = pd.read_csv(artifact_dir / 'train_data.csv', index_col=0, header=0) # index_col=0表示将第一列作为行索引，header=0表示将第一行作为列索引
    train = load_artifact_dataset(train_data, artifact_dir, 'train')

    valid_data = pd.read_csv(artifact_dir / 'valid_data.csv', index_col=0, header=0)
    valid = load_artifact_dataset(valid_data, artifact_dir, 'valid')

    # 用pickle加载graph.pkl文件
    with open(artifact_dir / 'graph.pkl', 'rb') as f: # 读取graph.pkl文件，其中rb表示以二进制格式打开一个文件用于只读
        # 检查graph.pkl文件内容是否有效
        # pickle.dump(graph, f)
        content = f.read()
        print('content',content)
        try:
            graph = pickle.load(f)
            print('f',f)
        except EOFError:
            graph = None
        print('graph',graph)

    return train, valid, graph
