import numpy as np
import pandas as pd
from itertools import combinations
import math
from scipy.stats import norm
import networkx as nx
from matplotlib import pyplot as plt
from scipy.linalg import pinv
from causallearn.search.ConstraintBased.PC import pc
import csv
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge  # 新增导入
from causallearn.graph.Node import Node


def load_csv(path):
    data_read = pd.read_csv(path, header=0)
    data = data_read.values
    var_names = data_read.columns.tolist()
    print(data.shape)
    return data, var_names

data1, var_names = load_csv("./MCI_data1.csv")
print(data1)


## 创建背景知识
class MyNode:
    def __init__(self, name=None):
        self.name = name
        self.node_type = None
        self.node_variable_type = None
        self.center_x = 0
        self.center_y = 0
        self.attributes = {}

    def get_name(self) -> str:
        return self.name

    def set_name(self, name: str):
        self.name = name

    def get_node_type(self):
        return self.node_type

    def set_node_type(self, node_type):
        self.node_type = node_type

    def get_node_variable_type(self):
        return self.node_variable_type

    def set_node_variable_type(self, var_type):
        self.node_variable_type = var_type

    def __str__(self):
        return self.name

    def get_center_x(self) -> int:
        return self.center_x

    def set_center_x(self, center_x: int):
        self.center_x = center_x

    def get_center_y(self) -> int:
        return self.center_y

    def set_center_y(self, center_y: int):
        self.center_y = center_y

    def set_center(self, center_x: int, center_y: int):
        self.center_x = center_x
        self.center_y = center_y

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, MyNode) and self.name == other.name

    def like(self, name: str):
        new_node = MyNode(name)
        new_node.node_type = self.node_type
        new_node.node_variable_type = self.node_variable_type
        new_node.center_x = self.center_x
        new_node.center_y = self.center_y
        new_node.attributes = self.attributes.copy()
        return new_node

    def get_all_attributes(self):
        return self.attributes

    def get_attribute(self, key):
        return self.attributes.get(key)

    def remove_attribute(self, key):
        if key in self.attributes:
            del self.attributes[key]

    def add_attribute(self, key, value):
        self.attributes[key] = value

# 使用自定义的MyNode类
var_name1 = 'APOE4'
var_name2 = 'ABETA.bl'

node1 = MyNode(var_name1)
node2 = MyNode(var_name2)

# 创建背景知识对象
background = BackgroundKnowledge()

# 添加必需的边
background.add_required_by_node(node1, node2)




# 禁止任何节点指向AGE
for var in var_names:
    if var != 'AGE':
        background.add_forbidden_edge(var, 'AGE')


# 传入背景知识参数
cg = pc(data1, alpha=0.05, indep_test='fisherz', stable=True, uc_rule=0, uc_priority=2,
        mvpc=True, correction_name='MV_Crtn_Fisher_Z', background_knowledge=background,  # 修改这里
        verbose=False, show_progress=False)

# 可视化
cg.draw_pydot_graph(labels=var_names)

# 保存图像
pyd = GraphUtils.to_pydot(cg.G, labels=var_names)
pyd.write_png('predict.png')