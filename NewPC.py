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


def load_csv(path):
    data_read = pd.read_csv(path, header=0)  # 加载数据，第一行作为列名
    data = data_read.values  # 获取数据
    var_names = data_read.columns.tolist()  # 获取列名
    print(data.shape)
    return data, var_names  # 返回数据和变量名


data1, var_names = load_csv("./final_data_SMC.csv")
print(data1)


cg = pc(data1, alpha=0.05, indep_test='fisherz', stable=True, uc_rule=0, uc_priority=2,
        mvpc=True, correction_name='MV_Crtn_Fisher_Z', background_knowledge=None, verbose= False, show_progress=False)

# visualization using pydot
cg.draw_pydot_graph(labels=var_names)

# save the graph
from causallearn.utils.GraphUtils import GraphUtils

pyd = GraphUtils.to_pydot(cg.G, labels=var_names)
pyd.write_png('predict.png')


