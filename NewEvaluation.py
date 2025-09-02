from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.SHD import SHD

from causallearn.search.ConstraintBased.PC import pc
from SimulateData import simulate_data
from causallearn.utils.GraphUtils import GraphUtils
import csv
import pandas as pd
import numpy as np
def load_csv(path):
    data_read = pd.read_csv(path)
    list = data_read.values.tolist()
    data = np.array(list)
    #print(data.shape)
    # print(data)
    return data

data1 = load_csv("./lucas0_train.csv")

cg = pc(data1, alpha=0.05, indep_test='fisherz', stable=True, uc_rule=0, uc_priority=2,mvpc=True, correction_name='MV_Crtn_Fisher_Z', background_knowledge=None, verbose= False, show_progress=True)
est=cg.G

simulate_data(n=1000, p=12, EN=5, path='lucas0_train.csv')#每实践一次就会刷新一次simulatedata1.csv
data2 = load_csv("./lucas0_train.csv")
cg2 = pc(data2, alpha=0.05, indep_test='fisherz', stable=True, uc_rule=0, uc_priority=2,mvpc=True, correction_name='MV_Crtn_Fisher_Z', background_knowledge=None, verbose= False, show_progress=True)
true_cpdag = cg2.G


# For arrows
arrow = ArrowConfusion(true_cpdag, est)

arrowsTp = arrow.get_arrows_tp()
arrowsFp = arrow.get_arrows_fp()
arrowsFn = arrow.get_arrows_fn()
arrowsTn = arrow.get_arrows_tn()

arrowPrec = arrow.get_arrows_precision()
arrowRec = arrow.get_arrows_recall()

# For adjacency matrices
adj = AdjacencyConfusion(true_cpdag, est)

adjTp = adj.get_adj_tp()
adjFp = adj.get_adj_fp()
adjFn = adj.get_adj_fn()
adjTn = adj.get_adj_tn()

adjPrec = adj.get_adj_precision()
adjRec = adj.get_adj_recall()

# Structural Hamming Distance
shd = SHD(true_cpdag, est).get_shd()

print(arrowsTp,arrowsFp,arrowsFn,arrowsTn,arrowPrec,arrowRec,shd)
# # arrowsTp (True Positives, 真正例): 表示正确预测存在的边（或箭头，对于有向图）的数量。即，在真实网络中存在，并且在预测网络中也存在的边。
# # arrowsFp (False Positives, 假正例): 表示在预测网络中存在，但在真实网络中不存在的边。这些是被错误地预测为存在的边。
# # arrowsFn (False Negatives, 假负例): 表示在真实网络中存在，但在预测网络中不存在的边。这些是被错误地预测为不存在的边。
# # arrowsTn (True Negatives, 真负例): 在真实网络和预测网络中都不存在的边的数量。
# # arrowPrec (Precision, 精确率): arrowsTp / (arrowsTp + arrowsFp)，即正确预测的边数与所有预测存在的边数之比。
# # arrowRec (Recall, 召回率): 所有正例样本中被正确预测为正例的比例。arrowsTp / (arrowsTp + arrowsFn)，即正确预测的边数与真实存在的边数之比。