import pandas as pd
import numpy as np
from tigramite import data_processing as pp
from tigramite import independence_tests, pcmci
import warnings
warnings.filterwarnings('ignore')

AD_group = pd.read_csv('AD_group.csv')

variables = [
    'AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4', 'CDRSB', 'ADAS11', 'ADAS13', 'ADASQ4',
    'MMSE', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform',
    'MidTemp', 'ICV', 'mPACCdigit', 'mPACCtrailsB', 'ABETA.bl', 'TAU.bl', 'PTAU.bl'
]
data = AD_group[variables].copy()

# 对分类变量（PTGENDER）进行数值编码
data['PTGENDER'] = data['PTGENDER'].map({'Male': 0, 'Female': 1})

data = data.dropna().reset_index(drop=True)

data_array = data.values
var_names = list(data.columns)

dataframe = pp.DataFrame(data_array, var_names=var_names)

from tigramite.independence_tests.parcorr import ParCorr
parcorr = ParCorr(significance='analytic')

pcmci_obj = pcmci.PCMCI(
    dataframe=dataframe,
    cond_ind_test=parcorr,
)

results = pcmci_obj.run_pcmci(tau_max=1, pc_alpha=0.05)

q_matrix = pcmci_obj.get_corrected_pvalues(results['p_matrix'], fdr_method='fdr_bh')

significant_links_dict = {0: [], 1: []}  # lag 0, lag 1

alpha_level = 0.05

for lag in [0,1]:  # lag=0, lag=1
    for i in range(q_matrix.shape[1]):  # from node i
        for j in range(q_matrix.shape[2]):  # to node j
            if q_matrix[lag, i, j] < alpha_level:
                significant_links_dict[lag].append((i, j))

print("\n发现的显著因果关系（包括时间滞后）:")
for lag, links in significant_links_dict.items():
    for (i, j) in links:
        print(f"{var_names[i]} --> {var_names[j]} (lag {lag})")


###画图
import networkx as nx
import matplotlib.pyplot as plt

# 只画lag=0的连接，代表同一时间点的因果关系
G = nx.DiGraph()

G.add_nodes_from(var_names)

for (i, j) in significant_links_dict[0]:
    G.add_edge(var_names[i], var_names[j])

plt.figure(figsize=(15, 10))
pos = nx.spring_layout(G, k=0.5, seed=42)
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800)
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=10, font_family='Arial')

plt.title('PCMCI (lag=0)', fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()


