# check 4 图生成过程要改，至少要保证有向的强连通部分足够大
import networkx as nx
import itertools
import json
import numpy as np


def Gab(a, b):
    G = nx.Graph()
    a = a + 1
    b = b + 1
    for i in range(b):
        edges = itertools.combinations(
            range(i * a, (i + 1) * a), 2
        )  # 生成一个范围为[i*a, (i+1)*a)]的整数序列，并从中选取两两组合的边
        G.add_edges_from(edges)
    G.add_edges_from(itertools.combinations(range(0, a * b, a), 2))
    return G


# Stochastic Block Model
# sizes - 不同 Block 的大小，如：3个 BLock，sizes = [75, 75, 300]
# probs - Block 的借点之间有边的概率 如： 3个 Block， probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]
def SBM(dic):

    sizes = dic["sizes"]
    probs = dic["probs"]

    G = nx.stochastic_block_model(
        sizes, probs, directed=True, selfloops=False
    )  # 根据sizes和probs创建随机块模型图
    return G


def LFR(dic):
    n = dic["n"]
    tau1 = dic["tau1"]
    tau2 = dic["tau2"]
    mu = dic["mu"]
    average_degree = dic["average_degree"]
    G = nx.generators.community.LFR_benchmark_graph(
        n, tau1, tau2, mu, average_degree, seed=0
    )
    communities = nx.get_node_attributes(G, "community")
    nx.set_node_attributes(G, {n: None for n in G.nodes()}, "community_idx")
    index = 0
    for node in G.nodes():
        if G.nodes[node]["community_idx"] == None:
            for c_node in G.nodes[node]["community"]:
                G.nodes[c_node]["community_idx"] = index
                del G.nodes[c_node]["community"]
            index += 1
        else:
            continue
    # print(nx.get_node_attributes(G, 'community_idx'))
    # print(nx.get_node_attributes(G, 'community'))
    # nx.write_gexf(G, "hhh.gexf")
    return G


# def football(dic):

# 定义主函数，用于测试LFR函数
if __name__ == "__main__":
    G = LFR({"n": 100, "tau1": 3, "tau2": 1.5, "mu": 0.5, "average_degree": 10})
    # print(G.nodes[15]["community"])
