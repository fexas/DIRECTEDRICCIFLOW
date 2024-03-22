# check 1
import networkx as nx
from algorithmx import jupyter_canvas
import multiprocessing
from multiprocessing import Pool
import concurrent.futures
import numpy as np
import cvxpy as cvx
import time
import os
import surgery as Surgery
from utils_gamma import *

########################################################################


class DirectGraph:
    def __init__(self, G, weight="weight"):
        self.G = G
        self.weight = weight
        self.lengths = {}  # all pair shortest path dictionary
        self.densities = {}
        self.EPSILON = 1e-7  # to prevent divided by zero
        self.exp_power = 2

    def _get_all_pairs_shortest_path(self):
        # Construct the all pair shortest path lookup
        lengths = dict(nx.all_pairs_dijkstra_path_length(self.G, weight=self.weight))
        return lengths

    def _get_single_node_in_nodes(self, x):
        # 获取每个节点的入边中的节点（不包括给出节点）
        in_edges = self.G.in_edges(x)
        in_nodes = [u for u, v in in_edges if v == x]
        return in_nodes

    def _get_single_node_out_nodes(self, x):
        # 获取每个节点的出边中的节点 （不包括给出节点）
        out_edges = self.G.out_edges(x)
        out_nodes = [v for u, v in out_edges if u == x]
        return out_nodes

    # 计算所有节点的平均转移概率核
    def _get_edge_density_distributions(self):
        densities = dict()

        def Gamma(i, j):
            return self.lengths[i][j]

        # 计算转移概率核
        def _get_probability_transition_kernel(x, out_nodes):

            out_weighted_sum = sum([Gamma(x, pout) for pout in out_nodes])
            probability_transition_kernel = dict()

            for node in self.G.nodes:
                if node in out_nodes:
                    probability_transition_kernel[node] = (
                        Gamma(x, node) / out_weighted_sum
                    )
                elif not out_nodes and node == x:
                    probability_transition_kernel[node] = (
                        1  # 如果 out_nodes 为空，则全部概率1赋至给出节点
                    )
                else:
                    probability_transition_kernel[node] = 0

            return probability_transition_kernel

        # 计算反向转移概率核
        # 这里我尝试的是和理论不同的kernel，对应in—out模式里in的kernel \overleftarrow{P}(y,z)=\frac{w_{zy}}{\sum_{u \rightarrow y}w_{uy}}
        # 理论部分对应的模型应该先要把 m(x) 的值解出来
        def _get_inverse_transition_kernel(x, in_nodes):

            in_weighted_sum = sum([Gamma(pin, x) for pin in in_nodes])
            inverse_transition_kernel = dict()

            for node in self.G.nodes:
                if node in in_nodes:
                    inverse_transition_kernel[node] = Gamma(node, x) / in_weighted_sum
                elif not in_nodes and node == x:
                    inverse_transition_kernel[node] = (
                        1  # 如果 in_nodes 为空，则全部概率1赋至给出节点
                    )
                else:
                    inverse_transition_kernel[node] = 0

            return inverse_transition_kernel

        # 计算平均概率核(所有节点)
        def _get_single_node_mean_transition_kernel(x):
            in_nodes = self._get_single_node_in_nodes(x)
            out_nodes = self._get_single_node_out_nodes(x)

            probability_transisiton_kernel = _get_probability_transition_kernel(
                x, out_nodes
            )
            inverse_transition_kernel = _get_inverse_transition_kernel(x, in_nodes)

            mean_transition_kernel = [
                (probability_transisiton_kernel[node] + inverse_transition_kernel[node])
                / 2
                for node in self.G.nodes
            ]

            return mean_transition_kernel

        for x in self.G.nodes():
            densities[x] = _get_single_node_mean_transition_kernel(x)

        return densities

    # 计算源节点和汇节点的概率分布
    def _distribute_densities(self, source, target):

        x = self.densities[source]
        y = self.densities[target]

        # 计算距离矩阵
        size = len(self.G.nodes)
        d = np.full((size, size), np.inf)

        # 遍历self.G.nodes列表中的每个节点，i是节点在列表中的索引，src是节点本身。
        for i, src in enumerate(self.G.nodes):
            for j, dst in enumerate(self.G.nodes):
                assert (
                    dst in self.lengths[src]
                ), "Target node not in list, should not happened, pair (%d, %d)" % (
                    src,
                    dst,
                )
                d[i][j] = self.lengths[src][dst]

        x = np.array([x]).T
        y = np.array([y]).T

        return x, y, d

    # 计算 *-ccoupling based Wasserstein 距离 (计算 *-coupling based curvature)
    def _optimal_transportation_distance(self, source, target, x, y, d):

        size = len(self.G.nodes)
        star_coupling = cvx.Variable((size, size))  # 输运方案 B(x,y)
        # 目标函数 sum(star_coupling(x,y) * d(x,y)) , 逐项相乘
        obj = cvx.Maximize(cvx.sum(cvx.multiply(star_coupling, d)))

        # 约束
        constrains = [cvx.sum(star_coupling) == 0]

        iterative_array = list(range(len(self.G.nodes)))
        for u in iterative_array:
            if u != source:
                constrains += [
                    cvx.sum(star_coupling[u, :], axis=0, keepdims=True) == -x[u]
                ]

        for v in iterative_array:
            if v != target:
                constrains += [
                    cvx.sum(star_coupling[:, v], axis=0, keepdims=True) == -y[v]
                ]

        #  2 >= B(source,target) >= 0
        constrains += [
            0 <= star_coupling[source, target],
            star_coupling[source, target] <= 2,
        ]  # 这里为什么是小于2

        # 其他情况 B(u,v) <= 0
        for u in iterative_array:
            for v in iterative_array:
                if u != source or v != target:
                    constrains += [star_coupling[u, v] <= 0]

        # 定义优化问题
        prob = cvx.Problem(obj, constrains)
        # 求解优化问题
        m = prob.solve(solver="ECOS")  # change solver here if you want

        return m

    # 计算 ricci curvature
    def _compute_ricci_curvature_single_edge(self, source, target):

        # 防止自循环
        assert source != target, "Self loop is not allowed."

        # 如果边权重过小, 则返回 0
        if self.lengths[source][target] < self.EPSILON:
            print(
                "Zero weight edge detected for edge (%s,%s), return Ricci Curvature as 0 instead."
                % (source, target)
            )
            return {(source, target): 0}

        # 计算 *- coupling based Wasserstein distance
        m = 1  # 初始化

        x, y, d = self._distribute_densities(source, target)
        m = self._optimal_transportation_distance(source, target, x, y, d)

        # 计算 Ricci curvature: k = k* = (m_{x,y})/d(x,y)
        result = m / self.lengths[source][target]  # 除以分母 d(x, y)
        # print("Ricci curvature (%s,%s) = %f" % (source, target, result))

        return {(source, target): result}

    def _wrap_compute_single_edge(self, stuff):
        return self._compute_ricci_curvature_single_edge(*stuff)

    def compute_ricci_curvature_edges(self, edge_list=None):
        # 如果 edge_list 为空，返回空列表
        if not edge_list:
            edge_list = []

        # 计算所有节点对的最短路径字典
        if not self.lengths:
            self.lengths = self._get_all_pairs_shortest_path()

        # 计算所有节点对应的密度分布--mean transistion kernel（如果尚未构建）
        if not self.densities:
            self.densities = self._get_edge_density_distributions()

        # 多线程并进
        proc = os.cpu_count()
        pool = Pool(processes=proc)

        # 计算所有边上的 Ricci curvature
        args = [(source, target) for source, target in edge_list]

        # 计算每条边的 Ricci 曲率 (多线程版)
        result = pool.map_async(self._wrap_compute_single_edge, args).get()
        pool.close()
        pool.join()

        return result

    def compute_ricci_curvature(self):
        # 检查图self.G中的边是否具有权重属性。如果没有权重属性，它会为所有边分配默认权重1.0
        if not nx.get_edge_attributes(self.G, self.weight):
            print('Edge weight not detected in graph, use "weight" as edge weight.')
            for v1, v2 in self.G.edges():
                self.G[v1][v2][self.weight] = 1.0

        # 计算每条边上的 Ricci curvature
        edge_ricci = self.compute_ricci_curvature_edges(self.G.edges())

        # 将计算出的边Ricci曲率从 edge_ricci 分配给图 self.G 中的相应边
        for rc in edge_ricci:
            for k in list(rc.keys()):
                source, target = k
                self.G[source][target]["ricciCurvature"] = rc[k]

    # ricci flow process
    def compute_ricci_flow(
        self,
        iterations=100,
        step=0.01,
        delta=1e-6,
        surgery={"name": "surgery", "portion": 0.25, "interval": 99},
        save_gexf_dir=None,
    ):
        if not nx.is_strongly_connected(self.G):

            print(
                "No strongly connected graph detected, compute on the largest strongly connected component instead."
            )

            self.G = nx.Graph(
                max(
                    [
                        self.G.subgraph(scc)
                        for scc in nx.strongly_connected_components(self.G)
                    ],
                    key=len,
                )
            )

            # 重新编号（从0开始）
            new_labels = {node: i for i, node in enumerate(self.G.nodes())}
            self.G = nx.relabel_nodes(self.G, new_labels)

            print("---------------------------")
            print(nx.info(self.G))

        self.G.remove_edges_from(nx.selfloop_edges(self.G))  # 除去自循环的边

        # 保存原始图
        nx.write_gexf(self.G, os.path.join(save_gexf_dir, "origin.gexf"))

        # 计算 Ricci flow
        t0 = time.time()

        if nx.get_edge_attributes(self.G, "original_RC"):
            print("original_RC detected, continue to refine the ricci flow.")
        else:
            self.compute_ricci_curvature()

            for v1, v2 in self.G.edges():
                self.G[v1][v2]["original_RC"] = self.G[v1][v2]["ricciCurvature"]

        # 开始 Ricci flow process (100次)
        self.rc_diff = []
        for i in range(iterations):

            # 存贮当前图
            nx.write_gexf(self.G, os.path.join(save_gexf_dir, "%d.gexf" % i))

            # 计算 \sum_{h \in E(G)}\kappa_hw_h
            sum_K_W = sum(
                self.G[v1][v2]["ricciCurvature"] * self.G[v1][v2][self.weight]
                for (v1, v2) in self.G.edges()
            )

            # 对边的权重进行归一化 (weighted sum normailzed to 1) -- 其实这个程序只需要跑一遍就可以了
            w = nx.get_edge_attributes(self.G, self.weight)
            sumw = sum(w.values())
            for k, v in w.items():
                w[k] = w[k] / sumw
                if w[k] < 0:
                    w[k] = min([self.EPSILON, -w[k]])
                # assert(w[k]>0)
            nx.set_edge_attributes(self.G, w, self.weight)

            # Ricci flow process：计算 w_e^{i+1} =  w_e^i s + (-\kappa_e^iw_e^i + w_e^i \sum_{h \in E(G)}\kappa_h^i w_h^i )
            for v1, v2 in self.G.edges():
                self.G[v1][v2][self.weight] *= 1.0 + step * (
                    sum_K_W - self.G[v1][v2]["ricciCurvature"]
                )

            # 如果两个点过于接近，则将两点合并
            G1 = self.G.copy()
            merged = True
            while merged:
                merged = False
                for v1, v2 in G1.edges():
                    if G1[v1][v2][self.weight] < delta * 10:
                        G1 = nx.contracted_edge(G1, (v1, v2), self_loops=False)
                        merged = True
                        print("Contracted edge: (%d, %d)" % (v1, v2))
                        break
            self.G = G1

            self.compute_ricci_curvature()
            print("=== Ricciflow iteration % d ===" % i)

            rc = nx.get_edge_attributes(self.G, "ricciCurvature")
            diff = max(rc.values()) - min(rc.values())

            print("Ricci curvature difference: %f" % diff)
            print(
                "max:%f, min:%f | maxw:%f, minw:%f"
                % (max(rc.values()), min(rc.values()), max(w.values()), min(w.values()))
            )

            # 如果所有的 curvature 相等 1/ ｜E｜，则 sum_K_W = (\sum_{h \in E(G)} w_h^i)/|E| = 1/|E|
            # w_{e}^{i+1}-w_{e}^{i} = - w_e^i/|E| + w_e^i/|E| = 0 , 没有变化，故有 ricci flow process 没有任何作用，故直接终止程序
            if diff < delta:
                print("Ricci curvature converged, process terminated.")
                break

            # 取出多余边 （community detection）
            # 这里是除去边的操作，我只在最后一次用到, 即 do_surgery = iteration - 1
            # surgery={"name": "surgery", "portion": 0.02, "interval": 99}
            # portion -- 指去除掉权重前0.02的边
            # interval -- 每隔 interval 次做一次surgery (这里我就做了一次surgery)
            surgery_func = surgery["name"]
            do_surgery = surgery["interval"]
            portion = surgery["portion"]
            if i != 0 and i % do_surgery == 0:
                self.G = getattr(Surgery, surgery_func)(self.G, self.weight, portion)

            # 因为上述surgery使得图结构发生改变，故
            # 之后执行 compute_ricci_curvature -> compute_ricci_curvature_edges 会重新算一次 densities
            self.densities = {}

        nx.write_gexf(self.G, os.path.join(save_gexf_dir, "%d.gexf" % iterations))

        print("\n%8f secs for Ricci flow computation." % (time.time() - t0))
