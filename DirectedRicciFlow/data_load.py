# check 3, 关于data_dir这一项，我可能根本没有（不打算找），对应的代码可以删除；args 是不是要改，不然总是NONE
import graph_generate
import os
import networkx as nx
from utils_gamma import *
from functools import partial


class Graph:

    def __init__(self, graph, args=None, data_dir=None):
        self.graph = graph
        if (
            data_dir == None and args["my_graph"] == False
        ):  # 如果没有提供数据目录且不是自定义图
            self.G = getattr(nx, graph, None)(
                *args["settings"]
            )  # 从networkx库中获取对应的图生成方法，并用设置参数（*args["settings"]）生成图
        elif (
            data_dir == None and args["my_graph"] == True
        ):  # 如果没有提供数据目录但是是自定义图
            print("*args[settings]:", args["settings"])
            self.G = getattr(graph_generate, graph)(
                args["settings"]
            )  # 从graph_generate库中获取对应的图生成方法 （调用什么具体方法由 graph name 决定），并用设置参数（*args["settings"]）生成图
        else:  # 如果提供了数据目录 (我现在大概率没有)
            if self.graph == "football":  # 如果图类型是足球比赛
                self.G = nx.read_gml(data_dir)  # 从数据目录中读取gml文件生成图形
            elif self.graph == "facebook":  # 如果图类型是facebook社交网络
                edge_list = [
                    x.strip() for x in open(data_dir, "r").readlines()
                ]  # 读取数据目录中的边列表文件
                self.G = nx.read_edgelist(
                    edge_list, delimiter=" "
                )  # 根据边列表生成图形
                circ_file = args["settings"][0]  # 获取设置参数中的圈子文件信息
                circs = [
                    x.strip()
                    for x in open(
                        os.path.join(os.path.dirname(data_dir), circ_file["circles"])
                    ).readlines()
                ]  # 读取圈子文件
                for circ in circs:  # 遍历每个圈子
                    items = circ.split()  # 将圈子信息分割成列表
                    circ_id = items[0]  # 获取圈子ID
                    circ_nodes = items[1:]  # 获取圈子中的节点
                    for node in circ_nodes:  # 遍历每个节点
                        if node in self.G.nodes():  # 如果节点在图形中
                            self.G.nodes[node][
                                "circle"
                            ] = circ_id  # 将节点的圈子信息存储为节点属性
            else:  # 如果图形类型既不是足球比赛也不是facebook社交网络
                edge_list = [
                    x.strip() for x in open(data_dir, "r").readlines()
                ]  # 读取数据目录中的边列表文件
                self.G = nx.read_edgelist(
                    edge_list, delimiter=","
                )  # 根据边列表生成图形

        n_edges = len(self.G.edges())  # 获取图形中的边数
        weight = {e: 1.0 for e in self.G.edges()}  # 为每条边设置权重为1.0
        nx.set_edge_attributes(self.G, weight, "weight")  # 将权重设置为边的属性
        self.node_colors()  # 调用node_colors方法设置节点颜色
        print(
            "Data loaded. \nNumber of nodes： {}\nNumber of edges: {}".format(
                self.G.number_of_nodes(), self.G.number_of_edges()
            )
        )  # 输出节点和边的数量

    def node_colors(self):
        if self.graph == "karate_club_graph":  # 如果图形类型是karate_club_graph
            for i in self.G.nodes():  # 遍历每个节点
                if self.G.nodes[i]["club"] == "Officer":  # 如果节点所属的社团是Officer
                    self.G.nodes[i]["color"] = "#377eb8"  # 设置节点颜色为蓝色
                else:  # 如果节点所属的社团不是Officer
                    self.G.nodes[i]["color"] = "#ff7f00"  # 设置节点颜色为橙色
        else:
            pass


if __name__ == "__main__":
    graph = Graph("karate_club_graph")  # 创建一个karate_club_graph类型的图形实例
    nx.write_gexf(graph.G, "karate.gexf")  # 将图形写入到karate.gexf文件中
