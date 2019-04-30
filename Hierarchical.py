# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:31:37 2019

@author: yifan.gong
"""

import math
import numpy as np

class Node(object):
    def __init__(self, center, left=None, right=None, distance=-1, iden=None, count=1, height = 0):
        """
        center: 节点（聚类）的中心
        left: 左节点
        right:  右节点
        distance: 左右节点的距离
        iden: 记录节点的编号
        count: 该聚类内点的数量
        height: 节点的高度
        """
        self.center = center
        self.left = left
        self.right = right
        self.distance = distance
        self.iden = iden
        self.count = count
        self.height = height

def distance(point1: np.ndarray, point2: np.ndarray):
    """
    计算两点之间的距离
    """
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)

class Hierarchical(object):
    def __init__(self, k = 1):
        assert k > 0
        self.k = k
        self.labels = None
      
    def center(self,node1: Node,node2: Node):
        '''
        计算两个聚类聚合后的新中心（中心用于计算聚类间的距离）
        '''
        point_len = len(node1.center)
        new_center = [(node1.center[i]*node1.count + node2.center[i]*node2.count)/(node1.count + node2.count)
                        for i in range(point_len)]      
        return new_center
    
    def fit(self, x):
        nodes = [Node(center=v, iden=i) for i,v in enumerate(x)]
        distances = {}
        point_num, point_len = np.shape(x)
        self.labels = [-1,] * point_num
        currentclustid = -1
        while len(nodes) > self.k:
            min_dist = math.inf
            nodes_len = len(nodes)
            closest_part = None
            for i in range(nodes_len - 1):
                for j in range(i + 1, nodes_len):
                    # 为了不重复计算距离，保存在字典内
                    d_key = (nodes[i].iden, nodes[j].iden)
                    if d_key not in distances.keys():
                        distances[d_key] = distance(nodes[i].center, nodes[j].center)
                    d = distances[d_key]
                    if d < min_dist:
                        min_dist = d
                        closest_part = (i, j)
            # 合并两个聚类
            part1, part2 = closest_part
            node1, node2 = nodes[part1], nodes[part2]
            new_center = self.center(node1,node2)
            new_node = Node(center=new_center,
                                   left=node1,
                                   right=node2,
                                   distance=min_dist,
                                   iden=currentclustid,
                                   count=node1.count + node2.count,
                                   height = max(node1.height,node2.height)+1)
            currentclustid -= 1
            del nodes[part2], nodes[part1]   # 一定要先删索引大的,先删小的会改变后面的索引
            nodes.append(new_node)
        self.nodes = nodes
        self.calc_label()

    def calc_label(self):
        """
        遍历树，根据根节点的索引标记叶子最终所属的聚类
        """
        for i, node in enumerate(self.nodes):
            self.leaf_traversal(node, i)

    def leaf_traversal(self, node: Node, label):
        """
        对树进行遍历
        """
        if node.left == None and node.right == None:
            self.labels[node.iden] = label
        if node.left:
            self.leaf_traversal(node.left, label)
        if node.right:
            self.leaf_traversal(node.right, label)
            

