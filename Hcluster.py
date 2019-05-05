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

class Hcluster(object):
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
            #循环寻找距离最短的类
            for i in range(nodes_len - 1):
                for j in range(i + 1, nodes_len):
                    d_key = (nodes[i].iden, nodes[j].iden)
                    if d_key not in distances.keys():
                        distances[d_key] = distance(nodes[i].center, nodes[j].center)
                    d = distances[d_key]
                    if d < min_dist:
                        min_dist = d
                        closest_part = (i, j)
            #合并聚类
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
            del nodes[part2], nodes[part1]
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
            

  #%%  
if __name__ == '__main__':
    import random
    from matplotlib import pyplot as plt
    import datetime
    
    t1 = datetime.datetime.now()
    
    n = 50
    n = n//2
    X = [random.random()*10 for i in range(n)]
    Y1 = [random.uniform(0,2)*val+random.uniform(5,30) for val in X]+[random.uniform(-2,0)*val+random.uniform(-20,5) for val in X]
    Y2 = [random.uniform(-2,0)*val+random.uniform(-30,5) for val in X]+[random.uniform(0,2)*val+random.uniform(5,20) for val in X]
    X1 = [random.random()*10 for i in range(n)]+[random.random()*10+10 for i in range(n)]
    X2 = [random.random()*10 for i in range(n)]+[random.random()*10-10 for i in range(n)]
    X = X1+X2
    Y = Y1+Y2
    data = np.array([(x,y) for x,y in zip(X,Y)])
    
    fig1 = plt.figure(figsize = (2,6),dpi = 200)
    ax1 = fig1.add_subplot(311)
#    ax0.scatter(X,Y,edgecolor='none',s = 5)
    ax1.scatter(X1,Y1,edgecolor='none',s = 5)
    ax1.scatter(X2,Y2,edgecolor='none',s = 5)
    
    k = 4
    H = Hcluster(k=k)
    H.fit(data)
    datafig = [[] for i in range(k)]
    for i,d in enumerate(data):
        datafig[H.labels[i]].append(d)
       
    ax2 = fig1.add_subplot(312)
    for i in range(k):
        d = np.array(datafig[i])
        ax2.scatter(d[:,0],d[:,1],edgecolor='none',s = 5)
        
