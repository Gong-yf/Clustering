# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:00:15 2019

@author: yifan.gong
"""
import numpy as np
import math
from MaxHeap import MaxHeap


class Node(object):
    def __init__(self, center=None, left=None, right=None, father=None, feature=None, flag=None):
        self.center = center
        self.father = father
        self.left = left
        self.right = right
        self.feature = feature
        self.flag = flag

    @property
    def is_leaf(self):
        return 0 if self.left or self.right else 1


class KDTree(object):
    '''生成kd树'''
    def __init__(self):
        self.tree = None

    def _choose_feature(self, data):
        '''选取方差最大的维度（特征）作为划分维度'''
        max_var = 0
        for i, l in enumerate(data.T):
            if np.var(l) > max_var:
                max_var = np.var(l)
                max_index = i
        return max_index

    def _split_feature(self, data, split_feature):
        '''在选定的维度上进行划分,首先对该维度排序，找出中间的点'''
        point_num = np.shape(data)[0]
        col = [(i, data[i, split_feature]) for i in range(point_num)]
        sorted_col = sorted(col, key=lambda x: x[1])
        k = point_num//2
        median_index = sorted_col[k][0]
        median_flag = sorted_col[k][1]
        '''保存中间点数据用于后续定义节点，将其他数据分配到左右两部分'''
        datapoint = data[median_index]
        data = np.delete(data, median_index, axis=0)
        left = []
        right = []
        for i, l in enumerate(data):
            if l[split_feature] < median_flag:
                left.append(l)
            else:
                right.append(l)
        return left, right, median_index, median_flag, datapoint

    def fit(self, data=None):
        root = self._fit(data=data)
        self.tree = root

    def _fit(self, root=None, data=None, father=None):
        '''递归生成树'''
        root, left_data, right_data = self._fit_tree(data, father)
        father = root
        if len(left_data) != 0:
            root.left = self._fit(root.left, left_data, father)
        if len(right_data) != 0:
            root.right = self._fit(root.right, right_data, father)
        return root

    def _fit_tree(self, data, father):
        '''生成一个节点，并将数据根据分类标准分配到左右两部分'''
        if len(data) == 1:
            return Node(center=data[0], father=father), [], []
        else:
            feature = self._choose_feature(data)
            left_data, right_data, median_index, median_flag, datapoint = self._split_feature(data, feature)
            r = Node(center=datapoint, father=father, feature=feature, flag=median_flag)
        return r, np.array(left_data), np.array(right_data)

    def nearest(self, x, kdtree):
        '''在给定kd树中寻找与x最近的点'''
        self.near_node = None
        self.near_distance = math.inf

        def find_near_node(node):
            if node.is_leaf:
                self.near_node = node
                self.near_distance = _distance(x, node.center)
                return
            if node is not None:
                distance_feature = x[node.feature] - node.flag
                find_near_node(node.left if distance_feature < 0 else node.right)
                check_distance = _distance(x, node.center)
                if self.near_node is None or self.near_distance > check_distance:
                    self.near_node = node
                    self.near_distance = check_distance
                if self.near_distance > abs(distance_feature):
                    find_near_node(node.right if distance_feature < 0 else node.left)
        find_near_node(kdtree)
        return self.near_node


class KNN(object):
    def __init__(self, train_data):
        self.train_data = train_data
        T = KDTree()
        T.fit(data=train_data)
        self.kdtree = T.tree
        self.near_list = []

    def _k_nearest(self, x, k=1):
        assert len(self.train_data) > k, 'k is larger than the size of data.'
        mh = MaxHeap(max_size=k, func=lambda x: x[1])

        def find_near_node(node):
            if node is not None:
                if node.is_leaf:
                    mh.add((node.center, _distance(x, node.center)))
                    return
                distance_feature = x[node.feature] - node.flag
                find_near_node(node.left if distance_feature < 0 else node.right)
                check_distance = _distance(x, node.center)
                near_distance = mh.result_list[0][1]
                if (not mh.is_full) or near_distance > check_distance:
                    mh.add((node.center, check_distance))
                if (not mh.is_full) or near_distance > abs(distance_feature):
                    find_near_node(node.right if distance_feature < 0 else node.left)
        find_near_node(self.kdtree)
        self.near_list = mh.result_list

    def k_nearest(self, x, k=1):
        self._k_nearest(x, k=k)
        return self.near_list


def pre_order(node):
    if node is not None:
        print(node.center, node.feature)
        pre_order(node.left)
        pre_order(node.right)


def _distance(x, y):
    '''计算两点之间的距离,欧氏距离'''
    distance = 0.0
    for a, b in zip(x, y):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    data = 10*np.random.randn(100, 2)
    fig1 = plt.figure(figsize=(8, 16))
    ax1 = fig1.add_subplot(211)
    ax1.scatter(data[:, 0], data[:, 1])

    x = np.random.randn(1, 2)[0]
    knn = KNN(data)
    k_near = knn.k_nearest(x, k=5)
    k_near_point = np.array([v[0] for v in k_near])
    ax2 = fig1.add_subplot(212)
    ax2.scatter(data[:, 0], data[:, 1])
    ax2.scatter(k_near_point[:, 0], k_near_point[:, 1], c='yellow')
    ax2.scatter(x[0], x[1], c='red')
