# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:00:15 2019

@author: yifan.gong
"""
import numpy as np
import math


class Node(object):
    def __init__(self, center=None, left=None, right=None, father=None, feature=None, flag=None):
        self.center = center
        self.father = father
        self.left = left
        self.right = right
        self.feature = feature
        self.flag = flag

    def brother(self):
        if self.father is None:
            bro = None
        elif self.father.left is self:
            bro = self.father.right
        else:
            bro = self.father.left
        return bro


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


def pre_order(node):
    if node is not None:
        print(node.center, node.feature)
        pre_order(node.left)
        pre_order(node.right)


def _find_leaf(x, kdtree, search_path=[]):
    if kdtree.feature is not None:
        search_path.append(kdtree)
        if x[kdtree.feature] < kdtree.flag:
            if kdtree.left:
                p, search_path = _find_leaf(x, kdtree.left, search_path)
            else:
                p = kdtree
                search_path.append(kdtree)
        else:
            if kdtree.right:
                p, search_path = _find_leaf(x, kdtree.right, search_path)
            else:
                p = kdtree
                search_path.append(kdtree)
    else:
        p = kdtree
        search_path.append(kdtree)
        return p, search_path
    return p, search_path


def nearest(x, kdtree):
    '''在给定kd树中寻找与x最近的点'''
    def find_near_node(node):
        if node is not None:
            leaf, search_path = _find_leaf(x, kdtree)
            near_node = leaf
            while(search_path):
                finish_node = search_path[-1]
                search_path = search_path[:-1]
                near_distance = _distance(x, near_node.center)
                check_node = search_path[-1]
                check_distance = _distance(x, check_node.center)
                if check_distance < near_distance:
                    near_node = check_node
                    near_distance = check_distance
                distance_feature = x[check_node.feature]-check_node.flag
                if near_distance > abs(distance_feature):
                    if check_node.left is finish_node:
                        find_near_node(check_node.right)
                    else:
                        find_near_node(check_node.left)
        return near_node
    near_node = find_near_node(kdtree)
    return near_node


def _distance(x, y):
    '''计算两点之间的距离,欧氏距离'''
    distance = 0.0
    for a, b in zip(x, y):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)


class KNN(object):
    def __init__(self, train_data):
        T = KDTree()
        T.fit(data=train_data)
        self.kdtree = T.tree


if __name__ == '__main__':
    data = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    T = KDTree()
    T.fit(data=data)
    a = _find_leaf((2, 4.5), T.tree)
    c = nearest((2, 4.5), T.tree)
