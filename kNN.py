# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:00:15 2019

@author: yifan.gong
"""
import numpy as np
import math


class MaxHeap(object):
    '''最大堆'''
    def __init__(self, max_size=50, data=None, func=lambda x: x):
        self.max_size = max_size
        self.result_list = [None]*max_size
        self.size = 0
        self.func = func
        self.data = data

    '''判断堆是否已满'''
    @property
    def is_full(self):
        return self.size == self.max_size

    '''加入一个新值,如果队列已满，则替换掉最大的元素'''
    def add(self, item):
        if self.is_full:
            if self.func(item) < self.func(self.result_list[0]):
                self.result_list[0] = item
                self.shift_down(0)
        else:
            self.result_list[self.size] = item
            self.size += 1
            self.shift_up(self.size-1)

    '''推出顶部元素（最大值）'''
    def poptop(self):
        assert self.size > 0, 'MaxHeap is empty.'
        r = deepcopy(self.result_list[0])
        self.result_list[0],self.result_list[self.size-1] = self.result_list[self.size-1],None
        self.size -= 1
        self.shift_down(0)
        return r

    def shift_up(self, idx):
        assert idx < self.size, 'index out of range'
        parent = (idx-1)//2
        while parent >= 0 and self.func(self.result_list[parent]) < self.func(self.result_list[idx]):
            self.result_list[parent], self.result_list[idx] = self.result_list[idx], self.result_list[parent]
            idx = parent
            parent = (idx-1)//2

    def shift_down(self, idx):
        child = 2*(idx+1)-1
        while child < self.max_size and self.result_list[child]:
            if child+1 < self.max_size and \
                self.result_list[child+1] and \
                    self.func(self.result_list[child+1]) > self.func(self.result_list[child]):
                child += 1
            if self.func(self.result_list[idx]) < self.func(self.result_list[child]):
                self.result_list[child], self.result_list[idx] = self.result_list[idx], self.result_list[child]
                idx = child
                child = 2*(idx+1)-1
            else:
                break

    '''生成最大堆：将data中元素依次加入'''
    def fit(self):
        for num in self.data:
            self.add(num)

    '''堆排序：将堆顶元素依次推出'''
    def heapsort(self):
        result = []
        self.fit()
        while self.size > 0:
            r = self.poptop()
            result.append(r)
            print(self.result_list)
        return result


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
            if node.is_leaf():
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
                print(self.near_node.center)
        find_near_node(kdtree)
        return self.near_node


class KNN(object):
    def __init__(self, train_data):
        self.train_data = data
        T = KDTree()
        T.fit(data=train_data)
        self.kdtree = T.tree
        self.near_list = []

    def k_nearest(self, k=1):
        if len(self.train_data) < k:
            mh = MaxHeap(max_size=k, data=self.train_data)
            r = mh.heapsort()
            return r
        



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
    data = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    T = KDTree()
    T.fit(data=data)
    x = (2, 2)
    c = T.nearest(x, T.tree)
    x_nearest = c.center
