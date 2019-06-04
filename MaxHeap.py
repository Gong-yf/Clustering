# -*- coding: utf-8 -*-
"""
@author: yifan.gong
"""


from copy import deepcopy

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


if __name__ == '__main__':
    import random

    data = [random.randint(1, 100) for i in range(31)]
    mh = MaxHeap(50, data)
    sorted_list = mh.heapsort()
