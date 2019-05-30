class MaxHeap(object):
    '''优先队列'''
    def __init__(self, max_size, data=None):
        self.data = data
        self.max_size = max_size
        self.result_list = [None]*max_size
        self.size = 0

    '''判断堆是否已满'''
    @property
    def is_full(self):
        return self.size == self.max_size

    '''加入一个新值'''
    def add(self, value):
        if self.is_full:
            if value < self.result_list[0]:
                self.result_list[0] = value
                self.shift_down(0)
        else:
            self.result_list[self.size] = value
            self.size += 1
            self.shift_up(self.size-1)

    '''推出顶部元素（最大值）'''
    def pop(self):
        assert self.size > 0, 'MaxHeap is empty.'
        r = self.result_list[0]
        self.result_list[0] = self.result_list[self.size-1]
        self.result_list[self.size-1] = None
        self.size -= 1
        self.shift_down(0)
        return r

    def shift_down(self, idx):
        assert idx < self.size, 'index out of range'
        parent = (idx-1)//2
        while parent >= 0 and self.result_list[parent] < self.result_list[idx]:
            self.result_list[parent], self.result_list[idx] = self.result_list[idx], self.result_list[parent]
            idx = parent
            parent = (idx-1)//2

    def shift_up(self, x):
        pass
