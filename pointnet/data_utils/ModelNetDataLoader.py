import numpy as np
import warnings
import os
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')



def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)  # 数据中心化
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))  # m是标准差
    pc = pc / m  # 数据标准化，使新数据接近标准高斯分布
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        point:point_set
        xyz: pointcloud data, [N, D]  N * 6 ??
        npoint: 采样点个数
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    '''
    1024*1
    array([0., 0., 0.])
    '''
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    '''
    numpy.random.randint(low, high=None, size=None, dtype='l')如果默认high=None，则取[0, low)
    返回[0,N)的一个随机数
    '''
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ModelNetDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        """
        :param root:
        :param npoint:
        :param split: 读取train或test
        :param uniform: 对xyz坐标
        :param normal_channel: True:6个channel都要
        :param cache_size:
        """

        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        '''catfile:目录文件'''
        self.cat = [line.rstrip() for line in open(self.catfile)]
        '''
        cat:目录
        ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        '''
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        '''
        zip:Make an iterator that aggregates elements from each of the iterables.
        dict(zip(key, value))创建一个字典
        {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4, 'bottle': 5, 'bowl': 6, 'car': 7, 'chair': 8, 'cone': 9, 'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13, 'dresser': 14, 'flower_pot': 15, 'glass_box': 16, 'guitar': 17, 'keyboard': 18, 'lamp': 19, 'laptop': 20, 'mantel': 21, 'monitor': 22, 'night_stand': 23, 'person': 24, 'piano': 25, 'plant': 26, 'radio': 27, 'range_hood': 28, 'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32, 'table': 33, 'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37, 'wardrobe': 38, 'xbox': 39}
        '''
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        '''
        rstrip:去年字符串右边的空格
        shape_ids[train] = ['airplane_0001',
        'airplane_0002',
        'airplane_0003',
        ...]
        '''

        assert (split == 'train' or split == 'test')
        '''如果条件为真，则程序正常运行'''
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        '''
        s.split(sep,maxsplit)->sep：用于指定分隔符，可以包含多个字符;maxsplit：可选参数，用于指定分割的次数 返回一个list
        'sep'.join(seq)->sep：分隔符,可以为空 seq：要连接的元素序列、字符串、元组、字典 返回一个字符串
        shape_names = ['airplane','airplane',...]
        '''
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        '''
        [
        (vase, "/data/liuwujie/data/modelnet40_normal_resampled/vase/vase_0575.txt")
        (vase, "/data/liuwujie/data/modelnet40_normal_resampled/vase/vase_0574.txt")
        ...
        ]
        '''
        print('The size of %s data is %d'%(split, len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
            '''??'''
        else:
            fn = self.datapath[index]
            '''
            (vase, "/data/liuwujie/data/modelnet40_normal_resampled/vase/vase_0574.txt")
            '''
            cls = self.classes[self.datapath[index][0]]
            '''类别对应的数字'''
            cls = np.array([cls]).astype(np.int32)
            '''cls变成数组'''
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            '''
            读取数据，指定分割符为,
            [
                [0.394500,-0.550000,-0.330900,0.935000,-0.320100,-0.152800]
                [-0.489700,0.756200,0.429300,-0.821500,-0.202200,0.533100]
                ...
            ]
            '''
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]
            '''取前面1024个点'''


            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
            '''对xyz坐标标准化，使其符合高斯分布'''

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)




if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/',split='train', uniform=False, normal_channel=True,)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point,label in DataLoader:
        print(point.shape)
        print(label.shape)