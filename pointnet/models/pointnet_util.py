import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;

    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]

    先随机初始化一个centroids矩阵，后面用于存储npoint个采样点的索引位置，大小为 B × n p o i n t B\times npoint B×npoint，其中B为BatchSize的个数，即B个样本;
    利用distance矩阵记录某个样本中所有点到某一个点的距离，初始化为 B × N B\times N B×N矩阵，初值给个比较大的值，后面会迭代更新;
    利用farthest表示当前最远的点，也是随机初始化，范围为0~N，初始化B个，对应到每个样本都随机有一个初始最远点；
    batch_indices初始化为0~(B-1)的数组；
    直到采样点达到npoint，否则进行如下迭代：
    设当前的采样点centroids为当前的最远点farthest；
    取出这个中心点centroid的坐标；
    求出所有点到这个farthest点的欧式距离，存在dist矩阵中；
    建立一个mask，如果dist中的元素小于distance矩阵中保存的距离值，则更新distance中的对应值，随着迭代的继续distance矩阵中的值会慢慢变小，其相当于记录着某个样本中每个点距离所有已出现的采样点的最小距离；
    最后从distance矩阵取出最远的点为farthest，继续下一轮迭代
    """

    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        # 更新第i个最远点
        centroids[:, i] = farthest
        # 取出这个最远点的xyz坐标
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # 计算点集中的所有点到这个最远点的欧式距离
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # 更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
        mask = dist < distance
        distance[mask] = dist[mask]
        # 从更新后的distances矩阵中找出距离最远的点，作为最远点用于下一轮迭代
        farthest = torch.max(distance, -1)[1]
        '''
        torch.max返回两个tensor，后一个是索引
        '''
    return centroids


def index_points(points, idx):
    """
    按照输入的点云数据和索引返回由索引的点云数据。
    例如points为 B × 2048 × 3 的点云，idx为 [ 1 , 333 , 1000 , 2000 ] ，
    则返回B个样本中每个样本的第1,333,1000,2000个点组成的 B × 4 × 3的点云集。当然如果idx为一个 [ B , D 1 , . . . D N ] 维度的，
    则它会按照idx中的维度结构将其提取成 [ B , D 1 , . . . D N , C ]
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    '''list[B, S]'''
    view_shape[1:] = [1] * (len(view_shape) - 1)
    '''list[B, 1]'''
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    '''list[1, S]'''
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    '''
    B * S
    tensor([[0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5],
        [6, 6, 6],
        [7, 7, 7],
        [8, 8, 8],
        [9, 9, 9]])
    '''
    new_points = points[batch_indices, idx, :]
    '''???'''
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    '''
    B * S * N
    tensor([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]])
    '''
    # sqrdists: [B, S, N] 记录中心点与所有点之间的欧几里德距离
    sqrdists = square_distance(new_xyz, xyz)
    '''B * S * N'''
    group_idx[sqrdists > radius ** 2] = N
    '''判断为True的置为N'''
    # 做升序排列，前面大于radius^2的都是N，会是最大值，所以会直接在剩下的点中取出前nsample个点
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # 考虑到有可能前nsample个点中也有被赋值为N的点（即球形区域内不足nsample个点），这种点需要舍弃，直接用第一个点来代替即可
    # group_first: [B, S, k]， 实际就是把group_idx中的第一个点的值复制为了[B, S, K]的维度，便利于后面的替换
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # 找到group_idx中值等于N的点
    mask = group_idx == N
    # 将这些点的值替换为第一个点的值
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    先用farthest_point_sample函数实现最远点采样FPS得到采样点的索引，再通过index_points将这些点的从原始点中挑出来，作为new_xyz
    利用query_ball_point和index_points将原始点云通过new_xyz 作为中心分为npoint个球形区域,其中每个区域有nsample个采样点
    每个区域的点减去区域的中心值
    如果每个点上面有新的特征的维度，则用新的特征与旧的特征拼接，否则直接返回旧的特征

    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)   # new_xyz作为中心划分球形区域
    torch.cuda.empty_cache()
    # idx:[B, npoint, nsample] 代表npoint个球形区域中每个区域的nsample个采样点的索引
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    # 如果每个点上面有新的特征的维度，则用新的特征与旧的特征拼接，否则直接返回旧的特征
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    直接将所有点作为一个group
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        """
        :param npoint:Number of point for FPS sampling
        :param radius:Radius for ball query
        :param nsample:Number of point for each ball query
        :param in_channel:
        :param mlp:A list for mlp input-output channel, such as [64, 64, 128]
        :param group_all:bool type for group_all or not
        """
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        '''
        ModuleList(
          (0): Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1))
          (1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
          (2): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
         )
         ModuleList(
          (0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         )
        '''
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]

        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, K,npoint]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):    # Multi-Scale Grouping
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        """
        multi-scale:如果有三个尺度那么最后每个球体就有三行特征
        Input:
            npoint: Number of point for FPS sampling
            radius: Radius for ball query
            nsample: Number of point for each ball query
            in_channel: the dimension of channel
            mlp: A list for mlp input-output channel, such as
            group_all: bool type for group_all or not
        """
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()   # batchnorm
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel
            # last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
        '''
        ModuleList(
          (0): ModuleList(
            (0): Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
            (2): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): ModuleList(
            (0): Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
            (2): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
          )
          (2): ModuleList(
            (0): Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
            (2): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        '''

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        '''将不同半径下的点点云特征保存在new_points_list中，再最后拼接到一起'''
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

