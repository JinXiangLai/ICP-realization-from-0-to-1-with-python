import numpy as np
from Geometry import Rigid, Correspondence
from matplotlib import pyplot as plt
from typing import List


def GenerateTestedTargetAndSourcePointCloud(point_num: int,
                                            transform: Rigid,
                                            radius: float = 1.0,
                                            add_noise: bool = False):
    '''Here generate two point clouds shaped in ring'''
    inc_step = radius * 2.0 * 2.0 / point_num
    source_pc = np.zeros((3, point_num), dtype=float)
    x = -1.0
    times = 0
    trans = transform.trans.reshape(3, 1)
    while (times < point_num):
        y = np.sqrt(radius - x**2)
        source_pc[:, times] = [x, y, 0]
        times += 1
        source_pc[:, times] = [x, -y, 0]
        x += inc_step
        times += 1

    target_pc = np.zeros((3, point_num), dtype=float)
    if (add_noise):
        noise = np.random.normal(0, 0.01, (3, point_num))
        target_pc = np.matmul(transform.GetRotationMatrix(),
                              source_pc) + trans + noise
    else:
        target_pc = np.matmul(transform.GetRotationMatrix(), source_pc) + trans

    return target_pc, source_pc


def ShowTargetAndSourcePointCloud(self, target_pc: np.ndarray,
                                  source_pc: np.ndarray, win_name: str):
    tar_x, tar_y, tar_z = target_pc[0, :], target_pc[1, :], target_pc[2, :]
    src_x, src_y, src_z = source_pc[0, :], source_pc[1, :], source_pc[2, :]

    fig, ax = plt.subplots(1,
                           1,
                           subplot_kw={
                               'projection': '3d',
                               'aspect': 'auto'
                           })
    ax.scatter(tar_x, tar_y, tar_z, s=1, c='r', zorder=10)
    ax.scatter(src_x, src_y, src_z, s=1, c='b', zorder=10)
    ax.set_title(win_name)
    plt.show()


def ShowCorrespondences(target_pc: np.ndarray, source_pc: np.ndarray,
                        correspondences: List[Correspondence], win_name: str):
    tar_x, tar_y, tar_z = target_pc[0, :], target_pc[1, :], target_pc[2, :]
    src_x, src_y, src_z = source_pc[0, :], source_pc[1, :], source_pc[2, :]

    fig, ax = plt.subplots(1,
                           1,
                           subplot_kw={
                               'projection': '3d',
                               'aspect': 'auto'
                           })
    ax.scatter(tar_x, tar_y, tar_z, s=1, c='r', zorder=10)
    ax.scatter(src_x, src_y, src_z, s=1, c='b', zorder=10)
    ax.set_title(win_name)

    for cor in correspondences:
        ax.plot([cor.target_[0], cor.source_[0]],
                [cor.target_[1], cor.source_[1]],
                [cor.target_[2], cor.source_[2]], '-g')
    plt.show()