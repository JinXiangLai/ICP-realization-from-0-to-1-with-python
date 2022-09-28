import numpy as np
from Geometry import Rigid


def GenerateTestedTargetAndSourcePointCloud(point_num: int,
                                            transform: Rigid,
                                            radius: float = 1.0,
                                            add_noise: bool = False):
    '''Here generate two point clouds shaped in ring'''
    inc_step = radius * 2.0 * 2.0 / point_num
    source_pc = np.zeros((3, point_num), dtype=float)
    x = -1.0
    times = 0
    trans = transform.trans_.reshape(3, 1)
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
