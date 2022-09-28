from typing import List
import numpy as np
from matplotlib import pyplot as plt
from KdTree import KdTree
from Geometry import *

class GeneralIcp():

    def __init__(self,
                 target_pc: np.ndarray,
                 source_pc: np.ndarray,
                 init_transform: Rigid,
                 max_dis: float = 1.5,
                 debug_mode: bool = False) -> None:
        self.target_pc_ = target_pc
        self.source_pc_ = source_pc
        self.transform_ = init_transform
        self.max_dis = max_dis
        self.debug_mode_ = debug_mode

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

    def ShowCorrespondences(self, target_pc: np.ndarray, source_pc: np.ndarray,
                            correspondences: List[Correspondence],
                            win_name: str):
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

    def FindCorrespondence(self, target_pc: np.ndarray, source_pc: np.ndarray):
        '''
        TODO: violent search is easy by inefficient
        '''
        correspondences = []
        _, col = target_pc.shape
        _, col_src = source_pc.shape

        i = 0
        j = 0
        for i in range(col):
            pt = target_pc[:, i]
            min_dis = 1e10
            cor_index = -1

            for j in range(col_src):
                ps = source_pc[:, j]
                dis = np.linalg.norm(pt - ps)
                if dis < self.max_dis and dis < min_dis:
                    cor_index = j
                    min_dis = dis
            if cor_index >= 0:
                correspondences.append(
                    Correspondence(pt, source_pc[:, cor_index]))

        if (self.debug_mode_):
            self.ShowCorrespondences(target_pc, source_pc, correspondences,
                                     'Show correspondences')
        return correspondences

    def FindCorrespondenceWithKdtree(self, target_pc: np.ndarray,
                                     source_pc: np.ndarray, kd_tree):
        correspondences = []
        _, col = source_pc.shape
        i = 0
        for i in range(col):
            ps = source_pc[:, i]
            # pt = kd_tree.SearchApproxClosestPoint(ps).value_
            pt = kd_tree.SearchClosestPoint(ps).value

            dis = np.linalg.norm(pt - ps)
            if dis < self.max_dis:
                correspondences.append(Correspondence(pt, ps))

        if (self.debug_mode_):
            self.ShowCorrespondences(target_pc, source_pc, correspondences,
                                     'Show correspondences')
        return correspondences

    def CalculateResidualVector(self, correspondences: List[Correspondence]):
        if len(correspondences) < 1:
            print("None correspondence!\n")
            return

        residual_vec = np.zeros((len(correspondences) * 3, 1), dtype=float)
        i = 0
        index = 0
        while i < residual_vec.shape[0]:
            residual_vec[i:i + 3,
                         0] = correspondences[index].CalculateResidual()
            i += 3
            index += 1

        return residual_vec

    def CalculateJacobain(self, correspondences: List[Correspondence],
                          transform: Rigid):
        jacobian = np.zeros((len(correspondences) * 3, 6), dtype=float)
        i = 0
        index = 0
        while i < jacobian.shape[0]:
            partial_derivative = correspondences[index].CalculateJacobian(
                transform)
            jacobian[i:i + 3, 0:6] = partial_derivative
            i += 3
            index += 1

        return jacobian

    def SolveAxEqualtoB(self, transform: Rigid, residual_vec: np.ndarray,
                        jacobian: np.ndarray):
        jacobian_square = np.matmul(jacobian.T, jacobian)
        b = -1 * np.matmul(jacobian.T, residual_vec)
        jacobian_square_inv = np.linalg.inv(jacobian_square)
        delta_x = np.matmul(jacobian_square_inv, b)

        if self.debug_mode_:
            print("delta_x: ", delta_x)
        transform.UpdateTransform(delta_x.reshape(6))

        return transform

    def UpdateSourcePointCloud(self, transform: Rigid, source_pc: np.ndarray):
        _, col = source_pc.shape
        trans = transform.trans_.reshape(3, 1)
        for i in range(col):
            source_pc[:, i] = (np.matmul(transform.GetRotationMatrix(),
                                         source_pc[:, i].reshape(3, 1)) +
                               trans).reshape(3)
        return source_pc

    def RunIteration(self, iterations=30, converge_delta=0.01) -> bool:
        last_transform = self.transform_.Copy()
        kd_tree = KdTree(self.target_pc_, True)
        for i in range(iterations):
            print("Runing iteration %d..." % (i + 1))
            changed_source_pc = self.UpdateSourcePointCloud(
                self.transform_, self.source_pc_.copy())
            if (self.debug_mode_):
                self.ShowTargetAndSourcePointCloud(self.target_pc_,
                                                   changed_source_pc,
                                                   "Iteration %d" % (i))

            #correspondences = self.FindCorrespondence(self.target_pc_, changed_source_pc)
            correspondences = self.FindCorrespondenceWithKdtree(
                self.target_pc_, changed_source_pc, kd_tree)

            if (len(correspondences) < 10):
                print("Correspondences size too small: ", len(correspondences),
                      "\n ICP run failed!")
                return False

            residual_vec = self.CalculateResidualVector(correspondences)

            jacobian = self.CalculateJacobain(correspondences, self.transform_)

            self.SolveAxEqualtoB(self.transform_, residual_vec, jacobian)

            translation_delta = last_transform.CalculateTranslationDelta(
                self.transform_)

            if (translation_delta < converge_delta):
                print("ICP run succeed!\n")
                self.ShowCorrespondences(self.target_pc_, changed_source_pc,
                                         correspondences,
                                         "Correspondence result")
                return True

            last_transform = self.transform_.Copy()

        print("ICP run failed!\n")
        return False
