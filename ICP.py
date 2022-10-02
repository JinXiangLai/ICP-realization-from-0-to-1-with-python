from typing import List
import numpy as np
from KdTree import KdTree
from Geometry import *
from Utils import *


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

    def FindCorrespondenceWithKdtree(self, source_pc: np.ndarray,
                                     kd_tree: KdTree):
        correspondences = []
        _, col = source_pc.shape
        i = 0

        for i in range(col):
            ps = source_pc[:, i]
            # pt = kd_tree.SearchApproxClosestPoint(ps).value
            pt = kd_tree.SearchClosestPoint(ps).value

            dis = np.linalg.norm(pt - ps)
            if dis < self.max_dis:
                correspondences.append(Correspondence(pt, ps))

        return correspondences

    def CalculateResidualVector(
            self, correspondences: List[Correspondence]) -> np.ndarray:
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
                          transform: Rigid) -> np.ndarray:
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

    def SolveAxEqualtoB(self, b: np.ndarray,
                        jacobian_square: np.ndarray) -> np.array:

        jacobian_square_inv = np.linalg.inv(jacobian_square)
        delta_x = np.matmul(jacobian_square_inv, b)

        return delta_x.reshape(6)

    def UpdateSourcePointCloud(self, transform: Rigid,
                               source_pc: np.ndarray) -> np.ndarray:
        _, col = source_pc.shape
        trans = transform.trans.reshape(3, 1)

        for i in range(col):
            source_pc[:, i] = (np.matmul(transform.GetRotationMatrix(),
                                         source_pc[:, i].reshape(3, 1)) +
                               trans).reshape(3)
        return source_pc

    def RunIteration(self,
                     max_iterations=30,
                     converge_delta=0.001,
                     use_lm: bool = False) -> bool:
        if use_lm:
            self.IterationWithLMOptimizer(max_iterations, converge_delta)
        else:
            return self.IterationWithGNoptimizer(max_iterations,
                                                 converge_delta)

    def IterationWithGNoptimizer(self,
                                 max_iterations: int = 30,
                                 converge_trans: float = 0.001) -> bool:
        last_transform = self.transform_.Copy()
        kd_tree = KdTree(self.target_pc_, True)

        for i in range(max_iterations):
            print("Runing iteration %d..." % (i + 1))

            changed_source_pc = self.UpdateSourcePointCloud(
                self.transform_, self.source_pc_.copy())
            correspondences = self.FindCorrespondenceWithKdtree(
                changed_source_pc, kd_tree)

            if (self.debug_mode_):
                ShowCorrespondences(self.target_pc_, changed_source_pc,
                                    correspondences, 'Show correspondences')

            if (len(correspondences) < 10):
                print("Correspondences size too small: ", len(correspondences),
                      "\n ICP run failed!")

                return False

            residual_vec = self.CalculateResidualVector(correspondences)
            jacobian = self.CalculateJacobain(correspondences, self.transform_)

            jacobian_square = np.matmul(jacobian.T, jacobian)
            b = -1 * np.matmul(jacobian.T, residual_vec)
            pose_delta = self.SolveAxEqualtoB(b, jacobian_square)

            self.transform_.UpdateTransform(pose_delta)

            trans_delta = last_transform.CalculateTranslationDelta(
                self.transform_)
            F_x = np.linalg.norm(residual_vec) / (residual_vec.shape[0] / 3)

            print('trans_delta = ', trans_delta, ' F_x=', F_x)

            if (trans_delta < converge_trans) or F_x < 0.0001:
                print("ICP run succeed!\n")
                ShowCorrespondences(self.target_pc_, changed_source_pc,
                                    correspondences, "Correspondence result")
                return True

            last_transform = self.transform_.Copy()

        print("ICP run failed!\n")
        return False

    def IterationWithLMOptimizer(self,
                                 max_iterations=30,
                                 converge_trans: float = 0.001) -> bool:
        # Initialize the first time  update parameters
        last_transform = self.transform_.Copy()
        kd_tree = KdTree(self.target_pc_, True)
        changed_source_pc = self.UpdateSourcePointCloud(
            self.transform_, self.source_pc_.copy())
        # find out the correspondences for calculating pose_delta
        correspondences = self.FindCorrespondenceWithKdtree(
            changed_source_pc, kd_tree)

        jacobian = self.CalculateJacobain(correspondences, self.transform_)
        jacobian_square = np.matmul(jacobian.T, jacobian)
        residual_vec = self.CalculateResidualVector(correspondences)
        b = -1 * np.matmul(jacobian.T, residual_vec)

        # Initialize u and v parameters
        # diagonal_num = np.diagonal(jacobian_square)
        # u = 0.01 * np.max(diagonal_num) # τ = 0.01 here
        u = 0.03
        v = 2

        for i in range(max_iterations):
            print("Runing iteration %d..." % (i + 1))

            # calculate the possible transform delta
            pose_delta = self.SolveAxEqualtoB(
                b, jacobian_square + u * np.identity(6))

            # Detect if the pose_delta can be accepted or not
            temp_transform = self.transform_.Copy()
            temp_transform.UpdateTransform(pose_delta)
            temp_changed_source_pc = self.UpdateSourcePointCloud(
                temp_transform, self.source_pc_.copy())
            temp_correspondences = self.FindCorrespondenceWithKdtree(
                temp_changed_source_pc, kd_tree)
            temp_residual_vec = self.CalculateResidualVector(
                temp_correspondences)

            # calculate true loss delta
            F_temp_x = np.linalg.norm(temp_residual_vec) / (
                temp_residual_vec.shape[0] / 3)
            F_x = np.linalg.norm(residual_vec) / (residual_vec.shape[0] / 3)
            F_delta = F_x - F_temp_x

            # calculate theory loss delta
            g = -1 * b
            # L_delta is always positive, so we don't want it to be zero
            L_delta = np.linalg.norm(
                0.5 * pose_delta *
                (u * pose_delta.T - g)) / (residual_vec.shape[0] / 3) + 1e-5

            print('F_temp_x= %.2f' % F_temp_x, '  F_x= %.2f' % F_x,
                  '  F_delta= %.2f' % F_delta, '  L_delta= %.2f' % L_delta)

            # calculate the ratio ρ
            r = F_delta / L_delta

            print('r = %.2f' % r)

            if (r > 0):  # accept pose_delta
                self.transform_ = temp_transform.Copy()
                if r < 0.25:  # oh, the first order Taylor expansion step is too large, reduce step
                    u = u * 2. 
                elif r > 0.75: # first order Taylor expansion step is a little small, increase step
                    u = u / 3.   
                else:
                    pass # approximate to the GN
                # u = u*np.max(np.array([1./3, 1-(2*r-1)**3]))
                v = 2
                trans_delta = np.linalg.norm(
                    self.transform_.CalculateTranslationDelta(last_transform))
                print('trans_delta = ', trans_delta)

                if trans_delta < converge_trans or F_temp_x < 0.0001:
                    print('ICP run succeed!')
                    ShowCorrespondences(self.target_pc_, changed_source_pc,
                                        correspondences,
                                        "Correspondence result")
                    return True

                # After update the transform, find new correspondences for calculating pose_delta
                changed_source_pc = self.UpdateSourcePointCloud(
                    self.transform_, self.source_pc_.copy())
                correspondences = self.FindCorrespondenceWithKdtree(
                    changed_source_pc, kd_tree)
                residual_vec = self.CalculateResidualVector(correspondences)
                jacobian = self.CalculateJacobain(correspondences,
                                                  self.transform_)
                jacobian_square = np.matmul(jacobian.T, jacobian)
                b = -1 * np.matmul(jacobian.T, residual_vec)

                last_transform = self.transform_.Copy()

            else:  # current pose_delta is unacceptable
                u *= v
                v *= 2

            if (self.debug_mode_):
                ShowCorrespondences(self.target_pc_, changed_source_pc,
                                    correspondences, 'Show correspondences')
            print('u= %.2f, v= %.2f' % (u, v))

        # No converge
        print('ICP run failed!\n')
        ShowCorrespondences(self.target_pc_, changed_source_pc,
                            correspondences, "Correspondence result")
        return False
