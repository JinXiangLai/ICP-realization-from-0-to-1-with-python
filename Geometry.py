import numpy as np


class Point():

    def __init__(self,
                 value: np.ndarray = np.array([1e9, 1e9, 1e9], dtype=float)):
        self.value_ = value  # 1D array
        self.left_point_ = None
        self.right_point_ = None

    def __str__(self) -> str:
        return self.value_

    def SetValue(self, value: np.ndarray):
        self.value_ = value

    @property
    def value(self):
        return self.value_


class Distance():  # for reference float type object

    def __init__(self, value: float = 1e9) -> None:
        self.value_ = value

    def SetValue(self, value: float):
        self.value_ = value

    @property
    def value(self):
        return self.value_


class Rigid():

    def __init__(
        self,
        translation: np.ndarray = np.zeros((3), dtype=float),
        quaternion: np.ndarray = np.array([1, 0, 0, 0], dtype=float)) -> None:
        self.trans_ = translation
        self.quat_ = self.NormalizeQuaternion(quaternion)

    def __str__(self) -> str:
        return "translation: %s\nquaternion: %s" % (str(
            self.trans_.tolist()), str(self.quat_.tolist()))

    def Copy(self):
        new_trans = self.trans_.copy()
        new_quat = self.quat_.copy()
        new_transform = Rigid(new_trans, new_quat)
        return new_transform

    def NormalizeQuaternion(self, quaternion: np.ndarray) -> np.ndarray:
        vector_norm = np.linalg.norm(quaternion)
        quaternion /= vector_norm
        return quaternion

    def GetRotationMatrix(self) -> np.ndarray:
        qr, qx, qy, qz = self.quat_
        R = np.array([
                [
                    qr**2+qx**2-qy**2-qz**2, 2*(qx*qy-qr*qz), 2*(qz*qx+qr*qy)
                ],
                [
                    2*(qx*qy+qr*qz), qr**2-qx**2+qy**2-qz**2, 2*(qy*qz-qr*qx)
                ],
                [
                    2*(qz*qx-qr*qy), 2*(qy*qz+qr*qx), qr**2-qx**2-qy**2+qz**2
                ]
            ],dtype=float)
        return R

    def UpdateTransform(self, pose_delta: np.ndarray):
        self.trans_ += pose_delta[:3]
        self.quat_[1:] += pose_delta[3:]
        self.NormalizeQuaternion(self.quat_)

    def CalculateTranslationDelta(self, transform) -> float:
        return np.linalg.norm(self.trans_ - transform.trans_)

    def CalculateRotationDelta(self, transform) -> float:
        pass


class Correspondence():

    def __init__(self, target: np.ndarray, source: np.ndarray) -> None:
        self.target_ = target
        self.source_ = source

    def CalculateResidual(self) -> np.ndarray:
        self.residual_ = self.source_ - self.target_
        return self.residual_

    def CalculateResidualWithWeight(self):
        pass

    def CalculateJacobian(self, transform: Rigid) -> np.ndarray:
        '''
        The jacobian matrix looks like:
            x  y  z  qx  qy  qz
        dX   1
        dY      1
        dZ         1
        '''
        qr, qx, qy, qz = transform.quat_
        ax, ay, az = self.source_

        jacobian = np.zeros((3, 6), dtype=float)
        jacobian[:, 0:3] = np.identity(3, dtype=float)
        jacobian[:,3:] =  np.array([
            [
                qy*ay+qz*az, -2*qy*ax+qx*ay+qr*az, -2*qz*ax-qr*ay+qx*az
            ],
            [
                qy*ax-2*qx*ay-qr*az, qx*ax+qz*az, qr*ax-2*qz*ay+qy*az
            ],
            [
                qz*ax+qr*ay-2*qx*az, -qr*ax+qz*ay-2*qy*az, qx*ax+qy*ay
            ]              
            ])

        return jacobian
