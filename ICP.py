from typing import *
import numpy as np
from matplotlib import pyplot as plt

#=========================================================================================#
#=================================== class Rigid =========================================#
#=========================================================================================#
class Rigid():

    def __init__(self,translation:np.ndarray=np.zeros((3,1),dtype=float), 
                 quaternion:np.ndarray=np.array([[1],[0],[0],[0]],dtype=float)) -> None:
        self.trans_ = translation
        self.quat_ = self.NormalizeQuaternion(quaternion)

    def __str__(self) -> str:
        return "translation: %s\nquaternion: %s"%(str(self.trans_.tolist()),str(self.quat_.tolist()))

    def Copy(self) :
        new_trans = self.trans_.copy()
        new_quat = self.quat_.copy()
        new_transform = Rigid(new_trans,new_quat)
        return new_transform
                                 
    def NormalizeQuaternion(self,quaternion:np.ndarray) -> np.ndarray :
        vector_norm = np.linalg.norm(quaternion)
        quaternion /= vector_norm
        return quaternion
    
    def GetRotationMatrix(self) -> np.ndarray :
        qr, qx, qy, qz = self.quat_[:,0]
        R = np.array([
                [qr**2+qx**2-qy**2-qz**2, 2*(qx*qy-qr*qz), 2*(qz*qx+qr*qy)],
                [2*(qx*qy+qr*qz), qr**2-qx**2+qy**2-qz**2, 2*(qy*qz-qr*qx)],
                [2*(qz*qx-qr*qy), 2*(qy*qz+qr*qx), qr**2-qx**2-qy**2+qz**2]
            ],dtype=float)
        return R

    def UpdateTransform(self, pose_delta:np.ndarray) :
        self.trans_ += pose_delta[0:3,0].reshape(3,1)
        self.quat_[1:,0] += pose_delta[3:,0]
        self.NormalizeQuaternion(self.quat_)

    def CalculateTranslationDelta(self, transform) -> float:
        return np.linalg.norm(self.trans_-transform.trans_)

    def CalculateRotationDelta(self, transform) -> float:
        pass

#=========================================================================================#
#============================== class Correspondence =====================================#
#=========================================================================================#
class Correspondence():
    def __init__(self,target:np.ndarray,source:np.ndarray) -> None:
        self.target_ = target
        self.source_ = source

    def CalculateResidual(self) -> np.ndarray:
        self.residual_ = self.source_ - self.target_
        return self.residual_ 

    def CalculateResidualWithWeight(self):
        pass

    def CalculateJacobian(self,transform:Rigid) -> np.ndarray:
        '''
        The jacobian matrix looks like:
            x  y  z  qx  qy  qz
        dX   1
        dY      1
        dZ         1
        '''
        qr, qx, qy, qz = transform.quat_[:,0]
        ax, ay, az = self.source_[:,0]

        jacobian = np.zeros((3,6),dtype=float)
        jacobian[:,0:3] = np.identity(3,dtype=float)
        jacobian[:,3:] =  np.array([
            [qy*ay+qz*az, -2*qy*ax+qx*ay+qr*az, -2*qz*ax-qr*ay+qx*az],
            [qy*ax-2*qx*ay-qr*az, qx*ax+qz*az, qr*ax-2*qz*ay+qy*az],
            [qz*ax+qr*ay-2*qx*az, -qr*ax+qz*ay-2*qy*az, qx*ax+qy*ay]              
            ])

        return jacobian

#=========================================================================================#
#==================================== class ICP ==========================================#
#=========================================================================================#
class ICP():
    def __init__(self,target_pc:np.ndarray, source_pc:np.ndarray, init_transform:Rigid,
                 max_dis:float=1.5, debug_mode:bool=False) -> None:
        self.target_pc_ = target_pc
        self.source_pc_ = source_pc
        self.transform_ = init_transform
        self.max_dis = max_dis
        self.debug_mode_ = debug_mode

    def ShowTargetAndSourcePointCloud(self, target_pc:np.ndarray, source_pc:np.ndarray, win_name:str):
        tar_x, tar_y, tar_z = target_pc[0,:], target_pc[1,:], target_pc[2,:]
        src_x, src_y, src_z = source_pc[0,:], source_pc[1,:], source_pc[2,:]
        
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'auto'})
        ax.scatter(tar_x, tar_y, tar_z, s=1, c='r', zorder=10)
        ax.scatter(src_x, src_y, src_z, s=1, c='b', zorder=10)
        ax.set_title(win_name)
        plt.show()



    def ShowCorrespondences(self, target_pc:np.ndarray, source_pc:np.ndarray, correspondences:List[Correspondence], 
                            win_name:str):
        tar_x, tar_y, tar_z = target_pc[0,:], target_pc[1,:], target_pc[2,:]
        src_x, src_y, src_z = source_pc[0,:], source_pc[1,:], source_pc[2,:]
        
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'auto'})
        ax.scatter(tar_x, tar_y, tar_z, s=1, c='r', zorder=10)
        ax.scatter(src_x, src_y, src_z, s=1, c='b', zorder=10)
        ax.set_title(win_name)

        for cor in correspondences:
            ax.plot([cor.target_[0,0],cor.source_[0,0]],[cor.target_[1,0],cor.source_[1,0]],
                    [cor.target_[2,0],cor.source_[2,0]],'-g')
        plt.show()


    def FindCorrespondence(self, target_pc:np.ndarray, source_pc:np.ndarray):
        '''
        TODO: violent search is easy by inefficient
        '''
        correspondences = []
        _, col = target_pc.shape
        _, col_src = source_pc.shape

        i = 0
        j = 0
        for i in range(col):
            pt = target_pc[:,i].reshape(3,1)
            min_dis = 1e10
            cor_index = -1

            for j in range(col_src):
                ps = source_pc[:,j].reshape(3,1)
                dis = np.linalg.norm(pt-ps)
                if dis<self.max_dis and dis<min_dis:
                    cor_index = j
                    min_dis = dis
            if cor_index>=0: 
                correspondences.append(Correspondence(pt, source_pc[:,cor_index].reshape(3,1)))
        
        if(self.debug_mode_):
            self.ShowCorrespondences(target_pc, source_pc, correspondences, 'Show correspondences')
        return correspondences


    def CalculateResidualVector(self, correspondences:List[Correspondence]):
        if len(correspondences) < 1:
            print("None correspondence!\n")
            return

        residual_vec = np.zeros((len(correspondences)*3,1),dtype=float)
        i = 0
        index = 0
        while i < residual_vec.shape[0]:
            residual_vec[i:i+3,0] = correspondences[index].CalculateResidual().reshape(3)
            i+=3
            index+=1
            
        return residual_vec



    def CalculateJacobain(self, correspondences:List[Correspondence], transform:Rigid):
        jacobian = np.zeros((len(correspondences)*3,6),dtype=float)
        i = 0
        index = 0
        while i<jacobian.shape[0]:
            partial_derivative = correspondences[index].CalculateJacobian(transform)
            jacobian[i:i+3,0:6] = partial_derivative
            i+=3
            index+=1

        return jacobian


    def SolveAxEqualtoB(self, transform:Rigid, residual_vec:np.ndarray, jacobian:np.ndarray):
        jacobian_square = np.matmul(jacobian.T,jacobian)
        b = -1 * np.matmul(jacobian.T, residual_vec)
        jacobian_square_inv = np.linalg.inv(jacobian_square)
        delta_x = np.matmul(jacobian_square_inv,b)
        
        if self.debug_mode_:
            print("delta_x: ",delta_x)
        transform.UpdateTransform(delta_x)

        return transform

    def UpdateSourcePointCloud(self, transform:Rigid, source_pc:np.ndarray):
        _, col = source_pc.shape
        for i in range(col):
            source_pc[:,i] = (np.matmul(transform.GetRotationMatrix(),source_pc[:,i].reshape(3,1)) + transform.trans_).reshape(3) 
        return source_pc

    def RunIteration(self, iterations=30, converge_delta=0.01) -> bool:
        last_transform = self.transform_.Copy()
        for i in range(iterations):
            print("Runing iteration %d..."%(i+1))
            changed_source_pc = self.UpdateSourcePointCloud(self.transform_, self.source_pc_.copy())
            if(self.debug_mode_):
                self.ShowTargetAndSourcePointCloud(self.target_pc_, changed_source_pc, "Iteration %d"%(i))

            correspondences = self.FindCorrespondence(self.target_pc_, changed_source_pc)

            if(len(correspondences)<10):
                print("Correspondences size too small: ", len(correspondences), "\n ICP run failed!") 
                return False 

            residual_vec = self.CalculateResidualVector(correspondences)

            jacobian = self.CalculateJacobain(correspondences, self.transform_)

            self.SolveAxEqualtoB(self.transform_, residual_vec, jacobian)

            translation_delta = last_transform.CalculateTranslationDelta(self.transform_)

            if(translation_delta<converge_delta):
                print("ICP run succeed!\n")
                return True

            last_transform = self.transform_.Copy()
            
        print("ICP run failed!\n")
        return False


#==================================== Utils Fun ==========================================#
def GenerateTestedTargetAndSourcePointCloud(point_num:int, transform:Rigid, radius: float=1.0, 
                                            add_noise:bool=False):
    '''Here generate two point clouds shaped in ring'''
    inc_step = radius * 2.0 *2.0 / point_num 
    source_pc = np.zeros((3,point_num),dtype=float)
    x = -1.0
    times = 0
    while(times<point_num):
        y = np.sqrt(radius - x**2)
        source_pc[:,times] = [x,y,0]
        times += 1
        source_pc[:,times] = [x, -y, 0]
        x += inc_step
        times += 1 
    
    target_pc = np.zeros((3,point_num),dtype=float)
    if(add_noise):
        noise = np.random.normal(0, 0.01, (3,point_num))
        target_pc = np.matmul(transform.GetRotationMatrix(), source_pc) +transform.trans_ + noise
    else:
        target_pc = np.matmul(transform.GetRotationMatrix(), source_pc) +transform.trans_
    
    return target_pc, source_pc


#==================================== Main Fun ===========================================#
def main():
    ground_truth_transform = Rigid(np.array([[1.0],[2.0],[3.0]]),np.array([[0.6],[0.1],[0.2],[0.1]]))
    target_pc, source_pc = GenerateTestedTargetAndSourcePointCloud(1000, ground_truth_transform, 1.0)
    
    # generate a transform guess
    init_translation = np.array([[0.8],[1.8],[3.2]])
    init_quaternion = np.array([[0.75],[0.09],[0.21],[0.08]]) 
    init_transform = Rigid(init_translation, init_quaternion) # 给定变换初值
    
    icp = ICP(target_pc, source_pc, init_transform, 1.5, False)
    icp.RunIteration()


if __name__=="__main__":
    main()