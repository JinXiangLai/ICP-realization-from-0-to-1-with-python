import numpy as np
from Geometry import Rigid
from Utils import GenerateTestedTargetAndSourcePointCloud
from ICP import GeneralIcp


def main():
    ground_truth_transform = Rigid(np.array([1.0, 2.0, 3.0]),
                                   np.array([0.6, 0.1, 0.2, 0.1]))
    target_pc, source_pc = GenerateTestedTargetAndSourcePointCloud(
        1000, ground_truth_transform, 1.0, True)

    # generate a transform guess
    init_translation = np.array([0.8, 1.2, 3.2])

    init_quaternion = np.array([0.7, 0.09, 0.21, 0.08])
    init_transform = Rigid(init_translation, init_quaternion)

    #icp = GeneralIcp(target_pc, source_pc, init_transform, 1.5, True)
    icp = GeneralIcp(target_pc, source_pc, init_transform, 1.5, False)
    #icp.RunIteration(max_iterations=30,converge_delta=0.001,use_lm = False)
    icp.RunIteration(max_iterations=50, converge_delta=0.001, use_lm=True)


if __name__ == "__main__":
    main()