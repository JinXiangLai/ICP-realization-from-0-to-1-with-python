import numpy as np
from Geometry import Point, Distance


class KdTree():

    def __init__(self, point_cloud: np.ndarray, build_now: bool = False):
        self.point_cloud_ = point_cloud
        self.root_ = None
        if build_now:
            self.root_ = self.BuildKdTree(point_cloud)

    def __BuildKdTreeProcess(self, point_cloud: np.array, depth: int,
                             left_boundary: int, right_boundary: int) -> Point:
        if self.root_ != None:
            print("Kd three hs existed!\n")
            return None
        if left_boundary > right_boundary:
            return None
        if left_boundary >= right_boundary:
            root = Point(point_cloud[:, left_boundary])
            return root

        compare_dimension = depth % 3
        # Find median to create current root node
        sort_part = point_cloud[:, left_boundary:right_boundary + 1]
        # must slice here to sort in plcae
        sort_part[:, :] = sort_part[:, sort_part[compare_dimension].argsort()]

        anchor_index = (right_boundary -
                        left_boundary) // 2 + left_boundary  # round
        root = Point(point_cloud[:, anchor_index])

        # recursion
        depth += 1
        root.left_point_ = self.__BuildKdTreeProcess(point_cloud, depth,
                                                     left_boundary,
                                                     anchor_index - 1)
        root.right_point_ = self.__BuildKdTreeProcess(point_cloud, depth,
                                                      anchor_index + 1,
                                                      right_boundary)

        return root

    def BuildKdTree(self, point_cloud: np.array) -> Point:
        if point_cloud.shape[1] == 0:
            print("Input point cloud is empty!\n")
            return None

        return self.__BuildKdTreeProcess(point_cloud, 0, 0,
                                         point_cloud.shape[1] - 1)

    def __SearchApproxClosestPointProcess(self,
                                          input: np.array,
                                          root: Point,
                                          dis: Distance,
                                          res: Point,
                                          depth: int = 0) -> None:
        if (root == None or dis == 0):
            return res

        cur_dis = np.linalg.norm(input - root.value)
        if cur_dis < dis.value:
            dis.SetValue(cur_dis)
            res.SetValue(root.value)

        compare_dimension = depth % 3

        if (input[compare_dimension] < root.value[compare_dimension]):
            return self.__SearchApproxClosestPointProcess(
                input, root.left_point_, dis, res, depth + 1)
        else:
            return self.__SearchApproxClosestPointProcess(
                input, root.right_point_, dis, res, depth + 1)

    def SearchApproxClosestPoint(self, input: np.array) -> Point:
        if (self.root_ == None):
            print("Error, Null self.root_")
            return None
        dis = Distance()
        res = Point()
        self.__SearchApproxClosestPointProcess(input, self.root_, dis, res, 0)

        return res

    def __SearchClosestPointProcess(self,
                                    input: np.array,
                                    root: Point,
                                    dis: Distance,
                                    res: Point,
                                    depth: int = 0) -> None:
        if (root == None or dis.value == 0):
            return

        cur_dis = np.linalg.norm(input - root.value)
        if cur_dis < dis.value:
            dis.SetValue(cur_dis)
            res.SetValue(root.value)

        compare_dimension = depth % 3
        if (input[compare_dimension] < root.value_[compare_dimension]):
            self.__SearchClosestPointProcess(input, root.left_point_, dis, res,
                                             depth + 1)  # recursion process

            if root.right_point_ == None:  # backtracking process
                return
            else:
                right_child_dis = np.linalg.norm(input -
                                                 root.right_point_.value)
                if right_child_dis < dis.value:
                    self.__SearchClosestPointProcess(input, root.right_point_,
                                                     dis, res, depth + 1)

        else:
            self.__SearchClosestPointProcess(input, root.right_point_, dis,
                                             res, depth + 1)
            if root.left_point_ == None:
                return
            else:
                left_child_dis = np.linalg.norm(input - root.left_point_.value)
                if left_child_dis < dis.value:
                    self.__SearchClosestPointProcess(input, root.left_point_,
                                                     dis, res, depth + 1)

    def SearchClosestPoint(self, input: np.array) -> Point:
        if (self.root_ == None):
            print("Error, Null self.root_")
            return None
        dis = Distance()
        res = Point()
        self.__SearchClosestPointProcess(input, self.root_, dis, res, 0)
        return res


def main():
    # reference website： https://en.wikipedia.org/wiki/K-d_tree
    data = np.array(
        [[7, 5, 9, 4, 8, 2], [2, 4, 6, 7, 1, 3], [0, 0, 0, 0, 0, 0]],
        dtype=float)
    kd_tree = KdTree(data, True)
    print(kd_tree.SearchApproxClosestPoint(np.array([8, 1, 1])).value_)
    print(kd_tree.SearchClosestPoint(np.array([8, 1, 1])).value_)


if __name__ == '__main__':
    main()
