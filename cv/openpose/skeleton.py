from confience_map import ConfidenceMap
import torch
from parity_fields import ParityFields
import torch.nn.functional as F
from coco import Coco


class Skeleton:
    def __init__(self, img_shape, keypoints, skeleton) -> None:
        self.img_shape = img_shape
        self.keypoints = keypoints
        self.skeleton = skeleton
        # default to 1 for now
        self.persons = 1
        self.sigma = 1

    def annotation_map(self):
        annotation_tensor = torch.zeros(
            self.img_shape)
        for index, keypoint in enumerate(self.keypoints):
            annotation_tensor[keypoint.x][keypoint.y] = 1
        return annotation_tensor

    def confidence_map(self):
        confidence_tensor = torch.zeros(
            (len(self.keypoints), ) + self.img_shape)
        confidence = ConfidenceMap()
        for index, KeyPoint in enumerate(self.keypoints):
            current_keypoint = KeyPoint.locaiton.reshape((1, 1, 2))

            if KeyPoint.is_visible():
                i, j = torch.meshgrid(
                    torch.arange(self.img_shape[0]),
                    torch.arange(self.img_shape[1]),
                    indexing='ij'
                )
                grid_tensor = torch.dstack([i, j]).float()
                results = confidence.function(
                    grid_tensor,
                    current_keypoint,
                    self.sigma
                )
                confidence_tensor[index] = results
        return confidence_tensor

    def paf_field(self):
        parity_fields = ParityFields()
        parity_tensor = torch.zeros(
            (len(self.skeleton), ) + self.img_shape + (2, ))

        for index, (keypoint_index_i, keypoint_index_j) in enumerate(self.skeleton):
            p_1 = self.keypoints[keypoint_index_i - 1]
            p_2 = self.keypoints[keypoint_index_j - 1]

            if (p_1.visible and p_2.visible):
                x, y = torch.meshgrid(
                    torch.arange(0, self.img_shape[0]),
                    torch.arange(0, self.img_shape[1]),
                    indexing='ij'
                )
                res = parity_fields.optimized_function(x, y, p_1.locaiton, p_2.locaiton, self.sigma, shape=(
                    self.img_shape[0],
                    self.img_shape[1]
                ))
                parity_tensor[index] = res
        return parity_tensor

    def E_generated(self, p_1, p_2, d_1, d_2):
        return self._trapezoidal(f_getter=lambda dx: self._f(dx, p_1, p_2, d_1, d_2))

    def E(self, L_c, d_1, d_2):
        return self._trapezoidal(f_getter=lambda dx: self._g(dx, L_c, d_1, d_2))

    def _f(self, u, p_1, p_2, d_1, d_2):
        p = (
            1 - u
        ) * d_1 + u * d_2
        self.parity_fields = ParityFields()
        res = self.parity_fields.optimized_function(p[0], p[1], p_1, p_2, 5)
        if not torch.is_tensor(res):
            res = torch.tensor([0, 0]).float()
        res = res @ (d_2 - d_1) / torch.norm(d_2 - d_1)
        return res

    def _g(self, u, L_c, d_1, d_2):
        p = (
            1 - u
        ) * d_1 + u * d_2
        res = L_c[p[0].long(), p[1].long(), :]
        res = res @ (d_2 - d_1) / torch.norm(d_2 - d_1)
        return res

    def _trapezoidal(self, f_getter, n=3):
        dx = 1 / n
        # (1 - u) * dj + udj
        results = 0
        for i in range(0, n):
            x_k = dx * i
            if i > 0 and (i + 1) < n:
                results += 2 * f_getter(x_k)
            else:
                results += f_getter(x_k)
        return dx/2 * results

    def _mergePoints(self, x, y):
        i, j = torch.meshgrid(
            x,
            y,
            indexing='ij'
        )
        grid_tensor = torch.dstack([i, j]).float()
        return grid_tensor

    def merge(self, confidence, paf):
        # it's not super clear for the paper how the non maximum suppression is supposed to work
        # you know the keypoints with a given confidence.
        # https://docs.nvidia.com/isaac/packages/skeleton_pose_estimation/doc/2Dskeleton_pose_estimation.html
        # ^ after some googling
        # -> So the maxpooling makes sense, but still not sure how they do maximum supression
        #   -> Do they just create a bounding box for all keypoints ?
        # -> Okay, I think I get it now
        #   -> you run a maxpool
        #   -> Extract where value is greater than > PARAMETER
        # ->  Then run that through the PAF
        #   :)
        # -> Still not entirely sure how to separate from each person
        #   -> I guess this is where the non maximum suppression comes in
        #confidence = torch.sigmoid(confidence)
        #        limit = 0.70
        #        maxpooled = confidence

        #        for keypoints in range(len(self.keypoints)):
        for _, (keypoint_index_i, keypoint_index_j) in enumerate(self.skeleton):

            def get_keypoints(index):
                x = F.pad(confidence[index], (2, 2, 2, 2))
                center = x[1:x.shape[0] - 1, 1:x.shape[1]-1]
                left = x[2:x.shape[0], 1:x.shape[1] - 1]
                right = x[:x.shape[0] - 2, 1:x.shape[1] - 1]
                top = x[1:x.shape[0] - 1, :x.shape[1] - 2]
                bottom = x[1:x.shape[0] - 1, 2:x.shape[1]]

                peak = (
                    (center > left).long() &
                    (center > right).long() &
                    (center > top).long() &
                    (center > bottom).long()
                )
    #            print(peak)
    #            exit(0)
                keypoints_x, keypoints_y = torch.where(peak > 0)
                return torch.dstack([keypoints_x, keypoints_y]).float()[0]
            """
            TODO: Sloppy merge, but this should be fixed !!!!

            Look at this a bit more, looks like the way they solve it by padding the array
            - Then reading 2+ x
            -              -2x
            -              2+y
            -              -2 y
            -   With a center of (1 + x, y + 1)
            -  ^ value with max in center = good item.
            """
            """
            min_max_item = [None, None, None]
            for i in keypoints_x:
                for j in keypoints_y:
                    results = (self.E(
                        paf[index, :, :, :],
                        d_1=i,
                        d_2=j
                    ))
                    if results.item() != 0 and not torch.isnan(results):
                        if (min_max_item[0] is None or min_max_item[0] < results.item()):

                           # if truth_point_i.locaiton[0] == i[0] and truth_point_i.locaiton[1] == i[1]:
                           #    if truth_point_j.locaiton[0] == j[0] and truth_point_j.locaiton[1] == j[1]:
                           #       print("Found match :)")

                            min_max_item[0] = results.item()
                            min_max_item[1] = i  # truth_point_i.locaiton #i
                            min_max_item[2] = j  # truth_point_j.locaiton #j
            """
            try:
                min_max_item = [1, None, None]
                min_max_item[1] = get_keypoints(keypoint_index_i - 1)[0]
                min_max_item[2] = get_keypoints(keypoint_index_j - 1)[0]
                # print(min_max_item)
                # print(min_max_item)
                yield (min_max_item)
                #print(f"limb {index}")
            except Exception as e:
                print(e)

    def skeleton_from_keypoints(self):
        for (i, j) in self.skeleton:
            if self.keypoints[i - 1].is_visible() and self.keypoints[j - 1].is_visible():
                yield (1, self.keypoints[i - 1].locaiton, self.keypoints[j - 1].locaiton)


if __name__ == "__main__":
    obj = Skeleton(
        img_shape=(640, 480),
        skeleton=[[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [
            6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]],
        keypoints=[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (325, 160, 2), (398, 177, 2), (0, 0, 0), (437, 238, 2),
                   (0, 0, 0), (477, 270, 2), (287, 255, 1), (339, 267, 2), (0, 0, 0), (423, 314, 2), (0, 0, 0), (355, 367, 2)]
    )
    # print(obj.confidence_map())
    # print(obj.paf_field())
    # exit(0)
    items = list(obj.merge(
        obj.confidence_map(),
        obj.paf_field()
    ))
#    items = list(
#        obj.skeleton_from_keypoints()
 #   )
    obj = Coco()
    obj.load_annotations()
    obj.results[0].plot_image_skeleton_keypoints(items)
