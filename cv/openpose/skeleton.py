from coco import KeyPoint
from confience_map import ConfidenceMap
import torch
from parity_fields import ParityFields
import os
import torch.nn as nn
import torch.nn.functional as F
from coco import Coco

class Skeleton:
    def __init__(self, img_shape, keypoints, skeleton) -> None:
        self.img_shape = img_shape
        self.keypoints = list(map(lambda x: KeyPoint(*x), keypoints))
        self.skeleton = skeleton
        # default to 1 for now
        self.persons = 1
        self.sigma = 1

    def confidence_map(self):
        confidence_tensor = torch.zeros(
            (len(self.keypoints), ) + self.img_shape)
        confidence = ConfidenceMap()
        for index, KeyPoint in enumerate(self.keypoints):
            current_keypoint = KeyPoint.locaiton.reshape((1, 1, 2))

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
        if os.path.isfile('parity_fields.pt'):
            return torch.load('parity_fields.pt')['tensor']
        parity_fields = ParityFields()
        parity_tensor = torch.zeros(
            (len(self.skeleton), ) + self.img_shape + (2, ))
        for index, (keypoint_index_i, keypoint_index_j) in enumerate(self.skeleton):
            p_1 = self.keypoints[keypoint_index_i - 1]
            p_2 = self.keypoints[keypoint_index_j - 1]

            if (p_1.visible and p_2.visible):
                for i in range(0, self.img_shape[0]):
                    for j in range(0, self.img_shape[1]):
                        res = parity_fields.unoptimized_function(
                            torch.tensor([i, j]), p_1.locaiton, p_2.locaiton, self.sigma)
                        parity_tensor[index, i, j] = res
        torch.save({
            'tensor': parity_tensor
        }, 'parity_fields.pt')
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
        res = self.parity_fields.unoptimized_function(p, p_1, p_2, 5)
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
        x = F.pad(confidence, (1, 1, 1, 1))
        pool = nn.MaxPool2d(3, stride=1)
        limit = 0.70
        maxpooled = pool(
            x
        ) if False else confidence

#        for keypoints in range(len(self.keypoints)):
        for index, (keypoint_index_i, keypoint_index_j) in enumerate(self.skeleton):
            (x, y) = (torch.where(maxpooled[keypoint_index_i - 1, :, :] > limit))
            (x_1, y_1) = (torch.where(maxpooled[keypoint_index_j - 1, :, :] > limit))

            x_y_1 = self._mergePoints(x, y).reshape((-1, 2))
            x_y_2 = self._mergePoints(x_1, y_1).reshape((-1, 2))

            print(x.shape)
            print(x_1.shape)

            """
            TODO: Sloppy merge, but this should be fixed !!!!
            """
            #print(x_y_1.shape)
            #print(x_y_2.shape)

            truth_point_i = self.keypoints[keypoint_index_i - 1]
            truth_point_j = self.keypoints[keypoint_index_j - 1]

            min_max_item = [None, None, None]
            for i in x_y_1:
                for j in x_y_2:
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
                            min_max_item[1] = i # truth_point_i.locaiton #i
                            min_max_item[2] = j # truth_point_j.locaiton #j
            print(min_max_item)
            yield (min_max_item)
            #print(f"limb {index}")

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
    #print(obj.confidence_map())
    #print(obj.paf_field())
    #exit(0)
    items = list(obj.merge(
        obj.confidence_map(),
        obj.paf_field()
    ))
#    items = list(
#        obj.skeleton_from_keypoints()
#    )
    obj = Coco()
    obj.load_annotations()
    obj.results[0].plot_image_skeleton_keypoints(items)

