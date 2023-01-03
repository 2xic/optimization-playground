from confience_map import ConfidenceMap
import torch
from parity_fields import ParityFields
import torch.nn.functional as F
from coco import Coco

class Skeleton:
    def __init__(self, img_shape, keypoints, skeleton, bbox) -> None:
        self.img_shape = img_shape
        self.keypoints = keypoints
        self.skeleton = skeleton

        # default to 1 for now
        self.persons = 1
        self.sigma = 5
        self.bbox = bbox
        self.debug = True

    
    def print(self, *args):
        if self.debug:
            print(args)

    def annotation_map(self, channels):
        heatmap = torch.zeros((channels, ) + self.img_shape)
        heatmap[
            :,
            int(self.bbox[1]):int(self.bbox[1])+int(self.bbox[3]),
            int(self.bbox[0]):int(self.bbox[0])+int(self.bbox[2]),
        ] = 1
        return heatmap

    def confidence_map(self):
        confidence_tensor = torch.zeros(
            (len(self.keypoints), ) + self.img_shape)
        confidence = ConfidenceMap()
        for index, keypoint in enumerate(self.keypoints):
            current_keypoint = keypoint.location.reshape((1, 1, 2))

            if keypoint.is_visible():
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
                res = parity_fields.optimized_function(x, y, p_1.location, p_2.location, self.sigma, shape=(
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

    def merge(self, confidence, paf, keypoint=None):
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
        skeleton = self.skeleton

        if keypoint is not None:
            skeleton = [skeleton[keypoint]]

        for index, (keypoint_index_i, keypoint_index_j) in enumerate(skeleton):
            """
            TODO: Sloppy merge, but this should be fixed !!!!
            """
         #   print(42)
            try:
                min_max_item = [1, None, None]
                candidates_a = self.extract_keypoints_from_confidence(confidence, keypoint_index_i - 1)
                candidates_b = self.extract_keypoints_from_confidence(confidence, keypoint_index_j - 1)

                score = []
                for a in candidates_a:
                    for b in candidates_b:
#                        print((a, b))
                        score.append((
                            a[0], 
                            b[0],
                            self.E(
                                paf[index],
                                a[0],
                                b[0]
                            )
                        ))
                self.print(len(score))
                score = list(sorted(score, key=lambda x: x[2]))
                if 0 < len(score):
                    min_max_item[1] = score[0][1]                
                    min_max_item[2] = score[0][0]              
                yield (min_max_item)
            except Exception as e:
                self.print(e)

    def extract_keypoints_from_confidence(self, confidence, index):
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
        keypoints_x, keypoints_y = torch.where(peak > 0)

        return torch.dstack([keypoints_x, keypoints_y]).float()

    def skeleton_from_keypoints(self):
        for (i, j) in self.skeleton:
            if self.keypoints[i - 1].is_visible() and self.keypoints[j - 1].is_visible():
                yield (1, self.keypoints[i - 1].location, self.keypoints[j - 1].location)


if __name__ == "__main__":
    coco = Coco()
    coco.load_annotations()

    metadata = coco.get_metadata(6)

    skeleton = Skeleton(
        img_shape=metadata['shape'],
        skeleton=metadata['skeleton'],
        keypoints=metadata['keypoints']
    )
    items = list(skeleton.merge(
        skeleton.confidence_map(),
        skeleton.paf_field()
    ))
    coco.results[6].plot_image_skeleton_keypoints(items)
