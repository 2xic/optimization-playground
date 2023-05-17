from dis import dis
from telnetlib import BM
import numpy as np
from bounding_box import ImageBoundingBox
from constants import Constants


def iou(b, b1):
    x_0 = max(b[0], b1[0])
    y_0 = max(b[1], b1[1])

    x_1 = min(b[2], b1[2])
    y_1 = min(b[3], b1[3])

    intersection = max(
        0,
        (x_1 - x_0) + 1
    ) * max(
        0,
        (
            y_1 - y_0
        ) + 1
    )

    size_box_0 = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    size_box_1 = (b1[2] - b1[0] + 1) * (b1[3] - b1[1] + 1)

    return (intersection / (size_box_0 + size_box_1 - intersection))

def nms(bounding_boxes, scores):
    b_nms = []
    b_scores = []
    for index_i, i in enumerate(bounding_boxes):
        discard = False
        for index_j, j in enumerate(bounding_boxes):
            has_close_iou = iou(i, j) > 0.8
            if has_close_iou and scores[index_i] < scores[index_j]:
                discard = True
        if not discard:
            b_nms.append(i)
            b_scores.append(scores[index_i])
    return zip(b_nms, b_scores)

def soft_nms(bounding_boxses, scores):
    def f(s_i, M, b_i):
        sigma = 0.01
        return s_i * np.exp(
            -iou(M, b_i)**2
            / sigma
        )
    copy_scores = [0, ] * len(scores)
    d = []
    bounding_boxses = list(zip(bounding_boxses, list(range(len(bounding_boxses)))))
    while len(bounding_boxses):
        m = scores.index(max(scores))
        M, _ = bounding_boxses[m]
        d.append(M)
        del bounding_boxses[m]
        del scores[m]
        for index, b_i_index in enumerate(bounding_boxses):
            b_i, bounding_box_index = b_i_index
            scores[index] = f(
                scores[index], 
                M,
                b_i
            )
            copy_scores[bounding_box_index] = scores[index]
    return d, copy_scores


if __name__ == "__main__":
    original_box = (0.4772736728191376, 0.07791886478662491,
                    0.4721715450286865, 0.07105270773172379)

    assert iou(original_box, original_box) == 1,  iou(original_box, original_box) 

    boxses_with_some_noise = [
        [original_box[i] * (1 + np.random.rand() * 0.1)
        for i in range(len(original_box))]
        for _ in range(4)
    ]
    scores = list(range(len(boxses_with_some_noise)))


    plot = ImageBoundingBox().load_image(
        "000000558840.jpg",
        Constants()
    )
    for i in nms(*soft_nms(boxses_with_some_noise, scores)):
        plot.load_bbox(
            i
        )
    plot.show()
