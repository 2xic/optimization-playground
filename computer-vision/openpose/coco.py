import json
from helpers import get_local_dir
from image_label import ImageLabel
from keypoint import KeyPoint

class Coco:
    def __init__(self) -> None:
        self.results = []
        self.skeleton = None

    def load_annotations(self):
        self.annotations = open(
            get_local_dir("annotations/person_keypoints_train2017.json"), "r").read()
        self.annotations = json.loads(self.annotations)
        self.images = {}
        for entry in self.annotations['annotations'][:10]:
            keypoints = entry['keypoints']
            bbox = entry['bbox']
            results = []
            for i in range(0, len(keypoints), 3):
                results.append(KeyPoint(*keypoints[i:i+3]))
            self.results.append(
                ImageLabel(
                    image_id=str(entry['image_id']),
                    keypoints=results,
                    bbox=bbox,
                )
            )
        self.skeleton = self.annotations['categories'][0]['skeleton']

        return self

    def show(self, index=0):
        self.results[index].show(self.skeleton)
        
    def get_metadata(self, index):
        return {
            "skeleton": self.skeleton,
            "keypoints": self.results[index].keypoints,
            "bbox": self.results[index].bbox,
            "path":  get_local_dir("train2017/" + self.results[index].name),
            "shape": (480, 640)
        }

if __name__  == "__main__":
    obj = Coco()
    obj.load_annotations()
    # obj.show(4)
#    obj.show(6)
