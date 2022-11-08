from coco import Coco

dataset = Coco().load_annotations().show()
print(dataset)

