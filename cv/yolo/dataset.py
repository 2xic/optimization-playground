import torchvision

coco_train = torchvision.datasets.CocoDetection(root="./",
                                annFile="./labels.json")
