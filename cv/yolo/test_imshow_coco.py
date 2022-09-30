from bounding_box import ImageBoundingBox
import os
from typing import List
from PIL import Image
# same structure as coco2yolo
example_dataset = {"000000558840.jpg": [{"category_id": 58, "bbox": [199.84, 200.46, 77.71, 70.88]}], "000000200365.jpg": [{"category_id": 58, "bbox": [234.22, 317.11, 149.39, 38.55]}, {"category_id": 58, "bbox": [239.48, 347.87, 160.0, 57.81]}, {"category_id": 58, "bbox": [296.65, 388.33, 1.03, 0.0]}, {"category_id": 58, "bbox": [251.87, 333.42, 125.94, 22.71]}], "000000495357.jpg": [
    {"category_id": 18, "bbox": [337.02, 244.46, 66.47, 66.75]}], "000000116061.jpg": [{"category_id": 18, "bbox": [213.81, 192.39, 53.94, 70.28]}], "000000016164.jpg": [{"category_id": 18, "bbox": [324.66, 247.92, 250.87, 181.02]}], "000000205350.jpg": [{"category_id": 18, "bbox": [260.18, 252.76, 67.91, 53.3]}], "000000000074.jpg": [{"category_id": 18, "bbox": [61.87, 276.25, 296.42, 103.18]}]}
categories = {"1": "person", "2": "bicycle", "3": "car", "4": "motorcycle", "5": "airplane", "6": "bus", "7": "train", "8": "truck", "9": "boat", "10": "traffic light", "11": "fire hydrant", "13": "stop sign", "14": "parking meter", "15": "bench", "16": "bird", "17": "cat", "18": "dog", "19": "horse", "20": "sheep", "21": "cow", "22": "elephant", "23": "bear", "24": "zebra", "25": "giraffe", "27": "backpack", "28": "umbrella", "31": "handbag", "32": "tie", "33": "suitcase", "34": "frisbee", "35": "skis", "36": "snowboard", "37": "sports ball", "38": "kite", "39": "baseball bat", "40": "baseball glove", "41": "skateboard", "42": "surfboard", "43": "tennis racket",
              "44": "bottle", "46": "wine glass", "47": "cup", "48": "fork", "49": "knife", "50": "spoon", "51": "bowl", "52": "banana", "53": "apple", "54": "sandwich", "55": "orange", "56": "broccoli", "57": "carrot", "58": "hot dog", "59": "pizza", "60": "donut", "61": "cake", "62": "chair", "63": "couch", "64": "potted plant", "65": "bed", "67": "dining table", "70": "toilet", "72": "tv", "73": "laptop", "74": "mouse", "75": "remote", "76": "keyboard", "77": "cell phone", "78": "microwave", "79": "oven", "80": "toaster", "81": "sink", "82": "refrigerator", "84": "book", "85": "clock", "86": "vase", "87": "scissors", "88": "teddy bear", "89": "hair drier", "90": "toothbrush"}


def coco2yolo(width, height, bounding_boxes: List[int]):
    x, y, delta_w, delta_h = bounding_boxes

    return (
        (x + delta_w / 2 ) / width,
        delta_w / 2 / width ,
        
        (y + delta_h / 2 ) / height,
        delta_h / 2/ height
    )

data = ImageBoundingBox()

for image, bounding_boxes in list(example_dataset.items())[0:]:
    current_bounding_box = bounding_boxes[0]
    bounding_boxes = current_bounding_box['bbox']
    image_path = os.path.join(
        "train2017",
        image
    )
    category_id = current_bounding_box['category_id']
 #   print(bounding_boxes)
    x, y = bounding_boxes[:2]
    x1, y1 = (x + bounding_boxes[3]), (y + bounding_boxes[-1])

    # x1, y2
    print((
        x, y, x1, y1
    ))

    image = Image.open(image_path)
    (width, height) = image.size

    data.load_image(
        image_path
    )

    data.load_bbox(
        coco2yolo(
            width=width,
            height=height,
            bounding_boxes=bounding_boxes
        )
    )
    data.show()
    break
