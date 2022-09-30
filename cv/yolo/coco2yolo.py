from collections import defaultdict
import json

class Coco2Yolo:
    def __init__(self) -> None:
        self.annotations = open("annotations/instances_train2017.json", "r").read()
        self.annotations = json.loads(self.annotations)
        self.id_category = {

        }

        for i in self.annotations['categories']:
            self.id_category[i['id']] = i['name']

        self.image_bbox = defaultdict(list)
        
        for i in self.annotations['annotations'][:10]:
            image_id = str(i['image_id'])

            name = list("000000000000")
            name[-len(image_id):] = image_id
            name = "".join(name) + ".jpg"

            self.image_bbox[name].append({
                'category_id': i['category_id'],
                'bbox': i['bbox']
            })
        
        

if __name__ == "__main__":
    example = Coco2Yolo()
    print(json.dumps(example.id_category))
    #print(json.dumps(example.image_bbox))
    