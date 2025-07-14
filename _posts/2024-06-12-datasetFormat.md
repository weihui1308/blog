---
title: Object Detection任务中的数据集格式转换
author: david
date: 2024-06-12 14:00:00
categories: [Blogging, Tutorial]
tags: [detection]
---

# Object Detection任务中的数据集格式转换
目前主流的Object Detection模型，如YOLO系列(YOLOv3, YOLOv5, YOLOv7, YOLOv8)，Faster R-CNN，Mask R-CNN，和DETR等，在训练和验证阶段，数据集格式有不同。

YOLO系列采用的是YOLO格式数据集，MMDetection实现的Faster R-CNN等模型采用的是COCO格式数据集。

这里记录一些不同数据集格式之间转换的代码。

## YOLOv5格式转换为COCO格式
YOLOv5格式数据集的目录结构为：  
- dataset
  - images
    - train
      - img1.jpg
      - img2.jpg
      - ...
    - val
      - img3.jpg
      - img4.jpg
      - ...
  - labels
    - train
      - img1.txt
      - img2.txt
      - ...
    - val
      - img3.txt
      - img4.txt
      - ...

---

COCO格式数据集
- dataset
    - train
      - img1.jpg
      - img2.jpg
      - ...
    - val
      - img3.jpg
      - img4.jpg
      - ...
    - annotations
      - train.json
      - val.json
      
---

转换代码如下：
```python
import os
import json
import cv2

def yolo_to_coco(yolo_file, img_id, ann_id_start, img_width, img_height):
    annotations = []
    with open(yolo_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            class_id, center_x, center_y, width, height = map(float, line.strip().split())
            # Convert YOLO format (normalized) to COCO format (absolute)
            x = (center_x - width / 2) * img_width
            y = (center_y - height / 2) * img_height
            w = width * img_width
            h = height * img_height
            bbox = [x, y, w, h]
            area = w * h
            annotations.append({
                "id": ann_id_start,
                "image_id": img_id,
                "category_id": int(class_id),
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            })
            ann_id_start += 1
    return annotations, ann_id_start

def create_coco_json(img_folder, yolo_folder, output_file, categories):
    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    for img_file in os.listdir(img_folder):
        if img_file.endswith(".jpg") or img_file.endswith(".png"):
            img_path = os.path.join(img_folder, img_file)
            img = cv2.imread(img_path)
            height, width, _ = img.shape

            images.append({
                "id": img_id,
                "width": width,
                "height": height,
                "file_name": img_file
            })

            yolo_file = os.path.join(yolo_folder, os.path.splitext(img_file)[0] + ".txt")
            if os.path.exists(yolo_file):
                img_annotations, ann_id = yolo_to_coco(yolo_file, img_id, ann_id, width, height)
                annotations.extend(img_annotations)

            img_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=4)

# Define your categories here
categories = [
    {"id": 0, "name": "person", "supercategory": "none"},
    # Add more categories as needed
]

# Paths to the image folder, YOLO annotations folder and output json file
img_folder = "/home/dataset/images/val"
yolo_folder = "/home/dataset/labels/val"
output_file = "./val_coco_annotations.json"

create_coco_json(img_folder, yolo_folder, output_file, categories)

```
该python脚本将训练集和验证集分开转换，需要手动修改对应的路径。categories为数据集中的类别信息，根据需要进行添加和删减。

## YOLOv5格式转换为YOLOv7格式
官方YOLOv7库支持的数据集格式和YOLOv5有一点小的区别，可以从二者数据集配置文件的差异对比查看。

YOLOv7的数据集配置文件：
```python
# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: ./coco/train2017.txt  # 118287 images
val: ./coco/val2017.txt  # 5000 images
# test: ./coco/test-dev2017.txt  # 20288 of 40670 images, submit to https://competitions.codalab.org/competitions/20794

# number of classes
nc: 80

# class names
names: [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
```
YOLOv5的数据集配置文件：
```python
path: /home/dataset/
train: images/train
val: images/val
# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
```
对比可见，不同于YOLOv5指定数据集中images的路径，YOLOv7改为指定包含有images路径的txt文件。因此需要准备这样一个txt文件，代码如下：
```python
import os

def list_files_in_directory(directory, output_file):
    """
    List all files in the given directory and its subdirectories,
    and write their paths to the output file.
    
    Args:
    directory (str): The root directory to start listing files from.
    output_file (str): The file to write the paths to.
    """
    with open(output_file, 'w') as file:
        for root, _, files in os.walk(directory):
            for name in files:
                file_path = os.path.join(root, name)
                file.write(file_path + '\n')

# Example usage:
directory = "/home/dataset/images/val/"
output_file = './dataset/val_with_patch.txt'

list_files_in_directory(directory, output_file)

```
虽然经历了这样的改动，但image和label的相对位置仍不变，为YOLOv5格式数据集的目录结构。

## COCO格式转换为YOLOv5格式
转换代码如下：
```python
import json
import os
import argparse
import numpy as np
parser = argparse.ArgumentParser(description='Test yolo data.')
parser.add_argument('-j', help='JSON file', dest='json', required=True)
parser.add_argument('-o', help='path to output folder', dest='out',required=True)

args = parser.parse_args()

json_file = args.json 
output = args.out 
class COCO2YOLO:
    def __init__(self):
        self._check_file_and_dir(json_file, output)
        self.labels = json.load(open(json_file, 'r', encoding='utf-8'))
        self.coco_id_name_map = self._categories()
        self.coco_name_list = list(self.coco_id_name_map.values())
        print("total images", len(self.labels['images']))
        print("total categories", len(self.labels['categories']))
        print("total labels", len(self.labels['annotations']))

    def _check_file_and_dir(self, file_path, dir_path):
        if not os.path.exists(file_path):
            raise ValueError("file not found")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def _categories(self):
        categories = {}
        for cls in self.labels['categories']:
            categories[cls['id']] = cls['name']
        return categories

    def _load_images_info(self):
        images_info = {}
        for image in self.labels['images']:
            id = image['id']
            file_name = image['file_name']
            if file_name.find('\\') > -1:
                file_name = file_name[file_name.index('\\')+1:]
            w = image['width']
            h = image['height']
            images_info[id] = (file_name, w, h)

        return images_info

    def _bbox_2_yolo(self, bbox, img_w, img_h):
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        centerx = bbox[0] + w / 2
        centery = bbox[1] + h / 2
        dw = 1 / img_w
        dh = 1 / img_h
        centerx *= dw
        w *= dw
        centery *= dh
        h *= dh
        return centerx, centery, w, h

    def _convert_anno(self, images_info):
        anno_dict = dict()
        for anno in self.labels['annotations']:
            bbox = anno['bbox']
            image_id = anno['image_id']
            category_id = anno['category_id']

            image_info = images_info.get(image_id)
            image_name = image_info[0]
            img_w = image_info[1]
            img_h = image_info[2]
            yolo_box = self._bbox_2_yolo(bbox, img_w, img_h)

            anno_info = (image_name, category_id, yolo_box)
            anno_infos = anno_dict.get(image_id)
            if not anno_infos:
                anno_dict[image_id] = [anno_info]
            else:
                anno_infos.append(anno_info)
                anno_dict[image_id] = anno_infos
        return anno_dict

    def save_classes(self):
        sorted_classes = list(map(lambda x: x['name'], sorted(self.labels['categories'], key=lambda x: x['id'])))
        print('coco names', sorted_classes)
        with open('coco.names', 'w', encoding='utf-8') as f:
            for cls in sorted_classes:
                f.write(cls + '\n')
        f.close()

    def coco2yolo(self):
        print("loading image info...")
        images_info = self._load_images_info()
        print("loading done, total images", len(images_info))

        print("start converting...")
        anno_dict = self._convert_anno(images_info)
        print("converting done, total labels", len(anno_dict))

        print("saving txt file...")
        self._save_txt(anno_dict)
        print("saving done")

    def _save_txt(self, anno_dict):
        
        for k, v in anno_dict.items():
            file_name = v[0][0].split(".")[0] + ".txt"
            
            # -----
            for obj in v:
                cat_name_tmp = self.coco_id_name_map.get(obj[1])
                #print(cat_name_tmp)
                if cat_name_tmp != 'person':
                    continue
                if np.array(obj[2]).all() > 0:
                    pass
                else:
                    continue
                if obj[2][-1] > 0.2:
                    pass
                else:
                    continue
            
                with open(os.path.join(output, file_name), 'a+', encoding='utf-8') as f:
                    #print(k, v)
                    print(obj)
                    cat_name = self.coco_id_name_map.get(obj[1])
                    category_id = self.coco_name_list.index(cat_name)
                    box = ['{:.6f}'.format(x) for x in obj[2]]
                    box = ' '.join(box)
                    line = str(category_id) + ' ' + box
                    f.write(line + '\n')
        


if __name__ == '__main__':
    c2y = COCO2YOLO()
    c2y.coco2yolo()
```
该代码支持根据标注文件中object的height进行筛选，例如只保留height大于0.2的objects。
