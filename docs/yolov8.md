# 使用YOLOv8的记录
[YOLOv8](https://github.com/ultralytics/ultralytics)是ultralytics公司继YOLOv3和YOLOv5推出的新一代YOLO模型，具有处理Classify，Detect，Segment，Track，Segment五大任务的能力。本文只关注在Detect任务上的使用。在调试过程中发现，YOLOv8较之前版本的YOLO模型更加易用，但代价是封装度更高了。下面记录使用过的代码片段和注释。
## YOLOv8的安装
先装一些基础的包，例如PyTouch，具体参看[官方仓库的readme](https://github.com/ultralytics/ultralytics)。  
然后执行下面的pip安装语句即可完成。
```bash
pip install ultralytics
```

## 使用YOLOv8做推理

```python
from ultralytics import YOLO

# Load a model
model = YOLO("/home/checkpoints/yolov8s.pt")  # load a pretrained model

# Use the model
results = model.predict(source="/home/images/000000027.jpg", save=True, line_width=3)  # predict on an image
# source可以为多种格式：img.jpg，0(webcam)，video.mp4，path等。
# save=True，将检测结果保存在runs/predict/目录下。
# model.predict()方法的参数列表可以从ultralytics/cfg/default.yaml文件中查看，根据需要进行设置

# Check the results
for result in results:
    # Detection
    result.boxes.xyxy   # box with xyxy format, (N, 4)
    result.boxes.xywh   # box with xywh format, (N, 4)
    result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
    result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
    result.boxes.conf   # confidence score, (N, 1)
    result.boxes.cls    # cls, (N, 1)
```

## 使用YOLOv8在数据集上做测试
```python
from ultralytics import YOLO

model = YOLO("/home/checkpoints/yolov8s.pt")

metrics = model.val(data='config/dataset.yaml', save_json=True)
# model.val()方法的参数列表可以从https://docs.ultralytics.com/modes/val/#arguments-for-yolo-model-validation中查看，根据需要进行设置
# 测试结果会保存在runs文件夹下
# save_json=True，会将检测结果保存在一个json文件里，便于后续的分析

metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category
```

## 使用YOLOv8在数据集上做训练
```python
from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8s.yaml")  # build a new model from YAML
model = YOLO("checkpoints/yolov8s.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="config/custom.yaml", epochs=100, imgsz=640)
```

