# Object Detection任务中tools
这里收集Object Detection任务中一些常用的代码片段，每个代码片段实现一个功能。
### 0. 基础知识
- YOLO格式数据集的bbox为(x_center, y_center, w, h)，而COCO格式数据集的bbox为(x_left, y_top, w, h)。
### 1. 可视化某一张image对应的label
前提：数据集格式为YOLO格式。  
需要修改对应的image路径和label路径，运行后会得到一张包含bounding boxes的图像。
```python
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


# 从txt文件加载标签
def load_labels_from_txt(txt_file):
    labels = []
    with open(txt_file, 'r') as file:
        for line in file:
            label = line.strip().split(' ')
            labels.append(label)
    return labels


# 在图像上绘制bounding box和标签
def draw_boxes(image, labels, class_names, font_path='arial.ttf'):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, 16)

    for label in labels:
        class_index = int(label[0])
        x_center, y_center, width, height = map(float, label[1:])

        # 计算bounding box的左上角和右下角坐标
        x1 = (x_center - width / 2) * image.width
        y1 = (y_center - height / 2) * image.height
        x2 = (x_center + width / 2) * image.width
        y2 = (y_center + height / 2) * image.height

        # 绘制bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # 计算标签显示位置
        label_text = class_names[class_index]
        label_bbox = draw.textbbox((0, 0), label_text, font=font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]
        label_x = x1
        label_y = max(y1 - label_height, 0)

        # 绘制标签背景
        bg_color = "red"
        draw.rectangle([label_x, label_y - 2, label_x + label_width, label_y + label_height], fill=bg_color)

        # 在标签背景上显示标签
        draw.text((label_x, label_y - 5), label_text, fill="white", font=font)

    return image


# 图像路径
image_path = "E:/supp/000000367195.jpg"
# 标签文件路径
label_file = "E:/supp/000000367195.txt"
# 类别名称列表
class_names = ['person', 'class2', 'class3']  # 替换成您的类别名称列表

# 加载图像
img = Image.open(image_path)

# 从txt文件加载标签
labels = load_labels_from_txt(label_file)

# 绘制bounding box和标签
output_image = draw_boxes(img.copy(), labels, class_names)

# 保存结果图像
output_image.save('results/000000367195.png')

# 显示结果图像
plt.figure(figsize=(10, 10))
plt.imshow(output_image)
plt.axis('off')
plt.show()
```

