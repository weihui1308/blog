---
title: 在浏览器中播放本地视频遇到的问题
author: david
date: 2024-12-05 14:00:00
categories: [Blogging, Tutorial]
tags: [video]
---


# 在浏览器中播放本地视频遇到的问题
### 起因
由于项目展示需要，我制作了一个html页面在浏览器里播放本地的视频。但是一切准备就绪后，发现页面无法加载和播放指定的视频。

查找资料后发现是因为视频编码的问题，如果想要在浏览器里播放，视频的编码需要是'H264'，而我使用的是'mp4v'。

我使用的生成展示视频的代码如下（主要功能是读取视频帧，用YOLO检测器进行检测并可视化结果，然后把这些包含检测结果的视频帧合成一个视频）：
```python
import cv2
from ultralytics import YOLO
import argparse
from ultralytics.utils.plotting import Annotator, colors

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--source', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--imgsz', type=int, default=640)
    opt = parser.parse_args()
    
    source = opt.source
    model = opt.model
    conf = opt.conf
    output_path = opt.save_path
    imgsz = opt.imgsz

    # Load a pretrained YOLO11n model
    model = YOLO(model)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"can't open: {source}")
        exit(1)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        for result in model(source=frame, conf=conf, imgsz=imgsz, stream=True):
            frame = result.plot()
        
        out.write(frame)
        
    cap.release()
    out.release()
    print(f"save to: {output_path}")
```

### 经过
因此我修改
```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
```
为
```python
fourcc = cv2.VideoWriter_fourcc(*'H264')
```
这里发生了一件不可思议的事情。修改后的代码成功运行，但是却没有保存视频文件。因为代码没有报错，我以为是保存路径设置得有问题，但多次检查参数确认无误后，还是没有保存视频文件。

查找资料得出：我使用的机器可能缺少必要的依赖，导致不支持'H264'编解码器，因此无法保存视频文件。

### 结果
我又把编码改了回去，重新使用'mp4v'，运行代码生成视频文件。

然后在机器上安装了FFmpeg, 在命令行中执行：
```shell
sudo apt install ffmpeg libavcodec-extra
```
安装成功后使用FFmpeg将视频从'mp4v'转码到'H264'，在命令行中执行：：
```shell
ffmpeg -i "./runs1/yolov11.mp4" -vcodec libx264 -crf 20 "./runs/yolov11.mp4"
```
这样得到的视频可以成功地在浏览器里播放。

### 总结
这里有一个比较坑人的是代码不报错，但是成功运行后不保存视频文件。首次碰到这种问题难以排查，我刚开始是怀疑自己路径写错了，折腾了一会儿才发觉是机器依赖缺失的问题。














