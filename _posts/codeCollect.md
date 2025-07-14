# 常用代码片段收集
从平时coding中收集一些常用的代码片段，记录在这里，方便以后复用。
### 1. 项目代码入口
```python
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov5s')
    parser.add_argument('--conf', type=float, default=0.5)
    return parser.parse_args()

def main(opt):
    print(opt.model)
    print(opt.conf)
    
if __name__ == '__main__':
    opt = parse_args()
    main(opt)
```
