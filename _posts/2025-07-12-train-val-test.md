---
title: 关于深度学习中训练集、验证集和测试集的说明与实践
author: david
date: 2025-07-12 14:00:00
categories: [Blogging, Tutorial]
tags: [deep learning]
toc: false
math: true
---

## 关于深度学习中训练集(Training Set)、验证集(Validation Set)和测试集(Test Set)的说明与实践
在刚开始接触深度学习时，训练集(Training Set)、验证集(Validation Set)和测试集(Test Set)是需要第一时间理解的概念，因为这涉及到深度学习的基本运行原理。

定义: 一个深度学习模型为$f$, 输入为$X$, 输出为$Y'$, 真实标签(GT, Ground Truth)为$Y$, $f$的可学习参数, 即优化参数为$\theta$。那么训练的过程就是:

$\theta^* = \arg\min_\theta \mathcal{L}(f_\theta(X), Y).$

这里$\mathcal{L}$是一个损失函数(loss function)，用于定量模型预测结果$Y'$和真实标签$Y$之间的距离，例如交叉熵损失(cross-entropy loss)或MSE损失(mean squared error loss)。

## 训练集(Training Set)、验证集(Validation Set)和测试集(Test Set)
### 1. 训练集(Training Set)
训练集是用来进行模型参数优化的数据集合，即实际用于进行上面公式中的优化过程。通常，训练集占整个数据集的70%-80%。
### 2. 验证集(Validation Set)
验证集是用于在模型训练过程中评估模型表现的，帮助调整设置不可进行训练的超参数（hyperparameters），例如学习率、模型结构等。通常验证集占数据的10%-15%。
### 3. 测试集(Test Set)
测试集不参与训练过程，也不参与超参数调整，它是一个用来估计模型最终效果的独立集合。通常测试集占数据的10%-15%。

---
如果把深度学习模型的学习过程类比为一个小学生学习的话，训练集就相当于课本，小学生可以通过阅读课本从里面学习知识。验证集相当于习题，用来检验小学生知识掌握得怎么样。而测试集相当于考试试卷，用来考察小学生的真实水平。

---
注意：训练集、验证集和测试集的数据互相不重叠。训练集和验证集参与模型训练过程，而测试集是不参与模型训练的。

下面用一段代码展示训练深度学习模型时三个数据集合的使用逻辑:

```python
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 训练model阶段
for e in range(opoch):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    print(f"Epoch {epoch+1}: Val Loss: {val_loss:.4f}, Val Acc: {correct / len(val_dataset):.4f}")

# model推理阶段, 也是测试model精度阶段
model.eval()
test_correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        test_correct += (preds == labels).sum().item()

print(f"Test Accuracy: {test_correct / len(test_dataset):.4f}")
```

