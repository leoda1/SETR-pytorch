## SETR和VIT语义分割模型在Pytorch当中的实现
---
### 目录
1. [所需环境 Environment](###所需环境)
2. [训练步骤 How2train](#训练步骤)
3. [预测步骤 How2predict](#预测步骤)
4. [评估步骤 miou](#评估步骤)
5. [参考资料 Reference](#Reference)

### 所需环境
见requirements.txt
```shell
conda activate yourenv
pip install -r requirements.txt
```
### 训练步骤
1. 加载你的数据集，数据集放置到VOCdeckit路径下。结构如下：
- VOCdevkit
  - VOC2007
    - ImageSets
      - Segmentation
        - train.txt
        - val.txt
    - JPEGImages
      - image.jpg 
    - SegmentationClass
      - label.png

2. 说明
    - num_classes设为你的数据集的类别数+1
    - optimizer_type设为你需要的，可选：sgd，adam
    - 如果训练SETR模型 则backbone = 'setr'，model = SETR(num_classes=num_classes, backbone=backbone, pretrained=pretrained)
    - 如果训练Vit模型 则backbone = ‘vit’, model = VIT(num_classes=num_classes, backbone=backbone, pretrained=pretrained)
    - 运行train.py

### 预测步骤
logs目录下会保存训练的best_weights.pth,。。。。。。。。还没完全弄完 继续更新中
