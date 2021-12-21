# faster_rcnn
# Faster R-CNN 源码笔记

## 环境配置

```
lxml
matplotlib
numpy==1.17.0
tqdm==4.42.1
torch==1.6.0
torchvision==0.7.0
pycocotools
Pillow
```



## Faster R-CNN框架图

![fasterRCNN](https://gitee.com/zhoujiancong/pic-data/raw/master/uPic/fasterRCNN.png)



## 文件结构：

```
  ├── backbone: 特征提取网络，可以根据自己的要求选择
  ├── network_files: Faster R-CNN网络（包括Fast R-CNN以及RPN等模块）
  ├── train_utils: 训练验证相关模块（包括cocotools）
  ├── my_dataset.py: 自定义dataset用于读取VOC数据集
  ├── train_mobilenet.py: 以MobileNetV2做为backbone进行训练
  ├── train_resnet50_fpn.py: 以resnet50+FPN做为backbone进行训练
  ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  ├── validation.py: 利用训练好的权重验证/测试数据的COCO指标，并生成record_mAP.txt文件
  └── pascal_voc_classes.json: pascal_voc标签文件
```

<img src="../../../Library/Application Support/typora-user-images/image-20211220192830202.png" alt="image-20211220192830202" style="zoom: 50%;" />

==完整文件目录==



## 预训练权重下载地址（下载后放入backbone文件夹中）：

* MobileNetV2 backbone: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
* ResNet50+FPN backbone: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
* 注意，下载的预训练权重记得要重命名，比如在train_resnet50_fpn.py中读取的是```fasterrcnn_resnet50_fpn_coco.pth```文件，
  不是```fasterrcnn_resnet50_fpn_coco-258fb6c6.pth```



## 数据集

Pascal VOC2012 train/val数据集下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

我本地把07和12的Pascal VOC数据集都下了



## 训练方法

- 确保提前准备好数据集
- 确保提前下载好对应预训练模型权重
- 若要训练mobilenetv2+fasterrcnn，直接使用train_mobilenet.py训练脚本
- 若要训练resnet50+fpn+fasterrcnn，直接使用train_resnet50_fpn.py训练脚本
- 若要使用多GPU训练，使用`python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_GPU.py`指令,`nproc_per_node`参数为使用GPU数量
- 如果想指定使用哪些GPU设备可在指令前加上`CUDA_VISIBLE_DEVICES=0,3`
- `CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_GPU.py`



## 原作者提供的注意事项

* 在使用训练脚本时，注意要将'--data-path'(VOC_root)设置为自己存放'VOCdevkit'文件夹所在的**根目录**
* 由于带有FPN结构的Faster RCNN很吃显存，如果GPU的显存不够(如果batch_size小于8的话)建议在create_model函数中使用默认的norm_layer，
  即不传递norm_layer变量，默认去使用FrozenBatchNorm2d(即不会去更新参数的bn层),使用中发现效果也很好。
* 在使用预测脚本时，要将'train_weights'设置为你自己生成的权重路径。
* 使用validation文件时，注意确保你的验证集或者测试集中必须包含每个类别的目标，并且使用时只需要修改'--num-classes'、'--data-path'和'--weights'即可，其他代码尽量不要改动



## 几个主要模块的介绍

### ```backbone```

```backbone```：选择想要使用的特征提取网络

<img src="https://gitee.com/zhoujiancong/pic-data/raw/master/uPic/image-20211219152419197.png" alt="image-20211219152419197" style="zoom:50%;" />



### ```network_files```

Faster R-CNN网络的各个模块

```transform.py```：对图像进行预处理

![标准化处理](https://gitee.com/zhoujiancong/pic-data/raw/master/uPic/image-20211218201531219.png)

![缩放图片大小](https://gitee.com/zhoujiancong/pic-data/raw/master/uPic/image-20211218201617260.png)

**这里对图像进行缩放的方式是，**先求出图片高和宽中的最小值与最大值，根据指定的最小边长与图片的最小边长计算缩放比例，如果使用该缩放比例计算的图片最大边长大于指定的最大边长，则缩放比例为指定的最大边长与图片最大边长的比值。

这样做的意义是尽量保持图片原先的比例，缩放然后复制到预测图中，空出来的地方padding 0，0的地方对图片检测没有影响。

其中还有一个```postprocess```模块，作用是对网络的预测结果进行后处理（主要将bboxes还原到原来的图像尺度上）。



### ==核心：RPN==

在 RPN 中，通过==采用 anchors 来解决边界框列表长度不定的问题==，即，在原始图像中统一放置固定大小的参考边界框. 不同于直接检测 objects 的位置，这里将问题转化为两部分：（1）anchor 是否包含相关的 object？（2）如何调整 anchor 以更好的拟合相关的 object？

针对每个anchors，有两个不同的输出：

（1）anchor 内是某个 object 的概率

* RPN 不关心 object 类别，只确定是 object 还是 background。

（2）anchor 边界框回归输出

* 边界框的输出用于调整 anchors 来更好的拟合预测的 object。



**代码实现：**

```rpn_function.py```：实现整个RPN模块

1. ```class RPNHead(nn.Module):```将backbone输出的图像传入3x3的卷积层，再分别进入分类和回归这两个1x1的卷积层。

   ![image-20211219154607586](https://gitee.com/zhoujiancong/pic-data/raw/master/uPic/image-20211219154607586.png)

   ![image-20211219154545527](https://gitee.com/zhoujiancong/pic-data/raw/master/uPic/image-20211219154545527.png)

2. ```class AnchorsGenerator(nn.Module):```生成anchors，计算所有anchor的尺寸、计算每个预测特征层上每个滑动窗口的预测目标数、计算预测特征图对应的原始图像上的所有anchors的坐标。

   ![image-20211219172013730](https://gitee.com/zhoujiancong/pic-data/raw/master/uPic/image-20211219172013730.png)![image-20211219172301949](https://gitee.com/zhoujiancong/pic-data/raw/master/uPic/image-20211219172301949.png)

3. ```class RegionProposalNetwork(torch.nn.Module):```

   * 计算anchors与真实bbox的iou信息；

   ```class Matcher(object):```

   * 计算每个anchors与gt匹配iou最大的索引（如果iou<0.3索引置为-1，0.3<iou<0.7索引为-2），然后根据是否>=0划分样本与废弃样本。
   * 根据计算出最匹配的gt，并进一步划分为正样本、负样本以及废弃样本（1、0、-1）。

   这里使用clamp设置下限0是为了方便取每个anchors对应的gt_boxes信息，负样本和舍弃的样本都是负值，所以为了防止越界直接置为0。因为后面是通过labels_per_image变量来记录正样本位置的，所以负样本和舍弃的样本对应的gt_boxes信息并没有什么意义，反正计算目标边界框回归损失时只会用到正样本。

   ```
   def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
   ```

   筛除小boxes框，nms处理，根据预测概率获取前post_nms_top_n个目标。



```faster_rcnn_framework.py:```

1. ```class FasterRCNNBase(nn.Module):```

   * 主要框架，对图像进行预处理，然后输入backbone得到特征图。

   * 将特征层以及标注的target信息传入rpn中，得到proposals及其损失。

   * 将rpn生成的数据以及标注target信息传入faster rcnn的后半部分。

   * 对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）。

2. ```class TwoMLPHead(nn.Module):```

   实现这一模块的功能

   ![image-20211220103603561](https://gitee.com/zhoujiancong/pic-data/raw/master/uPic/image-20211220103603561.png)

   ![image-20211220103357634](https://gitee.com/zhoujiancong/pic-data/raw/master/uPic/image-20211220103357634.png)

3. ```calss FastRCNNPredictor(nn.Module):```

   实现分类和边界框回归的预测，该模块的输入是Two MLPHead的输出

   ![image-20211220104825046](https://gitee.com/zhoujiancong/pic-data/raw/master/uPic/image-20211220104825046.png)

   ![image-20211220104711313](https://gitee.com/zhoujiancong/pic-data/raw/master/uPic/image-20211220104711313.png)



```roi_head:```

不是所有的样本都会被采用，需要经过采样操作处理。

```class ROIHeads(torch.nn.Module):```为每个proposal匹配对应的gt_box，并划分到正负样本中。

<img src="../../../Library/Application Support/typora-user-images/image-20211220162802631.png" alt="image-20211220162802631" style="zoom:50%;" />

```fastrcnn_loss```根据预测类别的概率信息和真实类别的信息，预测目标边界框回归信息和真实目标边界框信息，计算损失。

<img src="../../../Library/Application Support/typora-user-images/image-20211220162829980.png" alt="image-20211220162829980" style="zoom:50%;" />

Faster R-CNN没有采用简单的 L1 或 L2 loss 用于回归误差，而是采用 ==Smooth L1 loss==。 Smooth L1 和 L1 基本相同，但是，当 L1 误差值非常小时，表示为一个确定值 σ， 即认为是接近正确的，loss 就会以更快的速度消失。

R-CNN 对每个 proposal 的特征图，拉平flatten，并采用 ReLU 和两个全连接层进行处理：

1. 一个全连接层有 N+1 个神经单元，其中 N 是类别 class 的总数，加上1个 background class；
2. 一个全连接层有 4N 个神经单元.，回归预测输出（四个坐标）。



### ```train_mobilenet.py```

```train_mobilenetv2.py```：以MobileNetV2作为backbone，预训练模型。

```mobilenet_v2.pth```：对应的预训练权重。

预训练mobilenetv2：

![pre-train mobilenetv2](https://gitee.com/zhoujiancong/pic-data/raw/master/uPic/pre-train%20mobilenetv2.png)

本地跑，跑完一个epoch大概要花3个小时，

跑完后把得到的预训练权重存回指定文件夹，供预测阶段使用。



### ```predict.py```

预训练完成之后，运行该文件，进行预测。backbone可以直接用torchvision的包导入，自定义想要使用的网络模型。

![image-20211220191138739](https://gitee.com/zhoujiancong/pic-data/raw/master/uPic/image-20211220191138739.png)



### 评价指标mAP：```coco_eval.py```

```class CocoEvaluator(object):```

里面很多方法是直接用pycocotools这个包的，我对这个包还不太熟悉。

评价准则：指定 IoU 阈值对应的 Mean Average Precision （mAP），如 mAP@0.5，即IoU阈值为0.5得到的mAP.

![image-20211221005015679](../../../Library/Application Support/typora-user-images/image-20211221005015679.png)

![image-20211221005256622](../../../Library/Application Support/typora-user-images/image-20211221005256622.png)
