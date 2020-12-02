---
layout: post
title: "Deep Residual Network (2)"
tagline: "Deep Residual Network의 적용"
---

이 포스트에서는 Deep Residual Network를 적용한 [Pytorch Tutorial](https://github.com/yunjey/pytorch-tutorial.git) 코드를 분석해본다. ([코드 원본](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py))

[Deep Residual Network (1) 에서 정리한 논문](https://arxiv.org/pdf/1512.03385.pdf)의 4.2 항, CIFAR-10 에 대한 모델 구조를 구현한 코드이다. 또한 일부는 [Pytorch/vision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) 에서 제공하는 코드를 참조했다고 한다.

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
```
가장 먼저 필요한 package 들을 불러온다. 이전 다른 코드에도 사용되었던, pytorch를 사용해 학습을 진행할 때에 가장 기본적으로 사용되는 package 들이므로 별도의 설명은 생략한다.
<br>
<br>
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
학습에 사용할 device를 정의한다. gpu를 사용할 수 있다면 cuda, 그렇지 않은 경우에는 cpu로 정의한다. 이 과정을 거치는 것은 나중에 데이터와 모델을 device에 할당해주는 과정에서 매번 코드를 cuda 와 cpu로 변경하지 않고도 사용할 수 있게 하기 위함이다.
<br>
<br>
```python
num_epochs = 80
batch_size = 100
learning_rate = 0.001
```
학습할 epoch의 횟수, 데이터를 나누는 batch의 크기, learning rate 의 값을 설정해준다. 얼마든지 필요에 의해 변경해서 사용할 수 있는 변수들이다. 이 역시 변수를 미리 설정해 매번 번거롭게 숫자로 변경하는 작업을 하지 않도록 하기 위함이다.
<br>
<br>
```python
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])
```
학습에 사용될 이미지를 전처리하는 모듈을 미리 정의해준다.
- torchvision.transforms.Compose(transforms) : Transform objects list를 입력으로 받아 이들을 묶어서 순차적으로 실행해주는 함수이다. 단, torchscript를 지원하지 않으므로 필요한 경우에는 같은 기능을 하는 torch.nn.Sequential 을 사용하도록 하자.
- torchvision.transforms.Pad(padding, fill=0, padding_mode='constant') : 이미지의 상하좌우에 입력한 설정값만큼 padding을 삽입해주는 함수이다.
- torchvision.transforms.RandomHorizontalFlip(p=0.5) : 입력된 확률 값만큼 random하게 이미지를 좌우반전 시키는 함수이다.
- torchvision.transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant') : 입력된 사이즈에 맞춰 이미지를 random하게 잘라주는 함수이다.
- torchvision.transforms.ToTensor : 최종 처리된 데이터를 Tensor 데이터로 변환해주는 함수이다.
<br>
<br>
```python
```
학습과 모델 성능 평가에 사용할 이미지 데이터인 CIFAR-10 데이터셋을 각각 train과 test로 나누어 불러온다. train dataset은 위에서 정의했던 전처리 모듈을 사용하고, test는 ToTensor를 이용해 Tensor 데이터로 변환만 해준다.
<br>
<br>
```python
```
<br>
<br>
```python
```
<br>
<br>
```python
```
<br>
<br>
<출처>
[Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. (2016). Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, 770-778](https://arxiv.org/pdf/1512.03385.pdf)
[Pytorch Tutorial](https://github.com/yunjey/pytorch-tutorial.git)
[Pytorch Tutorial - Deep Residual Network](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py)
[Pytorch/vision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
[Pytorch Docs - torchvision.transforms](https://pytorch.org/docs/stable/torchvision/transforms.html)
