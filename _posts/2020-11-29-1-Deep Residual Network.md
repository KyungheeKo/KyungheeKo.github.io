---
layout: post
title: "Deep Residual Network"
tagline: "Deep Residual Network 개념과 적용"
---

이 포스트에서는 Deep Residual Network의 개념에 대해 간단하게 정리하고 이를 적용한 [Pytorch Tutorial](https://github.com/yunjey/pytorch-tutorial.git) 코드를 분석해본다.

![resnet1](https://github.com/KyungheeKo/KyungheeKo.github.io/blob/KyungheeKo/assets/img/resnet/resnet1.png)

위 그래프는 CIFAR-10 데이터를 각각 20-layer와 56-layer의 plain network를 이용해 분류했을 때의 Training error와 test error를 나타낸 것이다. 이 그래프에서도 알 수 있듯이 단순한 형태의 Neural Network는 깊어지면 깊어질수록 어느 시점까지는 성능이 개선되지만, 이후로는 오히려 정확도가 떨어지는 결과를 보인다.


출처 : [Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. (2016). Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, 770-778](https://arxiv.org/pdf/1512.03385.pdf)
