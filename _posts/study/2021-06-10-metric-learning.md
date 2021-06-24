---
title: Metric Learning, Deep Metric Learning이란?
excerpt: Diving deep into metric based deep learning.
categories: study
use_math: true
---

## 목차
- [목차](#목차)
- [필요 지식](#필요-지식)
- [간략 소개 (TL;DR)](#간략-소개-tldr)
- [Few Shot Learning 관점에서 살펴보는 Metric Learning의 필요성](#few-shot-learning-관점에서-살펴보는-metric-learning의-필요성)
  - [Verification 작업이 어려운 이유](#verification-작업이-어려운-이유)
- [Metric:](#metric)
  - [Euclidean Metric](#euclidean-metric)
  - [Mahalanobis Distance Metric](#mahalanobis-distance-metric)
  - [미리 정의된 Distance Metric의 한계](#미리-정의된-distance-metric의-한계)
- [Metric Learning](#metric-learning)
- [Deep Metric Learning](#deep-metric-learning)
  - [손실 함수 설계](#손실-함수-설계)
    - [Contrastive Loss](#contrastive-loss)
    - [Triplet Loss](#triplet-loss)
    - [문제점](#문제점)
- [Insight](#insight)

## 필요 지식

- 기초 선형 대수
- 기계학습/딥러닝 기본 개념


## 간략 소개 (TL;DR)

Metric Learning은 간략하게, 데이터 간의 유사도를 잘 수치화하는 거리 함수(metric function)를 학습하는 것이다.

Metric Learning을 통해 학습한 metric function은 clustering, few shot learning 등 여러 가지 분야에 쓰인다.

## Few Shot Learning 관점에서 살펴보는 Metric Learning의 필요성

![Face Verification](https://media.gettyimages.com/vectors/biometrics-of-a-woman-face-detection-recognition-and-identification-vector-id1131606997?k=6&m=1131606997&s=612x612&w=0&h=gzPfYEsjvJ95VICACTO0uPDo-itGdWqsWJyjUPMRv84=)

[Few shot learning](https://dongkyuk.github.io/study/so-many-learnings/), 
그중에서도 Face Verification의 예시를 통해 Metric Learning의 필요성을 살펴보자.

먼저 얼굴 인증(Face Verification)이란, 주어진 얼굴 이미지 쌍이 동일한 사람에 속하는지를 결정하는 작업이다. 
스마트폰 잠금을 열 때 FaceID를 생각하면 쉽다.

### Verification 작업이 어려운 이유

위의 Verification task는 얼굴 두개(등록한 얼굴, 인증시 얼굴)가 같은 클래스인지 아닌지 classify 한다는 면에서 근본적으로 Image Classification task이다.

또한, 일반적으로 신규 유저가 새로운 얼굴을 등록한다면, 많아야 10 몇 개 정도의 얼굴 사진을 데이터 세트에 추가할 수 있을 것이다.

그렇기에 Downstream task (신규 유저 classification)에서 데이터를 few samples(몇개)만 사용한다는 면에 [Few shot learning](https://dongkyuk.github.io/study/so-many-learnings/) task이기도 하다.

여기서 큰 문제가 있다. 딥러닝 모델은 데이터양에 성능의 의존성이 높다. 적은 양의 데이터로 학습할 경우 과적합(overfitting)으로 이어질 가능성이 매우 크다.

이러한 Few shot learning의 근본적인 문제를 해결하고자 [Meta learning](https://dongkyuk.github.io/study/so-many-learnings/) 방법론을 적용할 수 있는데,
오늘 알아보고자 하는 Metric Learning은 이런 Meta Learning approach 중 하나로 쓰일 수 있다.

## Metric:
Metric Learning에서 Metric이란 두 점 $x$와 $y$간의 Non-negative 함수로 ($d(x,y)$라고 하자), 이 두 점 사이의 이른바 '거리' 개념을 설명한다. 

Metric이 충족해야 하는 몇 가지 속성은 다음과 같다:

- *Non-negativity*

$d(x,y) \geqq 0$ and $d(x,y) = 0$, iff $x = y$

- *Triangular inequality*

$d(x,y) \leqq d(x,z) + d(z,y)$

- *Symmetry*

$d(x,y) = d(y,x)$

위의 조건들을 만족하는 대표적인 예시를 몇 가지 보자. (이 부분은 우리가 보려 하는 Deep Metric Learning과 아주 밀접한 관계는 아니므로 넘어가도 무방하다)

### Euclidean Metric
먼저 우리가 잘 아는 Euclidean Metric이 있다.

$$d({x}_1, {x}_2) = \sqrt{\sum_{i=1}^{d} ({x}_{1, i} - {x}_{2, i})^2}$$

다만 Euclidean Metric은 몇 가지 단점([자세한 설명](https://www.machinelearningplus.com/statistics/mahalanobis-distance/))들로 인해 고차원 데이터에서 사용이 힘들다. 간단하게 말하자면, 유클리드 거리는 클래스 간의 상관관계를 고려하지 않고 isotropic(모든 방향에서 동일) 하다.

이에 우리는 차원 간 관계를 캡처하는 non-isotropic 거리를 사용할 수 있다.  

### Mahalanobis Distance Metric

![Isotropic Euclidean distance V/S Non-isotropic Mahalanobis distance metric](https://miro.medium.com/max/1400/1*7CHW-oUiEkyk4_gHysXbvg.png)

Mahalanobis Distance Metric이 그러한 metric 중 하나다. 

$$d(x_1,x_2) = \sqrt{((x_1-x_2)^T M(x_1,x_2))}$$

여기서 $M$은 공분산 행렬의 역행렬이며 Euclidean Distance 제곱에 대한 가중치 항으로 작용한다. Mahalanobis 거리 방정식은 dimension 간의 관계를 decorrelate 시킨 후 계산한 Euclidean Metric이라 볼 수 있다. ([추가설명](https://stats.stackexchange.com/questions/326508/intuitive-meaning-of-vector-multiplication-with-covariance-matrix))

$M$이 Identity Matrix의 경우, Mahalanobis Distance가 Euclidean Distance와 동일하게 되는데, Euclidean Distance가 근본적으로 차원들이 서로 독립적이라는 가정을 내재하고 있다는 것을 알 수 있다.

### 미리 정의된 Distance Metric의 한계

위의 Euclidean Metric, Mahalanobis Distance Metric을 이미지 픽셀 벡터에 그대로 적용해서 classification을 할 수 있을까? 

다시 말해, Euclidean Metric을 바로 쓸 경우, pixel mse가 낮으면 같은 얼굴이고 크면 다른 얼굴일까?

*당연히 아니다.*

위에 본 미리 정의한 Distance Metric들은 데이터와 task에 대한 고려가 적기 떄문에 우리 용도에 적합하지 않다.(Mahalanobis Distance조차도 공분산을 고려한 Linear Transformation을 한 것일 뿐이다)

이러한 이유로 데이터에 적합한 거리 함수를 기계학습으로 직접 만드는 것이 Metric learning이며, 기계학습 중에서도 딥러닝을 활용하는 경우가 Deep Metric Learning이다.

## Metric Learning

기본적인 기계학습은 무엇을 하는지 생각해보자. 일반적으로 기계학습에서 우리는 데이터와 해당 라벨을 주면, 해당 입력 정보를 라벨에 매핑하는 규칙 집합이나 복잡한 함수를 고안한다.

마찬가지로 Metric Learning의 목표는 데이터로부터 Metric function을 학습하는 것이다. 

이를 달성하고자 보통 우리는 original feature space를 거리 계산하기 쉬운 embedding space로 mapping하는 embedding 함수 $f$를 학습한다. 

이렇게 학습된 Metric function $d$는 

$$d(x_1,x_2)=d_e(f(x_1),f(x_2))$$

이다. 여기서 $d_p$는 euclidean distance, cosine similarity 등 미리 정의한 embedding 간의 거리 함수이다.
 
![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fdpuky0%2FbtqIjeVyxZo%2FSnmmbKkMGT6aD1JSWybngk%2Fimg.png)
  
## Deep Metric Learning

드디어 본 주제인 Deep Metric Learning에 도달했다.

위에서 original feature space를 거리 계산하기 쉬운 embedding space로 mapping 하는 embedding 함수 $f$를 학습한다고 했는데, 어떻게 학습을 할까?

embedding 간의 거리 함수가 $d_e$, $x$와 $y$는 input data, $f$는 embedding 함수일 때, Classification task에서 우리는 다음과 같이 목표를 잡고 학습을 한다.

- $x$와 $y$가 동일 클래스일 때 $d_e(f(x), f(y))$를 작게 만든다
- $x$와 $y$가 다른 클래스일 경우, $d_e(f(x), f(y))$를 크게 만든다

Deep Metric Learning에서 이를 달성할 수 있는 두 가지 방법이 있다.

### 손실 함수 설계

첫 번째는 위의 목표에 알맞은 손실 함수 (Loss function)을 사용하는 것이다. 

그중에서도 대표적인 Contrastive Loss와 Triplete Loss를 알아보자.

#### Contrastive Loss
  
먼저 Siamese Network가 무엇인지 보자. 

Siamese Network는 두 개의 동일한 하위 네트워크(shared parameters)로 구성된 대칭 신경망 아키텍처이다 (아래 그림 참고).

Siamese Network는 주로 위의 목표를 달성하고자 Contrastive Loss와 함께 사용되는데, Contrastive Loss의 식을 살펴보면,

$$L(f(x), f(y), z) = z*d_e(f(x), f(y)) + (1-z)\text{max}(0, m-d_e(f(x), f(y)))$$

여기서, $m$은 margin이며, $z$는 ```int(class(x)==class(y))``` 이다.

![Siamese net](https://www.pyimagesearch.com/wp-content/uploads/2020/11/keras_siamese_networks_process.png)

식을 살펴보면, $x$와 $y$가 같은 클래스일 경우 $L(f(x), f(y), z) = d_e(f(x), f(y))$이기에, 같은 클래스 간 거리를 좁히는 방향으로 학습된다. 

반대로, $x$와 $y$가 다른 클래스일 경우 $L(f(x), f(y), z) = \text{max}(0, m-d_e(f(x), f(y)))$이기에, 다른 클래스 간 거리를 넓히는 방향으로 학습된다.

m이 무엇인지 의아할 텐데, $x$와 $y$가 다른 클래스인데 이미 거리가 충분히 먼 경우($d(f(x), f(y)) >= m$)일 때는 loss가 0이이 되어 더 상관하지 않도록 한다. 

이는 모델이 어려운 부분에 더 집중할 수 있게 해 학습을 도운다.


#### Triplet Loss
  
Triplet Network 또한 Siamese Network와 유사하지만 2개가 아닌 3개의 동일한 하위 네트워크로 구성된다. 

3가지 input을 활용하는데, 다음과 같다:

1. Anchor 
2. Positive (Anchor와 동일한 클래스에 속하는 샘플)
3. Negative (Anchor와 다른 클래스에 속하는 샘플)
   
![Triplet loss](https://miro.medium.com/max/2800/1*MIPdyhJGx6uLiob9UI9S0w.png)

식을 살펴보면,

$$L(f(a), f(p), f(n)) = \text{max}(0, d(f(a), f(p)) - d(f(a), f(n)) +m)$$

여기서, $m$은 margin이며, $a$,$p$,$n$은 각각 Anchor, Positive, Negative이다.

$d(f(a), f(p)) - d(f(a), f(n)$를 보면 같은 클래스 간 거리를 좁히고, 다른 클래스 간 거리를 넓히는 방향으로 학습된다는 것을 알 수 있다.

위의 Contrastive Loss와 비슷하게 $m$을 설정을 하는데, Triplet loss에서는 Positive Distance와 Negative Distance 간의 거리 차이가 $m$ 이상일 경우 loss에 반영을 시키지 않게 해서 학습을 도운다.

#### 문제점 
  
Contrastive Loss와 Triplet Loss는 직관적이라는 장점이 있지만, 효과적으로 학습하기 위해서는 mining을 통해서 학습에 좋은 (즉, 어려운) pair/triplet을 찾아야 한다는 단점이 있다.

## Insight

지금까지 Metric Learning의 필요성과 Deep Metric Learning에서 사용하는 loss들에 대해 알아보았다.  

2부에서는 Classification model 변형을 통해 하는 Deep Metric Learning에 대해 알아보자.