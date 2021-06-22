---
title:  "Bias Variance Tradeoff"
excerpt: 
categories: study
use_math: true
---
## Bias Variance Tradeoff

Bias-variance Tradeoff는 Supervised Learning에서 Overfitting, Underfitting을 이해하기 위해 핵심적인 개념이다.

## 직관적인 설명

![Bulls-eye](https://www.researchgate.net/profile/Junhua-Ding/publication/318432363/figure/fig3/AS:667604972761097@1536180837464/Visualizing-bias-and-variance-tradeoff-using-a-bulls-eye-diagram.ppm)

보통 Bias, Variance를 설명할 때 많이 보는 Bulls-eye Diagram이다.

위 그림으로 봤을 때, Bias Variance가 어떤 것을 의미하는지 알 수 있다.

- Bias는 우리가 학습한 모델의 expected (average) prediction과 correct value간의 차이이다.  

- Variance는 


## 수식으로 살펴보는 Bias Variance Decomposition

PRML 책에서 나오는 수식들을 바탕으로 Bias Variance Tradeoff를 살펴보자.

Input sample이 $x$, ground-truth value가 $t$, 모델의 prediction이 $y(x)$, Loss Function이 $L$인 경우, Loss의 기댓값은 다음과 같다.

$$E[L] = \iint L(t, y(x))p(x, t)dxdt$$

Regression 문제에서 흔히 쓰이는 Mean Squared Error를 썼다고 가정하면,

$$E[L] = \iint \{y(x)-t\}^2p(x, t)dxdt \qquad{(1)}$$

$\{y(x)-t\}^2$를 $E[t|x]$ 값을 더하고 빼고 해서 식을 전개 해보면, 

$$\begin{aligned} \{y({x})-t\}^2 &= \{y({x})-E[t|{x}]+E[t|{x}]-t\}^2\\\\
&= \{y({x})-E[t|{x}]\}^2 + 2\{y({x})-E[t|{x}]\}\{E[t|{x}]-t\}+\{E[t|{x}]-t\}^2 \qquad{(2)}\end{aligned}$$

$(2)$의 결과를 $(1)$에 대입한 후 양변을 $t$에 대해 적분하면, 

$$\begin{aligned} E[L] & = \int \{y({x})-E[t|{x}]\}^2 p({x}) d{x} + \int \{E[t|{x}] - t\}^2p({x})d{x} \\
& = \int \{y({x})-E[t|{x}]\}^2 p({x}) d{x} + \int var[t|{x}]p({x})d{x}  \end{aligned}$$

이렇게 우리는 Loss를 2개의 텀으로 나눌 수 있다. 각각을 살펴보면,

- 첫번째 텀은 
- 두번째 텀은 $t$의 분산을 $x$에 대해 평균을 구한 것인데, 쉽게 생각하면 샘플이 포함하고 있는 노이즈(noise)를 의미한다.

두번째 텀은 모델링을 잘한다고 해결되는 문제가 아닌, 데이터 자체가 가지고 있는 노이즈이기에 첫번째 텀에 주목을 해보자.

여기서 우리가 상기해야할 사실이 하나있다. 우리에게 주어진 데이터는 고정된 크기 N을 가진 데이터 D (즉, 샘플)이라는 것이다.

그렇기에 우리는 각각의 데이터 집합을 이용하여 예측 함수 y(x;D)를 만들어 낼 수 있다.



첫번째 텀을 위와 유사하게 decompostion을 진행해보자.

$$\{y(x;D)-E_D[y(x;D)]+E_D[y(x;D)]-h(x)\}^2\\=\{y(x;D)-E_D[y(x;D)]\}^2+\{E_D[y(x;D)-h(x)]\}^2\\\\+2\{y(x;D)-E_D[y(x;D)]\}\{E_D(x;D)-h(x)\} \qquad{(3.39)}$$

- 첫번째 텀은 bias 
