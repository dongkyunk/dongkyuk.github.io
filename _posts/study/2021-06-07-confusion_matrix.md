---
title:  "Confusion Matrix로 Classification Metric 용어들의 Confusion을 없애자"
categories: study
excerpt: Classification Metric들을 살펴보자
---
## 너무 많은 Metric들

머신러닝을 하다보면 Precision, Recall, Accuracy, F1 Score, F2 Score, AUROC 등의 metric을 접하는데,

이들의 정의와 효용성 매우 헷갈리기 마련이다. 

이번 시간에는 Confusion Matrix를 보면서 이 혼란스러움을 해결해보자 한다.

## Confusion Matrix, Accuracy

먼저 Confusion Matrix를 살펴보자.

![Image](https://shuzhanfan.github.io/assets/images/confusion_matrix.png) 

True/False의 경우 예측이 맞았는지 여부이고, Positive/Negative의 경우 예측한 클래스를 말한다.

여기서 우리가 많이 알고 있는 Accuracy의 정의가 무엇인지 생각해볼 수 있는데, 전체 예측 중 예측이 옳았던 비율이다.

$$\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}$$

## Type 1, 2 Error

![Image](https://shuzhanfan.github.io/assets/images/typeiandtypeiierror.jpg)

예측이 틀린 경우 (False일때)를 두가지 경우로 나눌 수 있고 이를 Type 1, 2 Error라고한다.

위 사진을 보면서 이해하면, 

- Type 1 Error의 경우 False Positive, 즉 Negative를 Positive라고 예측한 것이다. False Alarm이라 생각해도 무방하다.
- Type 2 Error의 경우 False Negative, 즉 Positive를 Negative라고 예측한 것이다. Underestimation이라 생각해도 무방하다.

## Precision, Recall, F1 Score

Precision의 경우 위의 Type 1 Error를 반영한 metric이다.

$$\text{Precision} = \frac{TP}{TP+FP}$$

Recall의 경우 위의 Type 2 Error를 반영한 metric이다.

$$\text{Recall} = \frac{TP}{TP+FN}$$

F1 Score의 경우 위의 Precision, Recall의 Harmonic Mean이다.

$$\text{F1 Score} = \frac{2}{\frac{1}{\text{Precision}}+\frac{1}{\text{Recall}}}$$

## Sensitivity, Specificity

여기서 Sensitivity, Specificity라는 개념이 등장하는데, 각 클래스 별로 얼마나 잘 classify라고 생각하면 좋을 것 같다.

Sensitivity의 경우 Positive Class를 얼마나 잘 classify 했는지 보여준다.

$$\text{Sensitivity} = \frac{TP}{TP+FN}$$

Specificity의 경우 Negative Class를 얼마나 잘 classify 했는지 보여준다.

$$\text{Specificity} = \frac{TN}{TN+FP}$$

## ROC Curve와 AUC

ROC Curve를 살펴보기 전에, 위에 말한 Sensitivity와 Specificity간에 tradeoff가 있다는 점을 인지해야한다.

![Image](https://machinelearningmastery.com/wp-content/uploads/2014/11/ROC1.png)

위의 이미지에서 threshold가 올라가서 strict하게 positive class를 classify할 경우, $TP,FP$는 줄고, $TN,FN$는 늘어난다.

그렇기에, Sensitivity는 줄고, Specificity는 늘어난다. 비슷하게 threshold가 내려가면 Sensitivity는 늘고, Specificity는 준다.

이 tradeoff를 그림으로 표현하면 아래와 같다.

![Image](https://miro.medium.com/max/650/1*ceB9hobuBUjnPpRKedA-VA.png)

ROC (Receiver Operating Characteristic) Curve는 위 그림에서 x축을 Specificity가 아닌 1-Specificity로 한 것이다.

![Image](https://miro.medium.com/max/652/1*QqZzGJwzYxnHWZ_axq6ynA.png)
