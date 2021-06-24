---
title:  "Meta learning, Transfer learning, Few shot learning, Finetuning 등 용어 정리"
categories: study
excerpt: 혼란스러운 learning 종류들의 정의를 확인해보자
---
## 너무 많은 용어들

딥러닝을 하다보면 Meta learning, Transfer learning, Few shot learning, Finetuning 등의 용어를 접하는데,

이들의 상관관계와 hierarchy가 매우 헷갈리기 마련이다. 학계 및 업계에서 너무 무분별하게 용어들을 사용하기 때문이 아닌가 싶다.

이번 시간에는 이 혼란스러움을 해결해보자 한다.

## 한줄 정리

- Meta Learning

Learning to learn. 즉, 기존 training time에 접해보지 않은 문제들을 빠르고/능숙하게 학습하도록 학습하는 것.
- Transfer Learning

특정 태스크 (Upstream task)를 학습한 모델의 일부를 다른 태스크 (Downstream task) 수행에 전이하여 재사용하는 기법.
- Few shot Learning

Transfer Learning의 일종으로, Downstream task에서 데이터를 few samples(몇개)만 사용하는 것
- One shot Learning

Transfer Learning의 일종으로, Downstream task에서 데이터를 한개만 사용하는 것
- Zero shot Learning

Transfer Learning의 일종으로, Downstream task의 데이터를 사용하지 않고 수행하는 것
- Finetuning

기존의 모델을 새로운 데이터에 low learning rate으로 재학습 시키는 것 (freeze 여부는 상황에 따라 다름)
