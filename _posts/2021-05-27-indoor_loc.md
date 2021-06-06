---
title:  "Indoor Location & Navigation (작성 중)"
categories: kaggle
excerpt: 와이파이 신호로 실내 내비게이션
---
이번에 Kaggle의 Indoor Location & Navigation 대회에 참여했다.

Deadline 2주 정도를 남기고 육군훈련소에 입소하느라 안타깝게 입상은 실패했다. (훈련소만 없었어도 은상은 했을 듯...) 

그래도 입소하기 전에 억하심정으로 training code [notebook](https://www.kaggle.com/dongkyunkim/pytorch-lightning-lb-4-965-with-postprocessing)을 다 올려서 upvote를 꽤 받았는데, 그 내용을 여기에 공유하고 설명하고자 한다. 

다음 게시물에는 이번 대회의 상위권 입상자들의 전략과 코드를 분석해볼까 한다. 

## How
흔히들 navigation을 생각하면 GPS를 떠오르기 쉽다. 

하지만, GPS는 외부에서 해야 좋은 정확도를 보장하고, 실내에서 사용하기에는 정확도가 부족하다. 

그렇기에 이 대회는 스마트폰으로 모을 수 있는 데이터를 바탕으로 주어진 건물에서 자신의 위치(x, y 좌표와 층수)를 예측하고자 한다.

## Data
먼저 우리에게 주어진 데이터를 먼저 살펴보자.

이 데이터는 XYZ10이라는 회사가 Microsoft Research와 함께 모은 데이터이다. 대략 200개가 넘는 건물에서 30000개 정도의 위치에서의 기록된 데이터를 제공한다. 

제공되는 센서 데이터는 다음과 같다:
- IMU (accelerometer, gyroscope)
- geomagnetic field (magnetometer) readings
- WiFi
- Bluetooth 

전체 데이터에 대한 EDA는 이 [노트북](https://www.kaggle.com/andradaolteanu/indoor-navigation-complete-data-understanding)을 참고했다. 
원본 데이터에 대한 설명은 [여기서](https://github.com/location-competition/indoor-location-competition-20) 볼 수 있다. 

당장 여기서 우리가 가장 유용하게 사용할 데이터는 Wifi 신호 데이터이다.
