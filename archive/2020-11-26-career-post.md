---
title: "앗! 내일이 기술 면접이라면?"
date: 2020-11-27 00:00:00 -0400
categories: career
---
### 0. About 

제가 기술 면접을 볼 때 필수적으로 알아야한다고 생각하는 것들을 위주로 계속 정리하는 글입니다. 

전날 가볍게 remind 하는 식으로 사용하면 좋을 것 같아 만들었습니다. (**간결함**이 생명!) 

많은 부분들이 [JaeYeopHan](https://github.com/JaeYeopHan)님의 [Interview_Question_for_Beginner](https://github.com/JaeYeopHan/Interview_Question_for_Beginner) 레포의 초간단 요약본입니다. 

```
P.S. 개발 분야별로 알아야 할 것들에 차이가 있어 기초적인 내용만 넣었습니다. 
저 같은 경우 ML쪽입니다. 이것 외에도 무조건 알면 좋을 것 같은 내용들이 있으면 알려주세요.
```

### 1. OS
#### 프로세스와 스레드의 차이
[자세한 내용](https://velog.io/@raejoonee/프로세스와-스레드의-차이)

요약 : 
> 프로세스: **운영체제**로부터 Code/Data/Stack/Heap 메모리 영역을 할당받은 **작업**의 단위.  
스레드: **프로세스**가 할당받은 자원을 이용하는 **실행 흐름**의 단위.

스레드는 **Code/Data/Heap 메모리 영역의 내용을 공유** -> 오류가 발생한다면 같은 프로세스 내의 다른 스레드 모두 강제 종료. 

> 프로세스는 **정보 공유** 시 IPC를 사용하는 등의 **번거로움**. 
스레드는 구조상 **쉬움**, 때문에 멀티태스킹보다 **멀티스레드가 자원을 아낌**

다만 스레드의 스케줄링은 운영체제가 처리하지 않기 때문에 프로그래머가 직접 **동기화 문제** 대응


#### 메모리 구조
[자세한 내용](https://velog.io/@hidaehyunlee/메모리-구조를-알아보자)

요약 :

> **코드** : 실행할 프로그램의 코드 / 프로그램 시작 부터 끝까지 메모리에 올라감. 
**데이터** : Global variable, Static variable, 문자열 상수 (char *s1 = "abcdefg"). 
**스택** : Local variable, Parameter / 함수의 호출이 완료되면 소멸, 컴파일 시 크기 결정. 
**힙** : 동적 할당과 해제 가능 (malloc, free) / 런타임에 크기 결정

#### 'Call by value' vs 'Call by reference'
https://codingplus.tistory.com/29
https://wayhome25.github.io/cs/2017/04/11/cs-13/
요약 :
>함수를 부를 때 변수를 복사를 하여 처리를 하느냐 (value) 아니면 직접 참조를 하느냐 (reference)

> 위에서도 나오듯 스택프레임의 Local Variable은 함수 호출 후 사라짐

### 2. 데이터 구조
https://github.com/JaeYeopHan/Interview_Question_for_Beginner/tree/master/DataStructure#array-vs-linked-list

요약 :

> **Array** : Index만 알면 O(1) random access 가능 / 연속성이 깨져서 추가 cost
**Linked List** : 각 노드가 다음 노드만 알고 있음 / Search, Insert, Delete 모두 O(n)

> **Stack** : Last In First Out (LIFO)
**Queue** : First In First Out (FIFO)

> **Tree** : 비선형 자료구조, Hierarchical Relationship, 그래프의 한 종류

> **Graph** : G = (V,E) / Vertex and Edges 

### 3. 알고리즘
https://github.com/JaeYeopHan/Interview_Question_for_Beginner/tree/master/Algorithm
알고리즘은 벼락치기가 힘들다 😱 (딴 분야도 그렇지만 ㅎㅎ)

접근법 요약 :

> Sorting (plus searching / binary search)
Divide and Conquer
Dynamic Programming / Memoization
Greediness
Recursion
Specific data structure (상위 10개만 알아도 충분)

>**자신있게 생각의 흐름을 말하자!! 💡💡💡**
 
### 4. Python
https://github.com/JaeYeopHan/Interview_Question_for_Beginner/tree/master/Python

Garbage Collection 관련 글 꽤 유용한듯

### 5. 참고 자료 및 보면 좋을 링크들

기초 : https://github.com/JaeYeopHan/Interview_Question_for_Beginner

https://wayhome25.github.io/blog/categories/#컴퓨터공학



세부 : https://github.com/MaximAbramchuck/awesome-interview-questions

데이터 사이언스 : https://github.com/zzsza/Datascience-Interview-Questions

### 마지막 업데이트: 2020/09/09 
