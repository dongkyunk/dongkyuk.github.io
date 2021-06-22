---
title:  "Github 블로그 단어 잘림 해결 방법"
categories: tips
excerpt: "줄바뀜을 할 때 단어가 중간에 잘리는 현상 해결책"
---

Jekyll로 한글 블로그를 만들면 줄바뀜을 할 때 단어가 중간에 잘리는 현상을 발견 할 수 있다. 디자이너는 아니지만 이런 가독성이 떨어지는 요소를 볼 때마다 상당히 불편하다. 

해결책은 아래와 같다.

```_sass/minimal-mistakes/_page.scss``` 에서 다음 코드를 추가한다.

```scss
.page__content {
  text-align: left;
  word-break: keep-all;
}
```

꼭 minimal mistakes theme이 아니더라도 ```_page.scss```를 바꿔주면 된다는건 똑같은 것 같다.