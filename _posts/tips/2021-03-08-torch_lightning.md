---
title:  "Pytorch Lightning 튜토리얼/TPU"
categories: tips
excerpt: Pytorch 코드를 간단하게!
---
Pytorch를 해본 사람이면 항상 코드에 반복되는 부분이 있음을 느낄거다. :thinking:	

이런 Boiler plate 코드가 없다면 중요한 연구에만 집중할 수 있을 것이고, 남의 코드를 읽기도 쉬워질 것이다. 

이에 대한 해결책으로 제시된 것이 Pytorch Lightning이다.

## How?

이렇게 모델을 정의하고
```python
class LitModel(LightningModule):

    def __init__(self):
        # Native PyTorch와 동일
    def forward(self, x):
        # Native PyTorch와 동일
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        return val_loss
```

데이터셋을 로딩해서
```python
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset)
```
학습한다.
```python
# init model
lit_model = LitModel()

# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
# trainer = pl.Trainer(gpus=8) (if you have GPUs)
trainer = pl.Trainer()
trainer.fit(lit_model, train_loader)
```

GPU/TPU 설정도 간단하다.
```
# train on 1 GPU
trainer = pl.Trainer(gpus=1)
```
```
# Train on TPUs
trainer = pl.Trainer(tpu_cores=8)
```
## More

Pytorch Lightning은 코드만 간단하게 해주는게 아니다.
각종 최신 및 편리한 기능들을 제공한다.

지원하는 기능들
- 16bit training
- Automatic early stopping
- Automatic Optimization
- 너무 많다! 

자세한건 [documentation](https://pytorch-lightning.readthedocs.io/en/latest/index.html)을 참고하자.


## Why

Pytorch Lightning은 여러모로 장점이 많은 라이브러리다. 

코드도 깔끔해지고 (특히 Tensorflow 2/Keras에서 Pytorch로 넘어오는 유저들은 익숙한 구성에 빠르게 적응할 것 같다), pytorch xla로 하면 골치 아프고 복잡한 TPU 호환도 깔끔하게 한줄로 해준다. 아직 다 파악하지도 못할만큼 각종 기능들 또한 제공해주는건 덤이다. 

나 또한 캐글 등 개인 딥러닝 프로젝트에서 애용 중이다 ([예시](https://github.com/dongkyuk/Kaggle_Vent_Pressure))

Pytorch Lightning으로 다들 번개 처럼 빠른 딥러닝 연구를 할 수 있길 바란다. :sunglasses:	


