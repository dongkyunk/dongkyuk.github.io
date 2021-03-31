---
title:  "Dacon 제 2회 컴퓨터 비전 학습 경진대회"
categories: kaggle
excerpt: Dacon 대회로 시작하는 Image classification baseline
---
캐글은 아니지만, 매우 유사한 한국의 Dacon에서 주최한 컴퓨터 비전 학습 경진대회를 얼마 전에 참가를 했다.  Private LB 0.88446 Accuracy를 달성해 🥇 4등 및 상금 수상권에 들었다. 상금 수령을 위해 도커 제출을 해야했는데 수령액에 비해 절차가 많아서 넘겼다 ㅎㅎ  

## What
10 ~ 14개의 글자(알파벳 a – Z, 중복 제외)가 무작위로 배치되어 있는 (256, 256) 크기의 이미지 속에 들어있는 알파벳과 들어있지 않은 알파벳을 분류하는 multi-label classification 문제다.

![Image](https://dl.airtable.com/.attachmentThumbnails/069cd9c89e116be040488d3c3e6d4058/457224f8)

## How
1. Efficientnet B5 (Adam, Cross entropy Loss)
Backbone은 Efficientnet B5를 썼다.

```python
import torch
import torch.nn as nn
import numpy as np
import torch_optimizer as optim
from efficientnet_pytorch import EfficientNet
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import Accuracy

class MnistModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b5')
        self.classifier = nn.Linear(1000, 26)

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.classifier(x)
        return x
```

Optimizer는 Adam, Loss는 MultilabelSoftMarginLoss (Sigmoid 씌운 CrossEntropy)를 썼다. RAdam, SGD, LookAheadWrapper, CosineAnnealingLR 등을 구현은 했지만 실험을 많이 해보진 못하고 제출을 했다.

```python
class MnistModel(LightningModule):
    def __init__(self, model: nn.Module = None):
        super().__init__()
        self.model = model
        self.criterion = nn.MultiLabelSoftMarginLoss()
        self.metric = Accuracy(threshold=0.5)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        acc = self.metric(y_hat, y)

        neptune.log_metric('train_loss', loss)
        neptune.log_metric('train_accuracy', acc)

        return loss     

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        acc = self.metric(y_hat, y)

        return {'loss': loss, 'acc': acc}


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_acc', avg_acc, prog_bar=True)

        neptune.log_metric('val_loss', avg_loss)
        neptune.log_metric('val_acc', avg_acc)   


    def configure_optimizers(self):
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
#       optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
#       optimizer = optim.RAdam(
#             model.parameters(),
#             lr= 1e-3,
#             betas=(0.9, 0.999),
#             eps=1e-8,
#             weight_decay=0,
#         )
#       optimizer = optim.Lookahead(optimizer, k=5, alpha=0.5)
#       scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=10, eta_min=0)
        return [optimizer]
```
2. Rotation Augmentation (Traing, Test time)

Augemntation은 rotation만 적용했다. Flip의 경우 성능이 오히려 하락을 했다. 유력한 가설은 p, d 등의 대칭적인 알파벳을 flip 할 경우 학습에 혼선을 주는 것 같았다. 

```python
import albumentations as A

transforms_train = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Normalize(),
    A.pytorch.transforms.ToTensor()
])

transforms_test = A.Compose([
    A.Normalize(),
    A.pytorch.transforms.ToTensor()
])
```

Test time augmentation도 성능 향상에 소폭 기여했다.

```python
import torch 
import numpy as np
from edafa import ClassPredictor

class TTA(ClassPredictor):
    def __init__(self,model,device,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = model
        self.device = device

    def predict_patches(self,patches):
        patches = torch.from_numpy(patches)
        patches = patches.permute(0, 3, 1, 2).to(self.device)

        res = self.model(patches)
        res = torch.sigmoid(res)
        res = res.cpu().detach().numpy()
        return res

# Test time augmentation
conf = '{"augs":["NO",\
                "ROT90",\
                "ROT180",\
                "ROT270"],\
        "mean":"ARITH"}'
model = TTA(model, device, conf)

```
 
3. 5 fold Ensemble
```python
def split_dataset(path: os.PathLike, num_split:int=5) -> None:
    df = pd.read_csv(path)
    kfold = KFold(n_splits=num_split)
    for fold, (train, valid) in enumerate(kfold.split(df, df.index)):
        df.loc[valid, 'kfold'] = int(fold)

    df.to_csv('data/split_kfold.csv', index=False)
```

4. Test time Probability Threshold

이게 중요한 부분이었는데, probability threshold를 통상적인 0.5가 아닌 0.6 정도로 했을 떄 성능 향상이 있었다. Model이 calibrate가 완벽히 되지 않았고, multi-label이라 그렇다는 것이 유력한 가설이다.

```python
outputs = model.predict_images(images)
outputs = torch.sigmoid(outputs) 
outputs = outputs > 0.6 
```

## 시간이 있었다면
시간이 더 있었다면
- Better Backbone (Efficientnet B7이라던지)
- Hyperparameter tuning
- Calibration (Temperature 등)
- Attention 추가 (?)
  
정도를 할 예정이었다.

## 느낀 점
1~7등까지 전부 우리보다 좋은 Backbone들을 썼음에도 이런 성과를 달성한게 꽤 놀라웠다. 시간과 리소스가 부족해서 실험을 많이 못했는데, probability threshold를 바꾼게 영향이 컸던 것 같다. Deep Learning을 하면서 model calibration에 대한 부분을 깊게 생각해본적이 없는데 상당히 흥미로운 주제인 것 같다. 

## Code
Reproduction을 염두에 두고 올린게 아니라 양해를 바란다.
<https://github.com/dongkyuk/Dacon-multi-mnist>