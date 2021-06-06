---
title:  "Indoor Location & Navigation"
categories: kaggle
excerpt: 와이파이 신호로 실내 내비게이션
use_math: true
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

Wifi 신호 데이터의 경우 공유기의 BSSID와 신호 세기인 RSSI로 나뉜다.

또한, 현재 어떤 건물인지에 대한 site_id가 제공이 된다.

## Model

이제 위의 BSSID와 RSSI를 사용하여 현재 사용자의 건물에서의 x, y, floor 위치를 예측하는 Regression 모델을 설계해보자.

BSSID의 경우 공유기 고유의 값이자 잠재적으로 위치에 대한 정보를 가지고 있는 정보이니 Learnable Embedding으로 만든다.

site_id 또한 비슷하게 Learnable Embedding으로 만든다.

그후 이 feature들을 fully connected layer에 넣고 Stacked LSTM에 넣는다.

여기서 LSTM을 사용한 것은 이 [노트북](https://www.kaggle.com/kokitanisaka/lstm-by-keras-with-unified-wi-fi-feats)을 참고한 것인데, 

저자는 LSTM을 사용해서 각 신호 간의 상관관계를 address하고자 한 것 같은데, 현재 주어진 데이터는 Sequential Data도 아니라는 점을 생각하면 상당히 flawed된 approach이다.

이에 개선된 transformer 모델을 작업하다가 훈련소 입소를 하게 되었다 (ㅠㅠ). Transformer 모델의 경우 다음 게시물에 올려볼 예정이다.

```python
class SeqLSTM(nn.Module):
    def __init__(self, wifi_num, bssid_dim, site_id_dim, embedding_dim=64):
        """SeqLSTM Model
        Args:
            wifi_num (int): number of wifi signals to use
            bssid_dim (int): total number of unique bssids
            site_id_dim (int): total number of unique site ids
            embedding_dim (int): Dimension of bssid embedding. Defaults to 64.
        """
        super(SeqLSTM, self).__init__()
        self.wifi_num = wifi_num
        self.feature_dim = 256

        # Embedding
        self.embd_bssid = nn.Embedding(bssid_dim, embedding_dim)
        self.embd_site_id = nn.Embedding(site_id_dim, embedding_dim)

        # Linear
        self.fc_rssi = nn.Linear(1, embedding_dim)
        self.fc_features = nn.Linear(embedding_dim * 3, self.feature_dim)
        self.fc_output = nn.Linear(16, 3)

        # Other
        self.bn_rssi = nn.BatchNorm1d(embedding_dim)
        self.bn_features = nn.BatchNorm1d(self.feature_dim)
        self.dropout = nn.Dropout(0.3),

        self.lstm1 = nn.LSTM(input_size=256, hidden_size=128,
                             dropout=0.3, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=16,
                             dropout=0.1, bidirectional=False)

    def forward(self, x):
        embd_bssid = self.embd_bssid(x['BSSID_FEATS'])  # (,wifi_num,embedding_dim)

        embd_site_id = self.embd_site_id(x['site_id'])  # (,embedding_dim)
        embd_site_id = torch.unsqueeze(embd_site_id, dim=1)  # (,1,embedding_dim)
        embd_site_id = embd_site_id.repeat(
            1, self.wifi_num, 1)  # (,wifi_num,embedding_dim)

        rssi_feat = x['RSSI_FEATS']  # (,wifi_num)
        rssi_feat = torch.unsqueeze(rssi_feat, dim=-1)   # (,wifi_num,1)
        rssi_feat = self.fc_rssi(rssi_feat)              # (,wifi_num,embedding_dim)
        rssi_feat = self.bn_rssi(rssi_feat.transpose(1, 2)).transpose(1, 2)
        rssi_feat = torch.relu(rssi_feat)

        x = torch.cat([embd_bssid, embd_site_id, rssi_feat],
                      dim=-1)  # (,wifi_num,embedding_dim*3)

        x = self.fc_features(x)  # (,wifi_num, feature_dim)
        x = self.bn_features(x.transpose(1, 2)).transpose(1, 2)
        x = torch.relu(x)

        x = torch.transpose(x, 0, 1)  # (wifi_num,,128)
        x, _ = self.lstm1(x)

        x = x[-1] # (256,16)
        x = torch.relu(x)

        output = self.fc_output(x).squeeze()  # (,3)

        return output
```

Loss의 경우 xy에 대해서는 MSE Loss를 사용했고, floor prediction은 l2 loss를 사용했다.

Floor Prediction의 경우 정확도가 높은 모델들이 많아서 input data로 주고 xy만 predict하는 모델들이 많았는데, 내 모델의 경우 
floor 또한 예측하게 해서 일종의 auxiliary task로 사용했다.

Optimizer는 Adam을 쓰고, ```ReduceLROnPlateau```만 쓰고 lr이나 종류 실험을 많이 하지는 않았다.

```python
def xy_loss(xy_hat, xy_label):
    xy_loss = torch.mean(torch.sqrt(
        (xy_hat[:, 0]-xy_label[:, 0])**2 + (xy_hat[:, 1]-xy_label[:, 1])**2))
    return xy_loss


def floor_loss(floor_hat, floor_label):
    floor_loss = 15 * torch.mean(torch.abs(floor_hat-floor_label))
    return floor_loss


class IndoorLocModel(LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.lr = Config().lr

        self.critertion_xy = xy_loss
        self.criterion_floor = floor_loss

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y, f = batch['x'].unsqueeze(
            -1), batch['y'].unsqueeze(-1), batch['floor']

        xy_label = torch.cat([x, y], dim=-1)

        output = self(batch)
        xy_hat = output[:, 0:2]
        f_hat = output[:, 2]

        loss_xy = self.critertion_xy(xy_hat, xy_label)
        loss_floor = self.criterion_floor(f_hat, f)
        loss = loss_xy + loss_floor

        return {'loss': loss, 'loss_xy': loss_xy, 'loss_floor': loss_floor, 'xy_label': xy_label, 'xy_hat': xy_hat, 'floor_hat': f_hat, 'f': f}

    def training_epoch_end(self, outputs):
        loss_xy = torch.mean(torch.stack(
            [output['loss_xy'] for output in outputs], dim=0))
        loss_floor = torch.mean(torch.stack(
            [output['loss_floor'] for output in outputs], dim=0))
        loss = torch.mean(torch.stack([output['loss']
                          for output in outputs], dim=0))

    def validation_step(self, batch, batch_nb):
        x, y, f = batch['x'].unsqueeze(
            -1), batch['y'].unsqueeze(-1), batch['floor']

        xy_label = torch.cat([x, y], dim=-1)

        output = self(batch)
        xy_hat = output[:, 0:2]
        f_hat = output[:, 2]

        return {'xy_label': xy_label, 'xy_hat': xy_hat, 'f_hat': f_hat, 'f': f}

    def validation_epoch_end(self, outputs):
        xy_label = torch.cat([output['xy_label'] for output in outputs], dim=0)
        xy_hat = torch.cat([output['xy_hat'] for output in outputs], dim=0)
        f_hat = torch.cat([output['f_hat']
                           for output in outputs], dim=0)
        f_hat = torch.squeeze(f_hat)
        f = torch.cat([output['f'] for output in outputs], dim=0)

        loss_xy = self.critertion_xy(xy_hat, xy_label)
        loss_floor = self.criterion_floor(f_hat, f)
        loss = loss_xy + loss_floor

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_metric', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss', }
```

## Postprocessing

사실 이 대회는 model 보다는 postprocessing이 훨씬 중요한 대회였다. (1,2등의 solution만 봐도 모델보다는 discrete optimization와 postprocessing technique에 대한 이야기가 훨씬 많다)

Postprocessing의 경우 핵심적인 것은 [snap to grid](https://www.kaggle.com/robikscube/indoor-navigation-snap-to-grid-post-processing), [cost minimiztion](https://www.kaggle.com/saitodevel01/indoor-post-processing-by-cost-minimization)이 있었고, 주최 측이 data leakage 등을 활용한 방법들도 있었다. (훈련소에 있는 동안 새로 올라온 postprocessing technique을 적용을 못한 것이 수상을 놓친 이유였다)

Snap to grid의 경우 데이터의 위치가 정확한 데이터가 아닌 grid에 대응되는 위치라는 것을 이용해 prediction을 가까운 grid로 옮기는 방법이다. (생각해보면 조금 치사하다 ㅎㅎ)

Cost minimization의 경우 저자가 말한 것을 인용하자면 To combine machine learning (wifi features) predictions with sensor data (acceleration, attitude heading), I defined cost function as follows, 

![Image](https://raw.githubusercontent.com/dongkyuk/dongkyuk.github.io/main/assets/images/cost_min.png)

위 수식을 사용해 absolute postion loss를 minimize함과 동시에 sensor data로 predict한 relative position change loss를 minimize하도록 했다. 

## 마치며

실내 내비게이션은 과거에 [해커톤](https://dongkyuk.github.io/project/cmu_maps/)으로 했었던 주제인데, Kaggle 대회로 참여하니 아주 흥미로웠다. 

두 기술이 결합되면 실제로 실시간 실내 내비게이션 서비스를 만들 수 있지 않을까라는 생각이 들기도 한다.

다음 게시물에는 1,2등 솔루션을 살펴 볼 예정이다.