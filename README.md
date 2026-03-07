#  Stocker — 股票分析與預測工具

一款以 Python 為基礎、結合 [Facebook Prophet](https://facebook.github.io/prophet/) 時間序列預測模型的股票分析工具，支援股價建模、視覺化分析及交易策略評估。

---

##  主要功能

- **歷史股價視覺化** — 依自訂日期範圍繪製股價走勢圖，支援絕對價格與百分比變化兩種顯示模式
- **Prophet 預測建模** — 訓練可加性時間序列模型，支援季節性與趨勢轉折點靈敏度的彈性配置
- **未來股價預測** — 預測指定天數內的股價走勢，並附帶信賴區間與漲跌方向標示
- **模型效能評估** — 使用平均絕對誤差（MAE）、漲跌方向準確率及信賴區間覆蓋率等指標評估模型表現
- **交易策略模擬** — 比較 Prophet 預測策略與「買進持有」策略的獲利差異
- **趨勢轉折點分析** — 自動偵測股價結構性轉折點，並可與 Google 搜尋趨勢進行相關性分析
- **超參數驗證** — 評估不同 Changepoint Prior Scale（CPS）對模型訓練與測試誤差的影響

---

##  環境需求

透過 pip 安裝所有相依套件：

```bash
pip install -r requirements.txt
```

**相依套件說明：**

| 套件 | 用途 |
|------|------|
| `pandas` | 資料處理與操作 |
| `numpy` | 數值計算 |
| `prophet` | 時間序列預測模型 |
| `matplotlib` | 資料視覺化 |
| `pytrends` | Google 搜尋趨勢擷取 |

---

##  專案結構

```
Stocker-main/
├── stocker.py        # Stocker 核心類別，包含所有分析方法
├── price.csv         # 範例股價資料
├── requirements.txt  # Python 相依套件清單
└── README.md         # 專案說明文件
```

---

##  使用方式

### 初始化

```python
import pandas as pd
from stocker import Stocker

# 載入股價資料（需為 DatetimeIndex 格式的 Series）
price = pd.read_csv('price.csv', index_col=0, parse_dates=True).squeeze()

stock = Stocker(price)
```

### 繪製歷史股價圖

```python
stock.plot_stock(start_date='2022-01-01', end_date='2023-01-01')
```

### 預測未來股價

```python
stock.predict_future(days=30)
```

### 評估模型預測表現

```python
stock.evaluate_prediction(start_date='2022-01-01', end_date='2023-01-01')
```

### 模擬交易策略

```python
# 以 10 股為單位，比較 Prophet 策略與買進持有策略的獲利
stock.evaluate_prediction(start_date='2022-01-01', end_date='2023-01-01', nshares=10)
```

### 趨勢轉折點分析

```python
# 視覺化模型偵測到的股價轉折點
stock.changepoint_date_analysis()

# 結合 Google 搜尋趨勢進行分析
stock.changepoint_date_analysis(search='stock market crash')
```

### Changepoint Prior 超參數驗證

```python
stock.changepoint_prior_validation(changepoint_priors=[0.001, 0.05, 0.1, 0.2])
```

---

##  技術方法

Stocker 採用 **Facebook Prophet** 可加性迴歸模型，其數學表示式如下：

$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$

| 元件 | 說明 |
|------|------|
| $g(t)$ | 趨勢項（含轉折點的分段線性函數） |
| $s(t)$ | 季節性項（透過傅立葉級數模擬月度與年度週期） |
| $h(t)$ | 假日與特殊事件效應 |
| $\epsilon_t$ | 殘差雜訊 |

預設模型配置：

- **訓練資料年限：** 3 年
- **Changepoint Prior Scale：** 0.05
- **季節性設定：** 月度與年度季節性啟用；週度與每日季節性停用

---

##  主要參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `training_years` | `3` | 用於模型訓練的歷史資料年數 |
| `changepoint_prior_scale` | `0.05` | 趨勢彈性靈敏度（數值越大，模型對趨勢變化越敏感） |
| `monthly_seasonality` | `True` | 是否啟用月度季節性 |
| `yearly_seasonality` | `True` | 是否啟用年度季節性 |
| `round_dates` | `True` | 自動將不在資料集中的日期調整為最近有效日期 |

---

##  模型評估指標

- **平均絕對誤差（MAE）** — 預測價格與實際價格的平均絕對差距
- **漲跌方向準確率** — 模型正確預測股價漲跌方向的比例
- **信賴區間覆蓋率** — 實際股價落在預測信賴區間內的比例

---

##  Google Colab 完整執行範例

以下為在 Google Colab 上執行股價預測的完整程式碼：

```python
# 下載專案與安裝套件
!git clone https://github.com/easonlee0512/Stocker
!pip install -r Stocker/requirements.txt
!pip install --upgrade matplotlib

import warnings

# 忽略非必要的警告訊息
warnings.filterwarnings('ignore')

import pandas as pd

# 讀入股價 Series
df = pd.read_csv('Stocker/price.csv', index_col='date', parse_dates=['date'])
price = df.squeeze()
price.head()

# 初始化 Stocker 物件
from Stocker.stocker import Stocker
stock = Stocker(price)

# 建立 Prophet 模型並預測未來 90 天
model, model_data = stock.create_prophet_model(days=90)

# 比較不同 Changepoint Prior Scale 的影響
stock.changepoint_prior_analysis(changepoint_priors=[0.001, 0.05, 0.1, 0.2])

# 評估模型預測準確度（預設回測最近一年）
stock.evaluate_prediction()

# 預測未來 100 天的漲跌走勢
stock.predict_future(days=100)
```
---

##  授權

本專案為開源專案，授權細節請參閱原始儲存庫說明。
