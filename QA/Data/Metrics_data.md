# BÁO CÁO ĐÁNH GIÁ CHẤT LƯỢNG DỮ LIỆU

## Dự án: Dự đoán xu hướng và giá close VN30F1M

> **Phạm vi báo cáo:** Giai đoạn 1 (Data Quality) – các metrics đánh giá chất lượng dữ liệu trước khi đưa vào training.
>
> **Mô hình dự kiến:** Classical ML (Logistic Regression, XGBoost, Random Forest) / LSTM / Transformer
>
> **Tasks:** Trend prediction (classification) và Close price prediction (regression)
>
> **Trading frequency:** Daily (2017-08-10 → 2026-05-12, 2.169 rows) và Hourly (2023-09-11 → 2026-05-12, 2.953 rows)
>
> **Nguồn dữ liệu:** Thư viện `vnstock` qua API KBS
>
> **Raw features:** `time`, `open`, `high`, `low`, `y_f1m` (close), `volume`, `vn30_index`

---

## MỤC LỤC

1. [Nguyên tắc tiền xử lý & Derived Features](#1-nguyên-tắc-tiền-xử-lý)
2. [Missing Value Ratio](#2-missing-value-ratio)
3. [Stationarity Tests (ADF + KPSS)](#3-stationarity-tests-kiểm-định-tính-dừng)
4. [Outlier Detection (IQR)](#4-outlier-detection)
5. [Pearson Correlation & VIF](#5-pearson-correlation--vif-multicollinearity)
6. [ACF / PACF Analysis](#6-acf--pacf-analysis)
7. [Information Coefficient (IC)](#7-information-coefficient-ic)
8. [Look-Ahead Bias Detection](#8-look-ahead-bias-detection)
9. [Class Balance](#9-class-balance-sau-khi-tạo-labels)
10. [Volatility Regime Analysis](#10-volatility-regime-analysis)
11. [Lưu ý đặc thù cho dữ liệu vnstock](#11-lưu-ý-đặc-thù-cho-dữ-liệu-vnstock--kbs)
12. [Tổng kết workflow đề xuất](#12-tổng-kết-workflow-đề-xuất)
13. [Ngưỡng quyết định nhanh](#13-ngưỡng-quyết-định-nhanh)

---

## 1. NGUYÊN TẮC TIỀN XỬ LÝ

### 1.1. Bắt buộc transform về Percentage Change

**Lý do:** Dữ liệu giá VN30F1M (2017–2026) có xu hướng tăng từ ~580 → ~2090 điểm (min thực tế: 578.7 điểm tháng 3/2020 COVID crash, giá khởi đầu 2017: ~746 điểm, max hiện tại: ~2089 điểm). Đây là chuỗi **non-stationary** với mean và variance thay đổi theo thời gian.

**Hệ quả nếu không transform:**
- Train data (2017–2024) có range giá khác hoàn toàn test data (2025–2026)
- Model học theo "thời gian" thay vì học pattern thực
- Mọi predictions trên test set sẽ bị skew
- Correlations giữa OHLC ≈ 1.00 (vô nghĩa)
- Boxplot/IQR false-flag toàn bộ giá gần đây là outliers

**Lựa chọn transform cho dự án này:**

| Phương pháp | Khi nào dùng |
|---|---|
| **Simple pct_change** (`x.pct_change()`) | Khi return < 5% (daily VN30F1M) → **khuyến nghị cho dự án này** |
| **Log return** (`np.log(x/x.shift(1))`) | Khi cần cộng dồn returns hoặc analyze dài hạn |

Vì daily returns VN30F1M thường < 3% và khoảng cách giá trị không quá lớn, **simple pct_change** là lựa chọn phù hợp.

### 1.2. Quy tắc áp dụng theo loại feature

| Loại feature | Cần transform? | Cách làm |
|---|---|---|
| OHLC raw prices (`open`, `high`, `low`, `y_f1m`) | ✅ Bắt buộc | `pct_change()` |
| Volume | ✅ Bắt buộc | `pct_change()` hoặc volume ratio |
| `vn30_index` | ✅ Bắt buộc | `pct_change()` |
| `basis_pct` = `(y_f1m - vn30_index) / vn30_index` | ❌ Không | Đã là ratio |
| `body_pct` = `(y_f1m - open) / open` | ❌ Không | Đã là ratio |
| `daily_range` = `(high - low) / open` | ❌ Không | Đã là ratio |
| RSI, Stochastic | ❌ Không | Đã bounded [0,100] |
| ATR | ⚠️ Cần | Chia cho price: `atr/close` |

### 1.3. Derived Features từ Raw Data

Raw dataset chỉ có 7 cột: `time`, `open`, `high`, `low`, `y_f1m`, `volume`, `vn30_index`. Tất cả derived features đều được tạo ra từ 6 cột giá/khối lượng này. Phân thành 4 nhóm:

---

#### Nhóm 1: Mandatory Transforms — pct_change() trên raw prices

Các cột raw price/volume phải được chuyển sang percentage change để đạt stationarity. **Không dùng raw values làm input cho model.**

| Derived Feature | Công thức | Từ cột nào | Ý nghĩa |
|---|---|---|---|
| `return` | `y_f1m.pct_change()` | `y_f1m` | Close-to-close return — **biến target chính** cho cả regression lẫn tạo label classification |
| `open_return` | `open.pct_change()` | `open` | Gap-up/gap-down so với phiên trước |
| `high_return` | `high.pct_change()` | `high` | Sức mạnh bull của phiên |
| `low_return` | `low.pct_change()` | `low` | Sức mạnh bear của phiên |
| `volume_change` | `volume.pct_change()` | `volume` | Thay đổi thanh khoản — IC = -0.012 (Poor) |
| `vn30_return` | `vn30_index.pct_change()` | `vn30_index` | Return của VN30 cash index |

> **Lưu ý:** `open_return`, `high_return`, `low_return` thường có VIF cao khi dùng cùng nhau do cùng phản ánh price movement. Kiểm tra VIF trước khi giữ lại tất cả.

---

#### Nhóm 2: Ratio / Candlestick Features — đã là tỷ lệ, không cần transform thêm

Ba features này là tỷ lệ tính trong **cùng một phiên**, nên đã scale-invariant và stationary theo định nghĩa. Không cần pct_change.

**`body_pct`** — Sức mạnh của phiên giao dịch (candlestick body)

```
body_pct = (y_f1m - open) / open
```

| Giá trị | Ý nghĩa thị trường |
|---|---|
| > 0 (dương, body xanh) | Phiên tăng — phe bull thắng |
| < 0 (âm, body đỏ) | Phiên giảm — phe bear thắng |
| ≈ 0 | Phiên giằng co, do dự |
| Body to (> 1%) | Momentum rõ ràng, conviction cao |
| Body nhỏ (< 0.3%) | Thị trường do dự, thường xuất hiện ở đỉnh/đáy |

**Thực tế trên daily VN30F1M (2017–2026):** min = -6.22%, max = +10.57%, mean ≈ 0.02%. IC = -0.067 (Good) — có predictive power thực.

**`basis_pct`** — Chênh lệch giá futures vs VN30 cash index (basis spread)

```
basis_pct = (y_f1m - vn30_index) / vn30_index
```

| Giá trị | Ý nghĩa thị trường |
|---|---|
| > 0 (contango) | Futures premium — thị trường kỳ vọng tăng, hoặc có arbitrage gap |
| < 0 (backwardation) | Futures discount — áp lực bán futures, hoặc kỳ vọng giảm |
| Basis bất thường lớn | Tín hiệu arbitrage sẽ kéo giá về cân bằng |

**Tại sao là feature mạnh nhất:** Khi basis lớn bất thường, có lực kéo giá futures về sát VN30 index (arbitrage pressure). Feature này capture được **mean-reversion signal** tự nhiên của thị trường phái sinh. IC = -0.121 (Excellent) — feature mạnh nhất trong dataset.

**Thực tế trên daily VN30F1M (2017–2026):** min = -6.17%, max = +5.94%, mean ≈ -0.21%. Mean âm cho thấy VN30F1M thường xuyên giao dịch nhẹ dưới VN30 index (slight backwardation).

**`daily_range`** (hay `range_pct`) — Biên độ nội phiên

```
daily_range = (high - low) / open
```

| Giá trị | Ý nghĩa thị trường |
|---|---|
| Cao (> 2%) | Phiên biến động mạnh — volatility cao, không chắc chắn |
| Thấp (< 0.5%) | Phiên sideway — thanh khoản thấp hoặc chờ đợi tin tức |

**Tác dụng:** Proxy cho intraday volatility. Kết hợp với `body_pct` để phân biệt:
- Range to + body to → trend mạnh với conviction
- Range to + body nhỏ → volatility cao nhưng không có direction rõ ràng (spinning top)
- Range nhỏ + body nhỏ → doji, thị trường hoàn toàn do dự

**Thực tế trên daily VN30F1M (2017–2026):** min = 0.15%, max = 11.36%, mean ≈ 1.61%. IC = +0.029 (Moderate).

---

#### Nhóm 3: Lag Features — từ kết quả PACF analysis

Tạo dựa trên significant lags xác định từ PACF. Chỉ giữ các lag có |IC| ≥ 0.02.

```python
df['return_lag_1'] = df['return'].shift(1)   # IC = +0.051 (Good)
df['return_lag_2'] = df['return'].shift(2)   # IC < 0.01 (Poor → drop)
df['return_lag_3'] = df['return'].shift(3)   # IC < 0.01 (Poor → drop)
df['return_lag_5'] = df['return'].shift(5)   # IC < 0.01 (Poor → drop)
```

> **Quan trọng:** `return_lag_1` là lag duy nhất có IC đáng kể cho daily data. Significant PACF lags [12, 14, 30] phản ánh seasonality tháng nhưng cần rolling IC test để xác nhận tính ổn định.

---

#### Nhóm 4: Rolling Features — volatility và momentum

```python
# Rolling volatility (annualized)
df['return_std_20'] = df['return'].rolling(20).std()  # IC = +0.042 (Moderate)

# Volatility regime features (dùng trong Volatility Regime Analysis)
df['vol_20d'] = df['return'].rolling(20).std() * np.sqrt(252)
df['volatility_lag_1'] = df['return_std_20'].shift(1)
df['volatility_ma_20'] = df['return_std_20'].rolling(20).mean()
```

> **Lưu ý look-ahead:** Rolling features **phải shift(1)** trước khi dùng làm input, vì `rolling(20).std()` tại ngày t bao gồm return của ngày t. Xem Section 8 để biết cách audit.

---

#### Tổng hợp: Bảng tất cả derived features

| Feature | Nhóm | Cần transform thêm? | IC (daily) | Giữ lại? |
|---|---|---|---|---|
| `return` | 1 - pct_change | ❌ | -0.046 | ✅ Target/Feature |
| `open_return` | 1 - pct_change | ❌ | — | ⚠️ Check VIF |
| `high_return` | 1 - pct_change | ❌ | — | ⚠️ Check VIF |
| `low_return` | 1 - pct_change | ❌ | — | ⚠️ Check VIF |
| `volume_change` | 1 - pct_change | ❌ | -0.012 | ❌ Poor IC |
| `vn30_return` | 1 - pct_change | ❌ | — | ⚠️ Check VIF vs return |
| `basis_pct` | 2 - ratio | ❌ | -0.121 | ✅ Excellent |
| `body_pct` | 2 - ratio | ❌ | -0.067 | ✅ Good |
| `daily_range` | 2 - ratio | ❌ | +0.029 | ✅ Moderate |
| `return_lag_1` | 3 - lag | ❌ | +0.051 | ✅ Good |
| `return_lag_2,3,5` | 3 - lag | ❌ | < 0.01 | ❌ Drop |
| `return_std_20` | 4 - rolling | ❌ (nhưng cần shift) | +0.042 | ✅ Moderate |

---

### 1.4. Lưu ý đặc biệt cho 2 tasks

- **Trend prediction (classification):** Label dựa trên `pct_change` của future close
- **Price prediction (regression):** Target có thể là `pct_change` (khuyến nghị) hoặc `log_return` để đảm bảo stationarity, KHÔNG dùng raw price làm target

---

## 2. MISSING VALUE RATIO

### 2.1. Định nghĩa và công thức

```
Missing Ratio = (Số giá trị thiếu / Tổng số giá trị) × 100%
Completeness  = 100% - Missing Ratio
```

### 2.2. Tại sao cần kiểm tra?

- Phát hiện vấn đề thu thập dữ liệu từ vnstock/KBS API
- Định hướng chiến lược imputation phù hợp
- Đảm bảo tính liên tục cho các technical indicators (MA, momentum)
- Phát hiện các ngày nghỉ giao dịch bất thường

### 2.3. Ngưỡng đánh giá

| Completeness | Mức độ | Hành động |
|---|---|---|
| 100% | Hoàn hảo | Không cần xử lý |
| 95–99% | Tốt | Imputation đơn giản (forward-fill) |
| 90–95% | Chấp nhận được | Cần điều tra nguyên nhân |
| < 90% | Cảnh báo | Phải tìm nguồn data tốt hơn |

### 2.4. Hậu quả nếu bỏ qua

- Model bị **NaN errors** khi training
- Forward-fill mù quáng làm sai lệch volatility trong thị trường futures
- Che giấu các **structural breaks** quan trọng

### 2.5. Áp dụng cho dataset VN30F1M

**Ghi chú:** Dataset hiện tại không có NaN. Tuy nhiên vẫn cần kiểm tra mở rộng:

- **Trading Day Coverage**: số ngày giao dịch thực tế / số ngày trong lịch HOSE (loại trừ weekend, holiday Việt Nam như Tết, 30/4, 2/9)
- **Bar Completeness** (cho hourly): số bars/ngày so với kỳ vọng (4–5 bars cho VN30F1M)

**Kết quả thực tế trên hourly dataset (2023-09-11 → 2026-05-12, 659 ngày giao dịch):**

| Số bars/ngày | Số ngày | Ghi chú |
|---|---|---|
| 2 bars | 1 ngày | 2026-05-12 — ngày đang cập nhật (incomplete, là ngày cuối dataset) |
| 4 bars | 339 ngày | Phiên giao dịch không có session 14:00 hoặc schedule cũ |
| 5 bars | 319 ngày | Đủ bars (10:00, 11:00, 13:00, 14:00, 15:00 hoặc 14:45) |

→ **1 ngày 2 bars là 2026-05-12** (ngày data pull cuối, chưa đủ phiên) — không phải data error.
→ Sự chênh lệch 4-bar vs 5-bar cần verify với lịch giao dịch HOSE: VN30F1M đổi trading hours (thêm session 14:30–14:45) từ một thời điểm nhất định.
→ Nếu không xử lý, **treat hourly như continuous series sẽ sai** — lunch break (11:30→13:00) và overnight gap (14:45→9:00) tạo ra artificial lags lớn trong ACF/PACF.

---

## 3. STATIONARITY TESTS (KIỂM ĐỊNH TÍNH DỪNG)

### 3.1. Tại sao bắt buộc cho dự án này?

Hầu hết thuật toán ML (kể cả **LSTM/Transformer** mà nhóm sử dụng) giả định data có **tính dừng**:
- Mean ổn định theo thời gian
- Variance ổn định
- Autocovariance không đổi theo time shift

Dữ liệu giá VN30F1M có trend rõ ràng → **non-stationary** → **bắt buộc** transform về returns trước khi test.

**Lưu ý đặc biệt cho LSTM/Transformer:** Mặc dù 2 model này có thể "học" một phần non-stationarity, nhưng performance sẽ tốt hơn nhiều khi input đã stationary.

### 3.2. ADF Test (Augmented Dickey-Fuller)

**Giả thuyết:**
- H0: Chuỗi có unit root (**không dừng**)
- H1: Chuỗi không có unit root (**dừng**)

**Ngưỡng diễn giải:**

| p-value ADF | Kết luận |
|---|---|
| < 0.01 | Chuỗi DỪNG với độ tin cậy 99% |
| 0.01 – 0.05 | Chuỗi DỪNG với độ tin cậy 95% (chấp nhận được) |
| 0.05 – 0.10 | KHÔNG kết luận được, cần test thêm |
| > 0.10 | Chuỗi KHÔNG DỪNG |

### 3.3. KPSS Test

**Giả thuyết (ngược ADF):**
- H0: Chuỗi **dừng**
- H1: Chuỗi **không dừng**

**Ngưỡng diễn giải:**

| p-value KPSS | Kết luận |
|---|---|
| > 0.10 | Chuỗi DỪNG (mạnh) |
| 0.05 – 0.10 | Chuỗi DỪNG (yếu) |
| 0.01 – 0.05 | Không dừng (yếu) |
| < 0.01 | Không dừng (mạnh) |

### 3.4. Kết hợp ADF + KPSS (Framework 4 trường hợp)

| ADF p-value | KPSS p-value | Kết luận | Xử lý |
|---|---|---|---|
| < 0.05 | > 0.05 | ✅ **CHẮC CHẮN DỪNG** | Đưa vào model |
| > 0.05 | < 0.05 | ❌ **CHẮC CHẮN KHÔNG DỪNG** | Phải differencing |
| < 0.05 | < 0.05 | ⚠️ Trend-stationary | Detrend trước |
| > 0.05 | > 0.05 | ⚠️ Difference-stationary | Differencing |

### 3.5. Hậu quả nếu không kiểm tra

- Model học spurious patterns (tương quan giả)
- LSTM/Transformer học theo "thời gian" thay vì pattern
- Coefficients trong logistic regression không reliable
- Out-of-sample performance sụp đổ
- Backtest có vẻ tốt nhưng live trading thất bại

### 3.6. Áp dụng cho VN30F1M

- Test trên **returns** (`pct_change`), KHÔNG test trên raw prices
- Test cả **squared returns** để check volatility clustering (GARCH effects)
- Với 2,169 mẫu daily (2017-08-10 → 2026-05-12) và 2,953 mẫu hourly (2023-09-11 → 2026-05-12) → đủ statistical power cho cả ADF và KPSS
- Vì có **2 timeframes** (daily và hourly), cần test riêng cho từng timeframe
- **⚠️ Lưu ý:** Hourly data chỉ bắt đầu từ 2023-09-11 (không có hourly data cho giai đoạn 2017–2023). Kết quả ADF/KPSS của hourly **không đại diện** cho giai đoạn COVID 2020 hay bear market 2022 — chỉ phản ánh bull market 2023–2026.

---

## 4. OUTLIER DETECTION

### 4.1. Phương pháp IQR (Interquartile Range)

**Công thức:**
```
Q1 = quantile(25%)
Q3 = quantile(75%)
IQR = Q3 - Q1
Lower bound = Q1 - k × IQR
Upper bound = Q3 + k × IQR
```

**Ngưỡng đánh giá theo k-multiplier:**

| Multiplier | Mức độ | Đặc điểm | Phù hợp khi nào |
|---|---|---|---|
| 1.5 × IQR | Standard | Catch ~99.3% normal points (giả định normal) | Data gần normal distribution |
| 2.0 × IQR | Lenient | Bỏ qua nhiều outliers hơn | Thị trường có volatility vừa |
| 3.0 × IQR | Extreme only | Chỉ catch true anomalies | Financial returns — fat tails |

### 4.1b. Tại sao k = 3.0 là lựa chọn đúng cho VN30F1M?

**Vấn đề với k nhỏ hơn:**

- **k = 1.5** giả định data xấp xỉ normal. Returns tài chính có kurtosis cao (fat tails), nên k=1.5 sẽ flag quá nhiều điểm hợp lệ là outlier. Với VN30F1M daily returns (std ≈ 1%), k=1.5 sẽ flag những phiên tăng/giảm ~2-3% — đây là biến động bình thường, không phải anomaly.
- **k = 2.0** vẫn có thể flag các market events hợp lệ như COVID recovery rallies (~3-4%/ngày) hay các phiên có tin tức lớn.

**Tại sao k = 3.0 phù hợp:**

Với VN30F1M daily:
- IQR của returns ≈ 1.3% (Q3 ≈ +0.8%, Q1 ≈ -0.5%)
- Ngưỡng k=3.0: lower ≈ -4.4%, upper ≈ +4.4%
- Chỉ những phiên vượt ±4-5% mới bị flag — tương đương mức độ COVID crash hoặc circuit breaker

Nếu một return vượt ngưỡng k=3.0, có 2 khả năng: (1) data error (API lỗi, tick sai), hoặc (2) market event cực đoan thật. Sau đó dùng bảng **4.5** để phân biệt.

**Rủi ro chọn k quá lớn (k > 4.0):** Bỏ sót cả data errors thật, model học noise.
**Rủi ro chọn k quá nhỏ (k < 2.0):** Flag và xóa quá nhiều market events hợp lệ → mất signal, dataset bị distort.

**Khuyến nghị cho VN30F1M:** Dùng **3.0 × IQR trên returns** (không phải prices) để chỉ catch true anomalies.

### 4.2. Tại sao IQR phù hợp hơn Z-Score?

- Returns tài chính có **fat tails** (kurtosis cao)
- Z-Score giả định normal distribution → over-flag outliers
- IQR robust với outliers (dùng median-based quartiles)

### 4.3. ⚠️ CẢNH BÁO QUAN TRỌNG

**KHÔNG áp dụng IQR trực tiếp lên raw prices.** Như đã phân tích trên dataset, boxplot trên `y_f1m` flag toàn bộ giá 2025-2026 là outliers vì price có trend tăng dài hạn. Đây là **misinterpretation** chứ không phải outliers thật.

### 4.4. Quy trình đúng

```python
# SAI - sẽ flag tất cả giá gần đây
outliers = IQR_method(df['y_f1m'])

# ĐÚNG - flag những price jumps bất thường
df['return'] = df['y_f1m'].pct_change()
outliers = IQR_method(df['return'], multiplier=3.0)
```

### 4.5. Phân biệt Data Error vs Market Event

| Đặc điểm | Data Error | Market Event |
|---|---|---|
| Volume | Thấp/bình thường | Cao bất thường (>3× avg) |
| Recovery | Không recover | V-shape recovery |
| Cross-asset | Chỉ ảnh hưởng 1 asset | Nhiều assets cùng affected |
| News | Không có | Có catalyst (Fed, COVID...) |

### 4.6. Hậu quả nếu bỏ qua

- Model học noise thay vì pattern
- Features bị skew do extreme values
- Backtest results bị bóp méo bởi vài data points

---

## 5. PEARSON CORRELATION & VIF (MULTICOLLINEARITY)

### 5.1. Pearson Correlation (Heatmap)

**Công thức:**
```
r(X,Y) = Cov(X,Y) / (σ_X × σ_Y)
```

**Ngưỡng đánh giá pairwise:**

| \|r\| | Mức độ | Hành động |
|---|---|---|
| > 0.9 | Cực mạnh | Hai biến gần như giống nhau, **loại 1** |
| 0.7 – 0.9 | Mạnh | Cảnh báo, cân nhắc loại bỏ |
| 0.3 – 0.7 | Trung bình | Bình thường, giữ lại |
| < 0.3 | Yếu | Hai biến độc lập, giữ lại |

**Khuyến nghị:** Vẽ **heatmap** để có cái nhìn tổng quan trước khi đi sâu vào VIF.

### 5.2. Variance Inflation Factor (VIF)

**Công thức:**
```
VIF_j = 1 / (1 - R²_j)
```
Trong đó `R²_j` là R-squared khi regress biến j lên TẤT CẢ các biến còn lại.

**Ngưỡng đánh giá multivariate:**

| VIF | Mức độ | Hành động |
|---|---|---|
| 1 – 2 | Lý tưởng | Không có vấn đề |
| 2 – 5 | Chấp nhận được | Bình thường |
| 5 – 10 | Cảnh báo | Cần điều tra |
| > 10 | Nghiêm trọng | Phải loại bỏ |

### 5.3. Khác biệt giữa Pearson và VIF

| Aspect | Pearson Correlation | VIF |
|---|---|---|
| Scope | 2 biến (pairwise) | Đa biến (multivariate) |
| Phát hiện | Quan hệ trực tiếp | Quan hệ tổng hợp |
| Bỏ sót | Quan hệ qua biến trung gian | Không bỏ sót |
| Có thể "overstate" | Có | Không |

**Ví dụ thực tế từ dataset:**
- `high_return ↔ low_return` correlation = 0.65 (nghe có vẻ cao)
- Nhưng VIF của `high_return` chỉ ~3.5 (chấp nhận được)
- → **VIF là chỉ số quyết định cuối cùng**

### 5.4. Quy tắc áp dụng theo model (nhóm sử dụng cả 3 loại)

| Model | Sensitivity với multicollinearity | Khuyến nghị |
|---|---|---|
| **Logistic Regression** | ⚠️ Rất nhạy | Bắt buộc check VIF, loại VIF > 10 |
| **Random Forest** | ✅ Ít nhạy | Vẫn nên check để interpret feature importance |
| **XGBoost** | ✅ Ít nhạy | Tương tự RF |
| **LSTM** | ⚠️ Khá nhạy | Nên giảm multicollinearity để training ổn định |
| **Transformer** | ⚠️ Khá nhạy | Tương tự LSTM |

### 5.5. Hậu quả nếu không check

- **Logistic Regression**: Coefficients không ổn định, dấu có thể bị flip
- **LSTM/Transformer**: Training chậm hơn, gradient không ổn định
- Feature importance bị "chia sẻ" giữa các biến tương quan
- Standard errors phồng to → không xác định được biến nào quan trọng
- Model overfitting do dùng dư thừa thông tin

### 5.6. Workflow đề xuất

```python
# Bước 1: Transform về returns
features_returns = df[['open', 'high', 'low', 'y_f1m', 'volume', 'vn30_index']].pct_change()

# Bước 2: Vẽ heatmap để có cái nhìn tổng quan
sns.heatmap(features_returns.corr(), annot=True)

# Bước 3: Tính VIF để quyết định loại bỏ
vif_df = compute_vif(features_returns.dropna())

# Bước 4: Loại từng feature có VIF cao nhất, lặp lại đến khi tất cả VIF < 10
```

---

## 6. ACF / PACF ANALYSIS

### 6.1. ACF (Autocorrelation Function)

**Định nghĩa:** Tương quan giữa quan sát hiện tại với chính nó tại các time lags khác nhau.

**Diễn giải pattern:**
- ACF dương ở lag thấp → **momentum** (xu hướng tiếp diễn)
- ACF âm ở lag thấp → **mean reversion** (đảo chiều)
- ACF gần 0 tất cả lags → **random walk** (khó dự đoán)

### 6.2. PACF (Partial Autocorrelation Function)

**Định nghĩa:** ACF nhưng loại bỏ ảnh hưởng của các lags trung gian.

**Tác dụng:** Xác định **đúng lags** nào nên dùng làm features cho LSTM/Transformer.

### 6.3. Ngưỡng significance

**Confidence band:** ±1.96/√n (với n = sample size)

| ACF/PACF tại lag k | Ý nghĩa |
|---|---|
| Vượt ra ngoài band | Có ý nghĩa thống kê |
| Trong band | Có thể là ngẫu nhiên |

**Với dataset hiện tại:**
- Daily (n=2,169, 2017–2026): confidence band ≈ ±0.042
- Hourly (n=2,953, **chỉ 2023–2026**): confidence band ≈ ±0.036 — nhưng lưu ý sample chỉ từ bull market, ACF pattern có thể không representative cho các regime khác

### 6.4. Diễn giải pattern điển hình

| Pattern | Nguyên nhân | Hành động |
|---|---|---|
| ACF decay nhanh, PACF cutoff | AR process | Dùng lags 1 đến p (cutoff point) |
| ACF cutoff, PACF decay | MA process | Khó dùng cho ML cổ điển |
| Cả hai decay từ từ | ARMA | Cần feature engineering |
| ACF decay rất chậm | Non-stationary | Phải differencing |

### 6.5. Tác dụng cho từng model

| Model | Cách dùng kết quả ACF/PACF |
|---|---|
| **Classical ML** | Chọn lag features (return_lag_1, return_lag_2, ...) từ significant PACF lags |
| **LSTM** | Xác định `sequence_length` dựa trên decay rate của ACF |
| **Transformer** | Xác định `context_window` dựa trên significant lags |

### 6.6. Hậu quả nếu bỏ qua

- Không biết lag nào nên dùng làm features → over-engineering hoặc miss important lags
- LSTM `sequence_length` chọn sai (quá ngắn miss pattern, quá dài noise)
- Bỏ sót **volatility clustering** (cực kỳ phổ biến trong finance)
- Không phát hiện **seasonality** (ví dụ: monthly rollover effect)

### 6.7. Áp dụng cho dataset VN30F1M

Từ kết quả ACF/PACF:
- **Returns daily**: significant lags = [1, 2, 12, 14, 30] → có slight mean reversion ở lag 1
- **Squared returns**: decay chậm từ 0.25 → **volatility clustering** rõ → nên thêm GARCH features
- **Volume change**: lag 1 = -0.28 (rất mạnh) → mean reversion mạnh trong volume

**Hourly data (2023-09-11 → 2026-05-12):** Cần test riêng vì có **lunch break gap** (11:30 → 13:00) và **overnight gap** (14:45 → 9:00 sáng hôm sau). Không thể treat như continuous series. Khi tính ACF/PACF cho hourly data, nên **loại bỏ first bar của mỗi ngày** (bar 10:00) ra khỏi overnight return calculation để tránh overnight gap làm nhiễu autocorrelation intraday thực sự.

---

## 7. INFORMATION COEFFICIENT (IC)

### 7.1. Định nghĩa

```
IC = Spearman_Correlation(feature, next_period_return)
```

Đo lường sức dự đoán của một feature với target tương lai.

**Tại sao dùng Spearman thay vì Pearson:**
- Robust với outliers
- Capture được monotonic relationships (không cần tuyến tính)
- Phù hợp với fat-tailed financial data

### 7.2. Ngưỡng đánh giá (Grinold & Kahn)

| \|IC\| | Quality | Đánh giá |
|---|---|---|
| > 0.10 | Excellent | Rất hiếm gặp, cảnh giác overfit |
| 0.05 – 0.10 | Good | Feature mạnh, đáng giữ |
| 0.02 – 0.05 | Moderate | Có ý nghĩa thực tiễn |
| 0.00 – 0.02 | Poor | Không có sức dự đoán |

**Lưu ý quan trọng:** Trong financial ML, IC = 0.05 đã được coi là "tốt". Đừng kỳ vọng IC = 0.5 như các domain khác.

### 7.3. Tại sao IC quan trọng?

- **Feature selection** dựa trên evidence, không phải intuition
- So sánh sức mạnh giữa các features
- Theo dõi feature degradation theo thời gian (rolling IC)
- Phát hiện features bị **leakage** (IC quá cao bất thường)

### 7.4. Áp dụng kết quả từ dataset (Daily VN30F1M)

| Feature | Công thức tóm tắt | IC | Quality | Quyết định |
|---|---|---|---|---|
| `basis_pct` | `(y_f1m - vn30_index) / vn30_index` | -0.121 | Excellent | ✅ Giữ |
| `body_pct` | `(y_f1m - open) / open` | -0.067 | Good | ✅ Giữ |
| `return_lag_1` | `return.shift(1)` | +0.051 | Good | ✅ Giữ |
| `return` | `y_f1m.pct_change()` | -0.046 | Moderate | ✅ Giữ (target) |
| `return_std_20` | `return.rolling(20).std()` | +0.042 | Moderate | ✅ Giữ |
| `daily_range` | `(high - low) / open` | +0.029 | Moderate | ✅ Giữ |
| `volume_change` | `volume.pct_change()` | -0.012 | Poor | ❌ Drop |
| `return_lag_2,3,5` | `return.shift(2/3/5)` | < 0.01 | Poor | ❌ Drop |

**Insights:**
- `basis_pct` là feature mạnh nhất vì capture **arbitrage pressure** giữa futures và cash — khi basis lệch lớn, thị trường có xu hướng mean-revert về equilibrium.
- IC âm của `basis_pct` và `body_pct` có nghĩa: basis cao hôm nay → return ngày mai thường âm (mean reversion), body xanh to hôm nay → tiếp theo thường pullback.
- IC dương của `return_lag_1` cho thấy có slight **momentum** ở lag 1.
- `volume_change` IC gần 0 → không có sức dự đoán, loại bỏ khỏi feature set.

### 7.5. Hậu quả nếu không kiểm tra

- Train model với features không có sức dự đoán → garbage in, garbage out
- Lãng phí computational resources cho dead features
- Model phức tạp nhưng không có edge thực sự

---

## 8. LOOK-AHEAD BIAS DETECTION

### 8.1. Định nghĩa

Sử dụng thông tin tương lai (chưa biết tại thời điểm dự đoán) trong features hoặc training.

### 8.2. Mức độ nguy hiểm

**CỰC KỲ NGHIÊM TRỌNG**. Theo Lopez de Prado, đây là nguyên nhân #1 khiến strategies thất bại khi triển khai live.

### 8.3. Các loại Look-Ahead Bias thường gặp

| Loại | Ví dụ | Cách phát hiện |
|---|---|---|
| **Target leakage** | Feature dùng `shift(-1)` | Audit feature engineering code |
| **Same-period** | Tính MA bao gồm giá hôm nay khi predict hôm nay | Verify rolling windows |
| **Cross-validation leakage** | Random k-fold cho time series | Dùng walk-forward CV |
| **Scaler leakage** | Fit scaler trên cả train+test | Fit chỉ trên train |

### 8.4. Phương pháp phát hiện

1. **Temporal Causality Test**: Future features không được correlate mạnh với past target
2. **Suspiciously High IC**: \|IC\| > 0.20 cho daily data → đáng ngờ
3. **Delay Test**: Thêm 1-day delay, performance không nên drop mạnh
4. **Accuracy threshold**: > 95% trên financial data → CHẮC CHẮN có bias

### 8.5. ⚠️ Lưu ý về False Positives

Test 1 (Temporal Causality) có thể **false positive** với:
- **Lag features**: `return_lag_1.shift(-1)` chính là `return` hiện tại → correlation = 1.0
- **Rolling features**: `return_mean_5.shift(-1)` chia sẻ 4/5 data points → correlation cao

→ **Skip lag/rolling features trong test này**, chỉ test trên features không phải lag.

Test 3 (Delay Test) chỉ meaningful khi **\|IC\| > 0.10**. Với IC nhỏ, % drop không có ý nghĩa.

### 8.6. Hậu quả nếu bỏ qua

- Backtest cho kết quả CỰC KỲ TỐT
- Live trading thua lỗ ngay lập tức
- Strategy "vanish" khi triển khai thực
- Mất tiền và uy tín

### 8.7. Best practices

```python
# RULE 1: Features chỉ dùng past data
df['feature_OK']  = df['close'].shift(1)   # ✅
df['feature_BAD'] = df['close'].shift(-1)  # ❌

# RULE 2: Labels là future data
df['label_OK']  = df['close'].shift(-5)    # ✅
df['label_BAD'] = df['close'].shift(5)     # ❌

# RULE 3: Rolling windows không bao gồm hiện tại
df['ma20_OK']  = df['close'].rolling(20).mean().shift(1)  # ✅
df['ma20_BAD'] = df['close'].rolling(20).mean()           # ⚠️

# RULE 4: Fit scaler chỉ trên train set
scaler.fit(X_train)
X_test_scaled = scaler.transform(X_test)  # ✅ KHÔNG fit lại
```

### 8.8. Đặc thù cho LSTM/Transformer

- **Sequence creation:** Đảm bảo `X[t]` chỉ chứa data từ `t-seq_len` đến `t-1`, KHÔNG bao gồm `t`
- **Normalization:** Compute mean/std trên train set, apply cho test set
- **Train/Val/Test split:** PHẢI theo thứ tự thời gian, KHÔNG shuffle

---

## 9. CLASS BALANCE (sau khi tạo labels)

### 9.1. Cách tạo labels

**Khuyến nghị: Triple Barrier Method** (Lopez de Prado) cho trend prediction task

```
Upper barrier: entry × (1 + pt_pct)    → label = +1 (Win)
Lower barrier: entry × (1 - sl_pct)    → label = -1 (Loss)
Time barrier:  max_holding days        → label = 0  (Timeout)
```

**Lý do chọn Triple Barrier:**
- Realistic: mô phỏng đúng cách trader giao dịch
- Loại bỏ noise: timeout cases có thể drop
- Có thể tune class balance qua PT/SL ratio
- Tạo binary labels (Up/Down) sau khi drop Timeout

### 9.2. Cấu hình đề xuất cho 2 timeframes và ảnh hưởng lên class balance

**Nguyên tắc cốt lõi:** `max_holding` quyết định tỷ lệ timeout class theo chiều **ngược lại với trực giác**:

> **Holding dài → ít timeout hơn** (thị trường có đủ thời gian chạm PT hoặc SL)
> **Holding ngắn → nhiều timeout hơn** (chưa đủ thời gian chạm barrier, giá vẫn ở giữa)

---

**Daily trading — max_holding dài → Timeout rất nhỏ → Binary Classifier**

```
PT = 2%, SL = 1%, max_holding = 10 days
```

Với 10 ngày và VN30F1M daily volatility ~1%/ngày, hầu hết positions đủ thời gian chạm PT (2%) hoặc SL (1%). Kết quả thực tế trên dataset:

| Class | Count | Tỷ lệ |
|---|---|---|
| Down (-1) | 1023 | 47.4% |
| Up (+1) | 958 | 44.4% |
| Timeout (0) | 178 | **8.2%** |

**Up và Down gần cân bằng nhau (47.4% vs 44.4%). Timeout chỉ 8.2% — quá nhỏ để là một class riêng có ý nghĩa thống kê.**

→ **Khuyến nghị: Drop timeout, chuyển sang Binary Classifier (Up vs Down).** Timeout 8.2% không đủ samples để model học được pattern của class này, giữ lại chỉ thêm noise.

```python
# Kiểm tra timeout rate thực tế trước khi quyết định
label_dist = df['label'].value_counts(normalize=True)
print(label_dist)

# Nếu timeout < 15% → drop timeout, chuyển binary
df_binary = df[df['label'] != 0].copy()
df_binary['label'] = (df_binary['label'] == 1).astype(int)
# Kết quả: Up=47.4%/(47.4%+44.4%) ≈ 52%, Down ≈ 48% — gần hoàn hảo
```

---

**Hourly trading — max_holding ngắn → Timeout cao hơn → Có thể giữ Triple Barrier**

```
PT = 0.5%, SL = 0.3%, max_holding = 5 bars (~1 ngày giao dịch)
```

Với chỉ 5 bars, nhiều positions chưa kịp chạm PT hoặc SL → timeout class lớn hơn đáng kể. Nếu cả 3 classes (Up/Down/Timeout) phân bố gần nhau (~30-35% mỗi class) → đây là điều kiện lý tưởng để giữ Triple Barrier đầy đủ.

→ **Kiểm tra timeout rate thực tế.** Nếu 3 classes cân bằng → giữ Triple Barrier. Nếu timeout vẫn nhỏ → chuyển binary.

---

**Bảng quyết định tổng quát:**

| max_holding | Timeout rate thực tế | Quyết định | Lý do |
|---|---|---|---|
| Ngắn (≤ 5 bars hourly) | Cao (có thể 25-40%) | ✅ Giữ Triple Barrier | 3 classes gần cân bằng |
| Dài (≥ 10 days daily) | Thấp (< 15%, ví dụ 8.2%) | ✅ Chuyển Binary | Timeout quá nhỏ, không đủ samples để học |

**Nguyên tắc chung:** Timeout rate thực tế phụ thuộc vào **tỷ lệ PT/SL so với volatility thị trường**. PT/SL càng gần volatility trung bình, timeout càng ít. Luôn kiểm tra phân phối thực tế trước khi quyết định.

**Nếu muốn giữ Triple Barrier với max_holding dài:** Tăng PT/SL (ví dụ `PT=5%, SL=3%, max_holding=10 days`) để timeout tăng lên — nhưng đánh đổi bằng việc labels phản ánh swing lớn hơn, ít samples hơn.

### 9.3. Imbalance Ratio

**Công thức:**
```
Imbalance Ratio = max(class_counts) / min(class_counts)
```

**Ngưỡng đánh giá:**

| IR | Mức độ | Hành động |
|---|---|---|
| 1.0 – 1.5 | Cân bằng | Không cần xử lý |
| 1.5 – 3.0 | Hơi mất cân bằng | `class_weight='balanced'` |
| 3.0 – 9.0 | Mất cân bằng | Class weights + threshold tuning |
| > 9.0 | Cực kỳ mất cân bằng | Resampling hoặc thay đổi labeling |

### 9.4. ⚠️ Lưu ý cho Triple Barrier

Với 3 classes (-1, 0, +1), imbalance ratio bị **inflate** do timeout class.

**Cách đúng:** So sánh **Up vs Down**, KHÔNG so với Timeout:

```python
up_count = (df['label'] == 1).sum()
down_count = (df['label'] == -1).sum()
real_imbalance = max(up_count, down_count) / min(up_count, down_count)
```

**Khuyến nghị:** Drop timeout cases và train binary classifier:

```python
df_binary = df[df['label'] != 0].copy()
df_binary['label'] = (df_binary['label'] == 1).astype(int)
```

### 9.5. Phân tích class balance theo thời gian

Class balance có thể vary theo market regime:

| Năm | Đặc điểm thị trường | Giá tham chiếu | Win/Loss rate kỳ vọng |
|---|---|---|---|
| 2017–2018 | Bull market mạnh | ~746 → ~1000 điểm | Win rate cao |
| 2019 | Sideways | ~900–950 điểm | Cân bằng, nhiều timeout |
| 2020 | COVID crash rồi recovery | ~579 → ~950 điểm | Volatile, Win/Loss xen kẽ |
| 2021 | Recovery rally | ~950 → ~1500 điểm | Win rate cao |
| 2022 | Bear market | ~1500 → ~1000 điểm | Loss rate cao |
| 2023–2024 | Bull market phục hồi | ~1000 → ~1700 điểm | Win rate cao |
| 2025–2026 | Bull market đỉnh cao | ~1700 → ~2089 điểm | Win rate rất cao |

→ Check yearly distribution để đảm bảo model robust qua các regimes.

### 9.6. Đối với Price Prediction (Regression)

Class balance KHÔNG áp dụng cho regression. Thay vào đó, kiểm tra:
- **Target distribution**: histogram của `next_return` để check normality
- **Outliers trong target**: tail events có thể skew model
- **Heteroskedasticity**: variance của target thay đổi theo thời gian không

### 9.7. Hậu quả nếu bỏ qua

- Model bias về majority class
- Accuracy 95% nhưng không thể trade được (toàn predict majority)
- Precision/Recall trade-off bị skew
- Trong production, signals minority class không reliable

---

## 10. VOLATILITY REGIME ANALYSIS

### 10.1. Tại sao cần?

VN30F1M trải qua nhiều **market regimes** khác nhau từ 2017–2026:
- 2017–2019: Bull market với volatility thấp (~746 → ~1000 điểm)
- 2020: COVID crash với volatility cực cao (giảm xuống ~579 điểm)
- 2021: Recovery rally (~900 → ~1500 điểm)
- 2022: Bear market (~1500 → ~1000 điểm)
- 2023–2026: Bull market trở lại (~1000 → ~2089 điểm)

**⚠️ Lưu ý đặc biệt cho Hourly data:** Dataset hourly chỉ bắt đầu từ 2023-09-11, tức là **chỉ có bull market 2023–2026**. Model train trên hourly data sẽ **không bao giờ thấy** COVID crash, bear market 2022. Đây là limitation nghiêm trọng cho regime generalization của hourly model.

Model train trên 1 regime có thể fail trên regime khác. Cần phát hiện regimes để:
- **Stratified train/test split** đảm bảo cả 2 set có đủ regimes
- **Adaptive model** với regime-specific parameters
- **Risk management** điều chỉnh position size theo regime

### 10.2. Phương pháp đo volatility

**Cách 1: Rolling Standard Deviation**
```python
df['vol_20d'] = df['return'].rolling(20).std() * np.sqrt(252)  # Annualized
```

**Cách 2: ATR (Average True Range)**
```python
df['tr'] = max(high-low, |high-prev_close|, |low-prev_close|)
df['atr_14'] = df['tr'].rolling(14).mean()
df['atr_pct'] = df['atr_14'] / df['close']  # Normalize
```

**Cách 3: EWMA Volatility (giống RiskMetrics)**
```python
df['ewma_vol'] = df['return'].ewm(span=20).std()
```

### 10.3. Phân loại regimes

**Theo Volatility Quantiles:**

| Regime | Definition | Tỷ lệ kỳ vọng |
|---|---|---|
| **Low Volatility** | vol < quantile(33%) | ~33% |
| **Medium Volatility** | quantile(33%) ≤ vol < quantile(67%) | ~33% |
| **High Volatility** | vol ≥ quantile(67%) | ~33% |

**Ngưỡng đánh giá thực tiễn cho VN30F1M (daily, annualized):**

| Annualized Vol | Regime |
|---|---|
| < 12% | Low (calm market) |
| 12% – 20% | Normal |
| 20% – 30% | High |
| > 30% | Extreme (crisis) |

### 10.4. Tác dụng cho model training

| Task | Cách áp dụng |
|---|---|
| **Train/test split** | Đảm bảo train và test có distribution regimes tương tự |
| **Feature engineering** | Thêm `vol_regime` làm categorical feature |
| **Model selection** | Train separate models cho từng regime |
| **Loss weighting** | Sample weights cao hơn cho high-vol periods |

### 10.5. Hậu quả nếu bỏ qua

- Train trên low-vol (2017–2019) → fail trên high-vol (2020)
- Backtest tốt nhưng live trading thua lỗ khi market regime đổi
- Model overconfident trong regimes nó chưa từng "thấy"
- Không thể explain tại sao model fail trong certain periods

### 10.6. Workflow đề xuất

```python
# Bước 1: Tính rolling volatility
df['vol_20d'] = df['return'].rolling(20).std() * np.sqrt(252)

# Bước 2: Phân loại regime
df['regime'] = pd.cut(
    df['vol_20d'],
    bins=[0, 0.12, 0.20, 0.30, np.inf],
    labels=['Low', 'Normal', 'High', 'Extreme']
)

# Bước 3: Kiểm tra distribution theo năm
yearly_regime = df.groupby([df['time'].dt.year, 'regime']).size().unstack()
print(yearly_regime)

# Bước 4: Đảm bảo train/test có đủ regimes
print("Train regimes:", df_train['regime'].value_counts(normalize=True))
print("Test regimes:", df_test['regime'].value_counts(normalize=True))
```

### 10.7. Volatility Clustering (GARCH effect)

Đây là phát hiện quan trọng từ ACF/PACF analysis:
- ACF của **squared returns** decay chậm
- → Volatility hôm nay correlated với volatility hôm qua
- → Có thể dự đoán volatility (dễ hơn dự đoán direction)

**Tận dụng:** Thêm features:
- `volatility_lag_1`, `volatility_lag_5`
- `volatility_ma_20`
- `volatility_zscore` (so với baseline)

---

## 11. LƯU Ý ĐẶC THÙ CHO DỮ LIỆU VNSTOCK / KBS

### 11.1. Về nguồn dữ liệu vnstock

Thư viện `vnstock` crawl data qua API các công ty chứng khoán (mặc định KBS). Cần lưu ý:

| Vấn đề | Mô tả | Khuyến nghị |
|---|---|---|
| **Adjusted vs Unadjusted** | Data có thể đã được adjust hoặc chưa | Verify với nguồn khác (TradingView) |
| **Survivorship bias** | API có thể chỉ trả về data của symbols còn active | Với VN30F1M không có vấn đề này |
| **Data revisions** | KBS có thể update lại data lịch sử | Cache data offline để có version cố định |
| **Rate limits** | API có giới hạn requests/phút | Implement retry logic và caching |

### 11.2. Corporate Actions và Rollover Adjustments

**Bạn ghi chú là chưa rõ về 2 khái niệm này. Đây là giải thích:**

**Corporate Actions (cho stock futures):**
- Cổ tức (dividends), chia tách (stock split), phát hành thêm
- Ảnh hưởng đến giá underlying → futures price phải adjust theo
- **VN30F1M là index futures**, nên ít bị ảnh hưởng trực tiếp bởi corporate actions của từng stock

**Rollover Adjustments (CỰC KỲ QUAN TRỌNG cho futures):**

VN30F1M là hợp đồng **1 tháng**, đáo hạn ngày Thứ 5 tuần thứ 3 mỗi tháng. Khi đáo hạn:
- Hợp đồng cũ (VN30F1M tháng N) → hết hạn
- Hợp đồng mới (VN30F1M tháng N+1) → trở thành "front month"
- Giá giữa 2 hợp đồng thường có **gap** (contango hoặc backwardation)

**Ảnh hưởng đến data:**
- Nếu data là "continuous contract" → đã được adjust để loại bỏ gap
- Nếu data là "raw front month" → có price jumps tại ngày rollover
- **Bạn cần verify với vnstock data của bạn**

**Cách kiểm tra:**
```python
# Tìm các ngày có price jump bất thường
df['return'] = df['y_f1m'].pct_change()
extreme_days = df[abs(df['return']) > 0.05]  # >5% jump
print(extreme_days)

# So sánh với lịch rollover (Thứ 5 tuần 3 mỗi tháng)
# Nếu các extreme days trùng với rollover dates → data CHƯA adjust
```

**Khuyến nghị:** Liên hệ với vnstock documentation hoặc test với:
```python
from vnstock import Vnstock
help(Vnstock)  # Check parameters về adjustment
```

### 11.3. Đặc thù thị trường Việt Nam

| Yếu tố | Đặc điểm | Ảnh hưởng |
|---|---|---|
| **Trading hours** | 9:00-11:30, 13:00-14:45 | Có lunch break → gap trong hourly data |
| **Settlement** | T+2 cho cash, T+0 cho futures | Cần lưu ý khi tính P&L |
| **Margin** | ~20% cho VN30F1M | Đòn bẩy 5x → MDD quan trọng gấp 5 lần |
| **Price limits** | ±7% cho stocks | Futures có thể có giới hạn rộng hơn |
| **Holidays** | Tết, 30/4, 2/9, ... | Cần custom calendar |
| **Circuit breakers** | Có thể tạm dừng giao dịch | Tạo gaps trong data |

### 11.4. Hậu quả nếu bỏ qua các đặc thù

- Backtest dùng raw front month data → fake P&L lớn tại rollover dates
- Hourly model treat lunch break như continuous → ACF/PACF sai
- Risk management không account leverage → margin call thực tế

---

## 12. TỔNG KẾT WORKFLOW ĐỀ XUẤT

```
STEP 1: Tiền xử lý & Feature Engineering
├── Verify y_f1m = close price (đã confirm)
├── Transform mandatory: open, high, low, y_f1m, volume, vn30_index → pct_change()
├── Tạo ratio features (không cần transform thêm):
│   ├── basis_pct = (y_f1m - vn30_index) / vn30_index  [IC=-0.121, Excellent]
│   ├── body_pct  = (y_f1m - open) / open               [IC=-0.067, Good]
│   └── daily_range = (high - low) / open               [IC=+0.029, Moderate]
├── Tạo lag features (chỉ giữ significant lags):
│   └── return_lag_1 = return.shift(1)                  [IC=+0.051, Good]
├── Tạo rolling features (nhớ shift(1) trước khi dùng):
│   └── return_std_20 = return.rolling(20).std()        [IC=+0.042, Moderate]
└── Verify rollover adjustment status (xem Section 11.2)

STEP 2: Kiểm tra Data Integrity
├── Missing Value Ratio (>95% completeness)
├── Trading Day Coverage (so với HOSE calendar)
├── OHLC Consistency (high≥max(O,C), low≤min(O,C))
└── Bar Completeness cho hourly data

STEP 3: Kiểm tra Time Series Properties
├── ADF + KPSS test trên returns
├── ACF/PACF analysis để chọn lags
└── Volatility regime analysis

STEP 4: Outlier Detection
└── IQR (3x multiplier) trên returns, KHÔNG trên prices

STEP 5: Feature Analysis
├── Pearson Heatmap (visualize correlations)
├── VIF (loại features có VIF > 10)
└── IC (loại features có |IC| < 0.02)

STEP 6: Label Generation
├── Triple Barrier Method cho trend task
│   ├── Daily: PT=2%, SL=1%, hold=10
│   └── Hourly: PT=0.5%, SL=0.3%, hold=5
├── Check class balance Up vs Down
└── Target distribution check cho price prediction task

STEP 7: Look-Ahead Bias Audit
├── Temporal Causality Test (skip lag/rolling features)
├── Suspiciously High IC check
└── Delay Test (chỉ cho features có |IC| > 0.10)

STEP 8: Train/Test Split
├── Chronological split (không shuffle)
├── Đảm bảo cả 2 set có đủ regimes
└── Embargo period giữa train và test
```

---

## 13. NGƯỠNG QUYẾT ĐỊNH NHANH

| Metric | Tốt | Cảnh báo | Phải xử lý |
|---|---|---|---|
| Missing Ratio | < 1% | 1–5% | > 5% |
| ADF p-value | < 0.05 | 0.05–0.10 | > 0.10 |
| KPSS p-value | > 0.10 | 0.05–0.10 | < 0.05 |
| Imbalance Ratio | < 1.5 | 1.5–3 | > 3 |
| Pearson \|r\| | < 0.3 | 0.3–0.7 | > 0.7 |
| VIF | < 5 | 5–10 | > 10 |
| Outliers (3×IQR) | < 1% | 1–3% | > 3% |
| IC | > 0.05 | 0.02–0.05 | < 0.02 |
| Suspicious accuracy | < 70% | 70–90% | > 95% |
| Annualized Vol | < 20% | 20–30% | > 30% |

---

## 14. TÀI LIỆU THAM KHẢO

1. **Lopez de Prado, M.** (2018). *Advances in Financial Machine Learning*. Wiley.
   - Chapter 3: Labeling (Triple Barrier Method)
   - Chapter 7: Cross-Validation in Finance
2. **Grinold, R. & Kahn, R.** (1999). *Active Portfolio Management*. McGraw-Hill.
   - Chapter 6: Information Coefficient
3. **Tsay, R.** (2010). *Analysis of Financial Time Series*. Wiley.
   - Chapter 2: Stationarity and Unit Root Tests
4. **Statsmodels Documentation** – Stationarity tests (ADF/KPSS)
5. **vnstock Documentation** – https://github.com/thinh-vu/vnstock
6. **HOSE (Sở giao dịch chứng khoán TP.HCM)** – Quy định giao dịch phái sinh VN30F1M

---

## PHỤ LỤC: CHECKLIST KIỂM TRA NHANH

Trước khi đưa data vào training, đảm bảo đã check tất cả:

- [ ] Đã transform OHLC, Volume, VN30 về pct_change (tạo: return, open_return, high_return, low_return, volume_change, vn30_return)
- [ ] Đã tạo ratio features: basis_pct, body_pct, daily_range (không cần transform thêm)
- [ ] Đã tạo lag features: return_lag_1 (giữ), return_lag_2,3,5 (drop — IC < 0.01)
- [ ] Đã tạo rolling features: return_std_20 (nhớ shift(1) khi dùng làm input)
- [ ] Missing values < 1%
- [ ] Trading day coverage > 95% so với HOSE calendar
- [ ] OHLC consistency 100% (no violations)
- [ ] ADF + KPSS confirm stationarity trên returns
- [ ] ACF/PACF đã xác định significant lags
- [ ] IQR (3x) phát hiện < 3% outliers trên returns
- [ ] Pearson heatmap đã visualized
- [ ] Tất cả VIF < 10
- [ ] IC > 0.02 cho tất cả features được giữ
- [ ] Triple barrier labels đã tạo, class balance OK
- [ ] Look-ahead bias đã audit (skip false positives)
- [ ] Volatility regimes đã được map
- [ ] Train/test split chronological, có embargo
- [ ] Rollover adjustment status đã verify

---

*Báo cáo này tập trung vào Giai đoạn 1 (Data Quality). Các giai đoạn 2 (Model Evaluation) và 3 (Trading Performance) sẽ được trình bày trong các báo cáo riêng.*