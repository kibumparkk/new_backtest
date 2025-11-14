# 백테스트 프로젝트

암호화폐 트레이딩 전략의 백테스팅을 위한 프로젝트입니다.

## 프로젝트 구조

```
new_backtest/
├── chart_day/           # 일별 가격 데이터 (Parquet 형식)
├── output/              # 백테스트 결과 저장 폴더
│   ├── *.png           # 시각화 결과
│   └── *.csv           # 성과 지표 및 월별 수익률
└── requirements.txt     # 필요한 Python 패키지
```

## 설치 방법

```bash
pip install -r requirements.txt
```

## 데이터 구조

### 데이터 파일 형식
- **파일 형식**: Parquet
- **위치**: `chart_day/` 폴더
- **파일명 패턴**: `{COIN}_KRW.parquet` (예: `BTC_KRW.parquet`)

### 데이터 스키마

모든 데이터 파일은 다음과 같은 구조를 가집니다:

| 컬럼명 | 데이터 타입 | 설명 |
|--------|------------|------|
| `timestamp` | datetime (index) | 날짜 (인덱스) |
| `open` | float64 | 시가 (KRW) |
| `high` | float64 | 고가 (KRW) |
| `low` | float64 | 저가 (KRW) |
| `close` | float64 | 종가 (KRW) |
| `volume` | float64 | 거래량 (코인 수량) |

### BTC 데이터 예시

```python
import pandas as pd

df = pd.read_parquet('chart_day/BTC_KRW.parquet')
print(df.head())
```

**출력 예시:**
```
                 open       high        low      close      volume
timestamp
2017-09-25  4201000.0  4333000.0  4175000.0  4322000.0  132.484755
2017-09-26  4317000.0  4418000.0  4311000.0  4321000.0   22.788340
2017-09-27  4322000.0  4677000.0  4318000.0  4657000.0   32.269662
2017-09-28  4657000.0  4772000.0  4519000.0  4586000.0   80.588243
2017-09-29  4586000.0  4709000.0  4476000.0  4657000.0   59.352373
```

### 데이터 특징
- **기간**: 2017-09-25 ~ 현재 (약 2,969일 이상)
- **결측치**: 없음
- **화폐 단위**: KRW (대한민국 원)
- **거래량 단위**: 코인 개수 (BTC, ETH 등)
- **인덱스**: timestamp (datetime 형식)

### 데이터 로드 방법

```python
import pandas as pd

# 데이터 로드
df = pd.read_parquet('chart_day/BTC_KRW.parquet')

# 인덱스가 이미 timestamp로 설정되어 있음
# 추가 설정 필요 없이 바로 사용 가능
print(f"데이터 기간: {df.index.min()} ~ {df.index.max()}")
print(f"총 데이터 수: {len(df)}일")
```

## 백테스트 가이드라인

### 1. 기본 설정

#### 1.1 백테스트 대상
- **기본**: BTC (BTC_KRW.parquet)만 백테스트
- 특별한 언급이 있는 경우에만 다른 코인 추가

#### 1.2 벤치마크 전략
- **전략**: 전일종가 > SMA30 (30일 단순이동평균)
- 조건 충족 시 매수, 미충족 시 매도/관망

#### 1.3 거래 비용
- **슬리피지**: 0.2% (매수/매도 시 각각 적용)
- 실제 체결가격 = 이론가격 × (1 ± 0.002)

### 2. 구현 요구사항

#### 2.1 이중 검증 (Cross-Check)
모든 백테스트 로직은 **두 가지 방식으로 구현**하여 결과를 비교 검증해야 합니다:

1. **벡터화 연산** (Pandas/NumPy)
   ```python
   # 예시: 벡터화 구현
   df['signal'] = (df['close'].shift(1) > df['sma30']).astype(int)
   df['returns'] = df['close'].pct_change()
   df['strategy_returns'] = df['signal'].shift(1) * df['returns']
   ```

2. **반복문 구현**
   ```python
   # 예시: 반복문 구현
   for i in range(1, len(df)):
       if df.loc[i-1, 'close'] > df.loc[i-1, 'sma30']:
           signal[i] = 1
       # ... 계산 로직
   ```

3. **검증**
   - 두 방법의 최종 수익률 차이가 0.01% 이내여야 함
   - 차이가 있을 경우 로직 오류로 판단하고 수정

#### 2.2 성과 과대평가 방지

다음 사항들을 반드시 고려하여 구현해야 합니다:

1. **Look-Ahead Bias 방지**
   - 당일 데이터로 당일 거래 금지
   - 신호 생성 시 `.shift(1)` 사용 필수
   - 예: t일 종가로 t+1일 거래

2. **Survivorship Bias 인지**
   - 현재 상장된 코인만 포함된 데이터 사용 시 유의
   - 상장폐지된 코인 미포함으로 인한 편향 인지

3. **Data Snooping 방지**
   - 전체 기간 데이터로 파라미터 최적화 지양
   - 가능하면 In-Sample / Out-of-Sample 분리

4. **거래비용 현실화**
   - 슬리피지 0.2% 필수 반영
   - 거래소 수수료 고려 (필요시)

5. **유동성 제약**
   - 대량 거래 시 시장 충격 고려
   - 소형 코인의 경우 추가 슬리피지 검토

### 3. 필수 산출 지표

#### 3.1 수익률 지표

1. **Total Return (총 수익률)**
   - **표기 방식**: 배수 (x)로 표기
   - **계산 공식**:
   ```python
   total_return = final_equity / initial_capital
   # 예시: 1원 → 10원이면 10.0x
   # 예시: 1원 → 2.5원이면 2.5x
   ```
   - 초기 자본: 1원
   - 복리 계산으로 누적
   - **중요**: 퍼센트(%)가 아닌 배수로 표기

2. **CAGR (연평균 복리수익률)**
   ```python
   # 계산 공식
   total_days = (end_date - start_date).days
   years = total_days / 365.25
   CAGR = (final_value / initial_value) ** (1 / years) - 1
   ```

3. **월별 수익률**
   - 각 월의 수익률 계산
   - CSV 파일로 저장: `output/monthly_returns.csv`

#### 3.2 리스크 지표

1. **MDD (Maximum Drawdown)**
   ```python
   # 계산 공식
   cumulative = (1 + returns).cumprod()
   running_max = cumulative.cummax()
   drawdown = (cumulative - running_max) / running_max * 100
   MDD = drawdown.min()
   ```

2. **Sharpe Ratio** (선택사항)
3. **Sortino Ratio** (선택사항)
4. **Win Rate** (선택사항)

### 4. 시각화 요구사항

**중요**: 모든 그래프는 하나의 그림에 서브플롯으로 구성하여 `output/backtest_results.png` 파일로 저장합니다.

#### 4.1 전체 레이아웃 구성

하나의 figure에 3개의 subplot을 다음과 같이 배치합니다:
- **subplot 1**: 누적 자산 곡선 (상단)
- **subplot 2**: Drawdown 차트 (중단)
- **subplot 3**: 월별 수익률 히트맵 (하단)

#### 4.2 구현 예시

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Figure 생성: 3개의 subplot (세로 배치)
fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 2], hspace=0.3)

# Subplot 1: 누적 자산 곡선 (로그 스케일)
ax1 = fig.add_subplot(gs[0])
ax1.plot(df.index, df['strategy_equity'], label='Strategy', linewidth=2)
ax1.plot(df.index, df['benchmark_equity'], label='Benchmark (SMA30)', linewidth=2, alpha=0.7)
ax1.set_yscale('log')
ax1.set_ylabel('Cumulative Returns (KRW, log scale)', fontsize=11)
ax1.set_title('Backtest Performance Analysis', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Subplot 2: Drawdown 차트 (%)
ax2 = fig.add_subplot(gs[1])
ax2.fill_between(df.index, 0, df['drawdown_pct'], color='red', alpha=0.3)
ax2.plot(df.index, df['drawdown_pct'], color='red', linewidth=1)
ax2.set_ylabel('Drawdown (%)', fontsize=11)
ax2.set_xlabel('Date', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linewidth=0.5)

# Subplot 3: 월별 수익률 히트맵
ax3 = fig.add_subplot(gs[2])

# 월별 수익률을 피벗 테이블로 변환
monthly_rets = df['strategy_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
pivot_table = monthly_rets.groupby([monthly_rets.index.year, monthly_rets.index.month]).sum().unstack()
pivot_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            ax=ax3, cbar_kws={'label': 'Monthly Return (%)'})
ax3.set_ylabel('Year', fontsize=11)
ax3.set_xlabel('Month', fontsize=11)
ax3.set_title('Monthly Returns (%)', fontsize=12)

# 저장
plt.savefig('output/backtest_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("시각화 완료: output/backtest_results.png")
```

#### 4.3 각 subplot 세부 요구사항

**Subplot 1: 누적 자산 곡선**
- Y축: 로그 스케일 (log scale)
- 초기값: 1원
- 포함 요소:
  - 전략 수익률 곡선
  - 벤치마크 수익률 곡선 (전일종가 > SMA30)
  - Buy & Hold 수익률 (선택사항)

**Subplot 2: Drawdown 차트**
- Y축: 퍼센트 (%)
- 형식: Area chart (영역 그래프)
- 색상: 빨간색 계열
- 0% 기준선 표시

**Subplot 3: 월별 수익률 히트맵**
- 형식: 히트맵 (Seaborn)
- 행: 연도 (Year)
- 열: 월 (Month, Jan-Dec)
- 컬러맵: RdYlGn (빨강-노랑-초록)
- 중심값: 0%
- 각 셀에 수치 표시

### 5. 결과 저장

#### 5.1 폴더 생성
```python
import os
os.makedirs('output', exist_ok=True)
```

#### 5.2 저장 파일 목록

1. **CSV 파일**
   - `output/performance_summary.csv`: 전체 성과 요약
     - Total Return (배수 x로 표기)
     - CAGR (%)
     - MDD (%)
     - 기타 성과 지표
   - `output/monthly_returns.csv`: 월별 수익률
   - `output/trade_log.csv`: 거래 내역 (선택사항)

2. **PNG 파일**
   - `output/backtest_results.png`: 백테스트 결과 종합 시각화
     - Subplot 1: 누적 수익률 (로그 스케일)
     - Subplot 2: Drawdown 차트
     - Subplot 3: 월별 수익률 히트맵

### 6. 백테스트 체크리스트

백테스트 실행 전 다음 사항을 확인하세요:

- [ ] BTC_KRW.parquet 데이터 존재 확인
- [ ] output/ 폴더 생성
- [ ] 벡터화 + 반복문 두 가지 방식 구현
- [ ] 두 방식의 결과 차이 < 0.01% 확인
- [ ] 슬리피지 0.2% 적용 확인
- [ ] Look-ahead bias 방지 확인 (shift 사용)
- [ ] Total Return 계산 (배수 x로 표기)
- [ ] CAGR 계산 구현
- [ ] MDD 계산 구현
- [ ] 하나의 그림에 3개 subplot으로 시각화 생성
  - [ ] Subplot 1: 누적 자산 그래프 (log scale)
  - [ ] Subplot 2: Drawdown 그래프 (%)
  - [ ] Subplot 3: 월별 수익률 히트맵
- [ ] 벤치마크 (전일종가 > SMA30) 비교

### 7. 코드 템플릿 예시

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# 출력 폴더 생성
os.makedirs('output', exist_ok=True)

# 데이터 로드
df = pd.read_parquet('chart_day/BTC_KRW.parquet')
# timestamp가 이미 인덱스로 설정되어 있음

# 초기 설정
INITIAL_CAPITAL = 1  # 1원
SLIPPAGE = 0.002     # 0.2%

# 데이터 확인
print(f"데이터 기간: {df.index.min()} ~ {df.index.max()}")
print(f"총 {len(df)}일")
print(f"컬럼: {list(df.columns)}")

# SMA 계산
df['sma30'] = df['close'].rolling(window=30).mean()

# 벤치마크 전략: 전일종가 > SMA30
df['benchmark_signal'] = (df['close'].shift(1) > df['sma30'].shift(1)).astype(int)

# === 방법 1: 벡터화 구현 ===
# (여기에 벡터화 로직 구현)

# === 방법 2: 반복문 구현 ===
# (여기에 반복문 로직 구현)

# === 결과 비교 ===
# assert abs(final_return_vectorized - final_return_loop) < 0.0001

# === 성과 지표 계산 ===
# Total Return (배수)
# total_return = final_equity / INITIAL_CAPITAL
# CAGR, MDD, 월별 수익률 등

# === 시각화 ===
# 하나의 그림에 3개 subplot 생성 (섹션 4 참조)
# - Subplot 1: 누적 수익률 (로그 스케일)
# - Subplot 2: Drawdown (%)
# - Subplot 3: 월별 수익률 히트맵
# output/backtest_results.png 저장

# === 결과 저장 ===
# CSV 파일 저장: performance_summary.csv, monthly_returns.csv

print("백테스트 완료!")
print(f"Total Return: {total_return:.2f}x")
print(f"CAGR: {cagr:.2%}")
print(f"MDD: {mdd:.2%}")
print(f"최종 자산: {final_equity:,.0f}원")
print(f"결과 저장 완료: output/backtest_results.png")
```

## 참고사항

- 데이터는 `chart_day/` 폴더의 Parquet 파일을 사용합니다
- 모든 코인은 KRW 거래쌍입니다
- 백테스트 기간은 데이터 범위에 따라 자동 설정됩니다
- 추가 전략 구현 시에도 동일한 가이드라인을 따라주세요

## 라이선스

MIT License
