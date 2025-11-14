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

모든 그래프는 `output/` 폴더에 PNG 형식으로 저장합니다.

#### 4.1 누적 자산 곡선
- **파일명**: `output/cumulative_returns.png`
- **Y축**: 로그 스케일 (log scale)
- **초기값**: 1원
- **포함 요소**:
  - 전략 수익률 곡선
  - 벤치마크 수익률 곡선
  - Buy & Hold 수익률 (참고용)

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df['strategy_equity'], label='Strategy')
ax.plot(df.index, df['benchmark_equity'], label='Benchmark (SMA30)')
ax.set_yscale('log')
ax.set_ylabel('Cumulative Returns (KRW, log scale)')
ax.set_xlabel('Date')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('output/cumulative_returns.png', dpi=300, bbox_inches='tight')
```

#### 4.2 Drawdown 차트
- **파일명**: `output/drawdown.png`
- **Y축**: 퍼센트 (%)
- **형식**: Area chart (영역 그래프)
- **색상**: 빨간색 계열 권장

```python
fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(df.index, 0, df['drawdown_pct'], color='red', alpha=0.3)
ax.plot(df.index, df['drawdown_pct'], color='red', linewidth=1)
ax.set_ylabel('Drawdown (%)')
ax.set_xlabel('Date')
ax.grid(True, alpha=0.3)
plt.savefig('output/drawdown.png', dpi=300, bbox_inches='tight')
```

#### 4.3 월별 수익률 히트맵
- **파일명**: `output/monthly_returns_heatmap.png`
- **형식**: 히트맵 (Seaborn)
- **행**: 연도, **열**: 월

```python
import seaborn as sns

# 월별 수익률을 피벗 테이블로 변환
monthly_rets = df['returns'].resample('M').sum() * 100  # 퍼센트로 변환
pivot_table = monthly_rets.groupby([monthly_rets.index.year, monthly_rets.index.month]).sum().unstack()

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax)
ax.set_ylabel('Year')
ax.set_xlabel('Month')
plt.savefig('output/monthly_returns_heatmap.png', dpi=300, bbox_inches='tight')
```

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
   - `output/cumulative_returns.png`: 누적 수익률 (로그 스케일)
   - `output/drawdown.png`: Drawdown 차트
   - `output/monthly_returns_heatmap.png`: 월별 수익률 히트맵

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
- [ ] 누적 자산 그래프 (log scale) 생성
- [ ] Drawdown 그래프 (%) 생성
- [ ] 월별 수익률 히트맵 생성
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
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# 초기 설정
INITIAL_CAPITAL = 1  # 1원
SLIPPAGE = 0.002     # 0.2%

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
# 누적 수익률, Drawdown, 월별 수익률 히트맵

# === 결과 저장 ===
# CSV 및 PNG 파일 저장

print("백테스트 완료!")
print(f"Total Return: {total_return:.2f}x")
print(f"CAGR: {cagr:.2%}")
print(f"MDD: {mdd:.2%}")
print(f"최종 자산: {final_equity:,.0f}원")
```

## 참고사항

- 데이터는 `chart_day/` 폴더의 Parquet 파일을 사용합니다
- 모든 코인은 KRW 거래쌍입니다
- 백테스트 기간은 데이터 범위에 따라 자동 설정됩니다
- 추가 전략 구현 시에도 동일한 가이드라인을 따라주세요

## 라이선스

MIT License
