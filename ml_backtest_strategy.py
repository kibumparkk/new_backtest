"""
머신러닝 기반 백테스트 전략

Random Forest 모델을 사용하여 기술적 지표를 기반으로
다음날 가격 움직임을 예측하는 전략입니다.

전략 개요:
- 모델: Random Forest Classifier
- 피처: SMA, RSI, MACD, Bollinger Bands, 거래량 지표 등
- 타겟: 다음날 수익률이 양수인지 여부
- 학습 방식: Walk-forward analysis (시계열 교차 검증)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 출력 폴더 생성
os.makedirs('output', exist_ok=True)

# 데이터 로드
print("=" * 60)
print("머신러닝 기반 백테스트 전략")
print("=" * 60)
print("\n데이터 로딩 중...")
df = pd.read_parquet('chart_day/BTC_KRW.parquet')

# 초기 설정
INITIAL_CAPITAL = 1  # 1원
SLIPPAGE = 0.002     # 0.2%

print(f"데이터 기간: {df.index.min()} ~ {df.index.max()}")
print(f"총 {len(df)}일")
print(f"컬럼: {list(df.columns)}\n")

# ===================================
# 1. 기술적 지표 계산
# ===================================
print("기술적 지표 계산 중...")

# 이동평균선
df['sma5'] = df['close'].rolling(window=5).mean()
df['sma10'] = df['close'].rolling(window=10).mean()
df['sma20'] = df['close'].rolling(window=20).mean()
df['sma30'] = df['close'].rolling(window=30).mean()
df['sma60'] = df['close'].rolling(window=60).mean()

# 가격과 이동평균선 비율
df['close_sma5_ratio'] = df['close'] / df['sma5']
df['close_sma20_ratio'] = df['close'] / df['sma20']
df['close_sma60_ratio'] = df['close'] / df['sma60']

# RSI (Relative Strength Index)
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['rsi'] = calculate_rsi(df['close'], period=14)

# MACD (Moving Average Convergence Divergence)
exp1 = df['close'].ewm(span=12, adjust=False).mean()
exp2 = df['close'].ewm(span=26, adjust=False).mean()
df['macd'] = exp1 - exp2
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
df['macd_diff'] = df['macd'] - df['macd_signal']

# Bollinger Bands
df['bb_middle'] = df['close'].rolling(window=20).mean()
bb_std = df['close'].rolling(window=20).std()
df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

# 거래량 지표
df['volume_sma20'] = df['volume'].rolling(window=20).mean()
df['volume_ratio'] = df['volume'] / df['volume_sma20']

# 가격 변화율
df['price_change_1d'] = df['close'].pct_change(1)
df['price_change_5d'] = df['close'].pct_change(5)
df['price_change_20d'] = df['close'].pct_change(20)

# 변동성 (표준편차)
df['volatility_20d'] = df['close'].pct_change().rolling(window=20).std()

# High-Low 범위
df['hl_ratio'] = (df['high'] - df['low']) / df['close']

# 수익률 계산 (타겟 변수)
df['returns'] = df['close'].pct_change()
df['next_day_return'] = df['returns'].shift(-1)  # 다음날 수익률
df['target'] = (df['next_day_return'] > 0).astype(int)  # 다음날 상승 여부

# ===================================
# 2. 벤치마크 전략 (SMA30)
# ===================================
print("벤치마크 전략 (SMA30) 계산 중...")

# 시그널 생성: 전일 데이터로 당일 포지션 결정
df['benchmark_signal'] = (df['close'].shift(1) > df['sma30'].shift(1)).astype(int)

# 벤치마크 전략 수익률 (슬리피지 미적용)
df['benchmark_returns'] = df['benchmark_signal'] * df['returns']

# 벤치마크 자산 곡선 계산
df['benchmark_equity'] = INITIAL_CAPITAL * (1 + df['benchmark_returns']).cumprod()

# ===================================
# 3. 머신러닝 전략 구현
# ===================================
print("\n머신러닝 모델 학습 및 예측 중...")
print("이 작업은 시간이 걸릴 수 있습니다...\n")

# 피처 선택
feature_columns = [
    'close_sma5_ratio', 'close_sma20_ratio', 'close_sma60_ratio',
    'rsi', 'macd', 'macd_diff', 'bb_position',
    'volume_ratio', 'price_change_1d', 'price_change_5d',
    'price_change_20d', 'volatility_20d', 'hl_ratio'
]

# 결측치 제거
df_clean = df.dropna().copy()

print(f"학습 가능한 데이터: {len(df_clean)}일")
print(f"피처 개수: {len(feature_columns)}개")
print(f"피처 목록: {feature_columns}\n")

# Walk-forward analysis 설정
TRAIN_PERIOD = 365  # 학습 기간: 1년
RETRAIN_PERIOD = 90  # 재학습 주기: 3개월

# 예측 결과 저장
predictions = pd.Series(index=df_clean.index, dtype=float)
predictions[:] = 0  # 초기값 0 (현금 보유)

# Walk-forward 학습 및 예측
n_splits = 0
for i in range(TRAIN_PERIOD, len(df_clean), RETRAIN_PERIOD):
    # 학습 데이터: 과거 TRAIN_PERIOD일
    train_end = i
    train_start = max(0, train_end - TRAIN_PERIOD)

    # 예측 데이터: 다음 RETRAIN_PERIOD일
    test_start = train_end
    test_end = min(len(df_clean), train_end + RETRAIN_PERIOD)

    # 학습 데이터 준비
    X_train = df_clean.iloc[train_start:train_end][feature_columns]
    y_train = df_clean.iloc[train_start:train_end]['target']

    # 테스트 데이터 준비
    X_test = df_clean.iloc[test_start:test_end][feature_columns]

    # 모델 학습
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # 예측
    pred = model.predict(X_test_scaled)
    predictions.iloc[test_start:test_end] = pred

    n_splits += 1

    if n_splits % 5 == 0:
        print(f"진행상황: {n_splits}번째 재학습 완료 (날짜: {df_clean.index[test_start]})")

print(f"\n총 {n_splits}번 재학습 완료\n")

# 예측 결과를 원본 데이터프레임에 병합
df_clean['ml_signal'] = predictions

# 시그널에 shift(1) 적용: 전날 예측으로 당일 포지션 결정
df_clean['ml_signal_shifted'] = df_clean['ml_signal'].shift(1)

# ML 전략 수익률 계산 (슬리피지 미적용)
df_clean['ml_returns'] = df_clean['ml_signal_shifted'] * df_clean['returns']

# ML 전략 자산 곡선 계산
df_clean['ml_equity'] = INITIAL_CAPITAL * (1 + df_clean['ml_returns']).cumprod()

# Buy & Hold 전략
df_clean['bnh_equity'] = INITIAL_CAPITAL * (1 + df_clean['returns']).cumprod()

# ===================================
# 4. 성과 지표 계산
# ===================================
print("성과 지표 계산 중...")

def calculate_performance_metrics(returns, equity, strategy_name):
    """성과 지표 계산 함수"""
    # 결측치 제거
    returns = returns.dropna()
    equity = equity.dropna()

    # Total Return (배수)
    total_return = equity.iloc[-1] / INITIAL_CAPITAL

    # CAGR 계산
    total_days = (equity.index[-1] - equity.index[0]).days
    years = total_days / 365.25
    cagr = (equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1

    # MDD 계산
    cumulative = equity
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    mdd = drawdown.min()

    # Sharpe Ratio (연환산)
    daily_returns = returns
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

    # Win Rate
    winning_trades = (returns > 0).sum()
    total_trades = (returns != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # 연간 거래 횟수
    trades_per_year = total_trades / years if years > 0 else 0

    return {
        'Strategy': strategy_name,
        'Total Return': f"{total_return:.2f}x",
        'CAGR': f"{cagr:.2%}",
        'MDD': f"{mdd:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Win Rate': f"{win_rate:.2%}",
        'Total Trades': int(total_trades),
        'Trades/Year': f"{trades_per_year:.1f}",
        'Final Equity': f"{equity.iloc[-1]:,.0f}"
    }

# 각 전략 성과 계산
ml_metrics = calculate_performance_metrics(
    df_clean['ml_returns'],
    df_clean['ml_equity'],
    'ML Strategy (Random Forest)'
)

benchmark_metrics = calculate_performance_metrics(
    df_clean['benchmark_returns'],
    df_clean['benchmark_equity'],
    'Benchmark (SMA30)'
)

bnh_metrics = calculate_performance_metrics(
    df_clean['returns'],
    df_clean['bnh_equity'],
    'Buy & Hold'
)

# 성과 요약 출력
print("\n" + "=" * 60)
print("백테스트 결과 요약")
print("=" * 60)

for strategy in [ml_metrics, benchmark_metrics, bnh_metrics]:
    print(f"\n{strategy['Strategy']}")
    print("-" * 60)
    for key, value in strategy.items():
        if key != 'Strategy':
            print(f"  {key:20s}: {value}")

# CSV 파일로 저장
performance_df = pd.DataFrame([ml_metrics, benchmark_metrics, bnh_metrics])
performance_df.to_csv('output/performance_summary.csv', index=False, encoding='utf-8-sig')
print("\n성과 요약 저장: output/performance_summary.csv")

# 월별 수익률 계산
monthly_returns_ml = df_clean['ml_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
monthly_returns_df = pd.DataFrame({
    'Month': monthly_returns_ml.index.strftime('%Y-%m'),
    'ML Strategy Return (%)': monthly_returns_ml.values
})
monthly_returns_df.to_csv('output/monthly_returns.csv', index=False, encoding='utf-8-sig')
print("월별 수익률 저장: output/monthly_returns.csv")

# ===================================
# 5. 시각화
# ===================================
print("\n시각화 생성 중...")

# Figure 생성: 3개의 subplot (세로 배치)
fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 2], hspace=0.3)

# Subplot 1: 누적 자산 곡선 (로그 스케일)
ax1 = fig.add_subplot(gs[0])
ax1.plot(df_clean.index, df_clean['ml_equity'], label='ML Strategy (Random Forest)', linewidth=2.5, color='#2E86AB')
ax1.plot(df_clean.index, df_clean['benchmark_equity'], label='Benchmark (SMA30)', linewidth=2, alpha=0.7, color='#A23B72')
ax1.plot(df_clean.index, df_clean['bnh_equity'], label='Buy & Hold', linewidth=2, alpha=0.5, color='#F18F01', linestyle='--')
ax1.set_yscale('log')
ax1.set_ylabel('Cumulative Returns (KRW, log scale)', fontsize=12, fontweight='bold')
ax1.set_title('ML-Based Backtest Performance Analysis', fontsize=16, fontweight='bold', pad=20)
ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')

# 주요 성과지표 텍스트 박스 추가 (우측 상단)
# years 계산
total_days = (df_clean.index[-1] - df_clean.index[0]).days
years = total_days / 365.25

ml_total_return = df_clean['ml_equity'].iloc[-1] / INITIAL_CAPITAL
ml_cagr = (df_clean['ml_equity'].iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1
ml_drawdown = ((df_clean['ml_equity'] - df_clean['ml_equity'].cummax()) / df_clean['ml_equity'].cummax()).min()
ml_sharpe = (df_clean['ml_returns'].mean() / df_clean['ml_returns'].std()) * np.sqrt(252)

metrics_text = f'''ML Strategy Performance:
Total Return: {ml_total_return:.2f}x
CAGR: {ml_cagr:.2%}
MDD: {ml_drawdown:.2%}
Sharpe: {ml_sharpe:.2f}'''

ax1.text(0.98, 0.97, metrics_text, transform=ax1.transAxes,
         fontsize=11, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
         family='monospace')

# Subplot 2: Drawdown 차트 (%)
ax2 = fig.add_subplot(gs[1])
ml_cumulative = df_clean['ml_equity']
ml_running_max = ml_cumulative.cummax()
ml_drawdown_series = (ml_cumulative - ml_running_max) / ml_running_max * 100

ax2.fill_between(df_clean.index, 0, ml_drawdown_series, color='#D62828', alpha=0.3)
ax2.plot(df_clean.index, ml_drawdown_series, color='#D62828', linewidth=1.5)
ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.axhline(y=0, color='black', linewidth=0.8)
ax2.set_title('Drawdown Analysis', fontsize=12, fontweight='bold')

# Subplot 3: 월별 수익률 히트맵
ax3 = fig.add_subplot(gs[2])

# 월별 수익률을 피벗 테이블로 변환
monthly_rets = df_clean['ml_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
monthly_rets.index = pd.to_datetime(monthly_rets.index)

# 연도와 월로 분리
years_list = monthly_rets.index.year
months_list = monthly_rets.index.month

# 데이터프레임 생성
monthly_df = pd.DataFrame({
    'year': years_list,
    'month': months_list,
    'return': monthly_rets.values
})

# 피벗 테이블 생성
pivot_table = monthly_df.pivot(index='year', columns='month', values='return')
pivot_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# 히트맵 생성
sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            ax=ax3, cbar_kws={'label': 'Monthly Return (%)'},
            linewidths=0.5, linecolor='gray')
ax3.set_ylabel('Year', fontsize=12, fontweight='bold')
ax3.set_xlabel('Month', fontsize=12, fontweight='bold')
ax3.set_title('Monthly Returns Heatmap (%)', fontsize=12, fontweight='bold')

# 저장
plt.tight_layout()
plt.savefig('output/backtest_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("시각화 완료: output/backtest_results.png")

# ===================================
# 6. 피처 중요도 분석
# ===================================
print("\n피처 중요도 분석 중...")

# 마지막 모델의 피처 중요도 (참고용)
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n피처 중요도 TOP 10:")
print(feature_importance.head(10).to_string(index=False))

feature_importance.to_csv('output/feature_importance.csv', index=False, encoding='utf-8-sig')
print("\n피처 중요도 저장: output/feature_importance.csv")

# ===================================
# 7. 거래 분석
# ===================================
print("\n거래 분석 중...")

# 포지션 변화 감지
df_clean['position_change'] = df_clean['ml_signal_shifted'].diff()
trades = df_clean[df_clean['position_change'] != 0].copy()

print(f"총 거래 횟수: {len(trades)}회")
print(f"매수 신호: {(trades['position_change'] == 1).sum()}회")
print(f"매도 신호: {(trades['position_change'] == -1).sum()}회")

# 거래 로그 저장
if len(trades) > 0:
    trade_log = trades[['close', 'ml_signal_shifted', 'position_change', 'ml_returns']].copy()
    trade_log.to_csv('output/trade_log.csv', encoding='utf-8-sig')
    print("거래 로그 저장: output/trade_log.csv")

print("\n" + "=" * 60)
print("백테스트 완료!")
print("=" * 60)
print("\n생성된 파일:")
print("  - output/backtest_results.png      : 백테스트 결과 시각화")
print("  - output/performance_summary.csv   : 성과 요약")
print("  - output/monthly_returns.csv       : 월별 수익률")
print("  - output/feature_importance.csv    : 피처 중요도")
print("  - output/trade_log.csv             : 거래 로그")
print("\n" + "=" * 60)
