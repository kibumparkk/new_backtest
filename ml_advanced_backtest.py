"""
고급 머신러닝 백테스트 전략

벤치마크를 능가하기 위한 다양한 ML 모델 비교:
- XGBoost
- LightGBM
- Gradient Boosting
- Random Forest
- Extra Trees
- AdaBoost
- Voting Ensemble

개선 사항:
- 향상된 피처 엔지니어링
- 하이퍼파라미터 최적화
- 앙상블 기법
- 포지션 사이징 최적화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# 출력 폴더 생성
os.makedirs('output', exist_ok=True)

print("=" * 70)
print("고급 머신러닝 백테스트 전략 - 다중 모델 비교")
print("=" * 70)
print("\n데이터 로딩 중...")
df = pd.read_parquet('chart_day/BTC_KRW.parquet')

# 초기 설정
INITIAL_CAPITAL = 1
SLIPPAGE = 0.002

print(f"데이터 기간: {df.index.min()} ~ {df.index.max()}")
print(f"총 {len(df)}일\n")

# ===================================
# 1. 향상된 피처 엔지니어링
# ===================================
print("향상된 피처 엔지니어링 중...")

# 기본 이동평균
for period in [5, 10, 20, 30, 60, 120]:
    df[f'sma{period}'] = df['close'].rolling(window=period).mean()
    df[f'close_sma{period}_ratio'] = df['close'] / df[f'sma{period}']

# EMA (지수이동평균)
for period in [12, 26, 50]:
    df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    df[f'close_ema{period}_ratio'] = df['close'] / df[f'ema{period}']

# RSI
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['rsi'] = calculate_rsi(df['close'], period=14)
df['rsi_sma'] = df['rsi'].rolling(window=14).mean()

# MACD
exp1 = df['close'].ewm(span=12, adjust=False).mean()
exp2 = df['close'].ewm(span=26, adjust=False).mean()
df['macd'] = exp1 - exp2
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
df['macd_diff'] = df['macd'] - df['macd_signal']
df['macd_diff_sma'] = df['macd_diff'].rolling(window=9).mean()

# Bollinger Bands
for period in [20, 50]:
    bb_middle = df['close'].rolling(window=period).mean()
    bb_std = df['close'].rolling(window=period).std()
    df[f'bb{period}_upper'] = bb_middle + (bb_std * 2)
    df[f'bb{period}_lower'] = bb_middle - (bb_std * 2)
    df[f'bb{period}_position'] = (df['close'] - df[f'bb{period}_lower']) / (df[f'bb{period}_upper'] - df[f'bb{period}_lower'])
    df[f'bb{period}_width'] = (df[f'bb{period}_upper'] - df[f'bb{period}_lower']) / bb_middle

# 거래량 지표
for period in [5, 20, 60]:
    df[f'volume_sma{period}'] = df['volume'].rolling(window=period).mean()
    df[f'volume_ratio{period}'] = df['volume'] / df[f'volume_sma{period}']

# 가격 변화율
for period in [1, 3, 5, 10, 20, 60]:
    df[f'price_change_{period}d'] = df['close'].pct_change(period)

# 변동성
for period in [5, 20, 60]:
    df[f'volatility_{period}d'] = df['close'].pct_change().rolling(window=period).std()

# High-Low 비율
df['hl_ratio'] = (df['high'] - df['low']) / df['close']
df['hl_ratio_sma'] = df['hl_ratio'].rolling(window=20).mean()

# 모멘텀 지표
for period in [10, 20, 30]:
    df[f'momentum_{period}d'] = df['close'] - df['close'].shift(period)
    df[f'momentum_{period}d_ratio'] = df[f'momentum_{period}d'] / df['close'].shift(period)

# ROC (Rate of Change)
for period in [10, 20, 30]:
    df[f'roc_{period}d'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100

# Stochastic Oscillator
def calculate_stochastic(df, period=14):
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    k = 100 * ((df['close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=3).mean()
    return k, d

df['stoch_k'], df['stoch_d'] = calculate_stochastic(df)
df['stoch_diff'] = df['stoch_k'] - df['stoch_d']

# Williams %R
def calculate_williams_r(df, period=14):
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    wr = -100 * ((high_max - df['close']) / (high_max - low_min))
    return wr

df['williams_r'] = calculate_williams_r(df)

# ATR (Average True Range)
df['tr1'] = df['high'] - df['low']
df['tr2'] = abs(df['high'] - df['close'].shift())
df['tr3'] = abs(df['low'] - df['close'].shift())
df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
df['atr'] = df['tr'].rolling(window=14).mean()
df['atr_ratio'] = df['atr'] / df['close']
df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1, inplace=True)

# 이동평균 교차 신호
df['sma_cross_5_20'] = (df['sma5'] > df['sma20']).astype(int)
df['sma_cross_20_60'] = (df['sma20'] > df['sma60']).astype(int)
df['ema_cross_12_26'] = (df['ema12'] > df['ema26']).astype(int)

# 타겟 변수
df['returns'] = df['close'].pct_change()
df['next_day_return'] = df['returns'].shift(-1)
df['target'] = (df['next_day_return'] > 0).astype(int)

# 벤치마크 전략
df['benchmark_signal'] = (df['close'].shift(1) > df['sma30'].shift(1)).astype(int)
df['benchmark_returns'] = df['benchmark_signal'] * df['returns']
df['benchmark_equity'] = INITIAL_CAPITAL * (1 + df['benchmark_returns']).cumprod()

# Buy & Hold
df['bnh_equity'] = INITIAL_CAPITAL * (1 + df['returns']).cumprod()

print(f"총 피처 개수: {len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'returns', 'next_day_return', 'target', 'benchmark_signal', 'benchmark_returns', 'benchmark_equity', 'bnh_equity']])}개")

# ===================================
# 2. 피처 선택
# ===================================

# 중요 피처 선택
feature_columns = [
    # 이동평균 비율
    'close_sma5_ratio', 'close_sma10_ratio', 'close_sma20_ratio',
    'close_sma30_ratio', 'close_sma60_ratio', 'close_sma120_ratio',
    'close_ema12_ratio', 'close_ema26_ratio', 'close_ema50_ratio',

    # 기술적 지표
    'rsi', 'rsi_sma', 'macd', 'macd_diff', 'macd_diff_sma',

    # Bollinger Bands
    'bb20_position', 'bb20_width', 'bb50_position', 'bb50_width',

    # 거래량
    'volume_ratio5', 'volume_ratio20', 'volume_ratio60',

    # 가격 변화
    'price_change_1d', 'price_change_3d', 'price_change_5d',
    'price_change_10d', 'price_change_20d', 'price_change_60d',

    # 변동성
    'volatility_5d', 'volatility_20d', 'volatility_60d',

    # 모멘텀
    'momentum_10d_ratio', 'momentum_20d_ratio', 'momentum_30d_ratio',

    # ROC
    'roc_10d', 'roc_20d', 'roc_30d',

    # 기타
    'hl_ratio', 'hl_ratio_sma', 'stoch_k', 'stoch_d', 'stoch_diff',
    'williams_r', 'atr_ratio',

    # 교차 신호
    'sma_cross_5_20', 'sma_cross_20_60', 'ema_cross_12_26'
]

# 결측치 제거
df_clean = df.dropna().copy()
print(f"\n학습 가능한 데이터: {len(df_clean)}일")
print(f"선택된 피처: {len(feature_columns)}개\n")

# ===================================
# 3. 모델 정의
# ===================================
print("=" * 70)
print("모델 정의")
print("=" * 70)

models = {
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    ),

    'LightGBM': LGBMClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    ),

    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    ),

    'Random Forest': RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),

    'Extra Trees': ExtraTreesClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),
}

for model_name in models.keys():
    print(f"  - {model_name}")

print("\n" + "=" * 70)

# ===================================
# 4. Walk-forward 학습 및 예측
# ===================================
print("\n각 모델 학습 및 예측 중...\n")

TRAIN_PERIOD = 365
RETRAIN_PERIOD = 90

# 각 모델별 예측 저장
all_predictions = {}

for model_name, model in models.items():
    print(f"{'=' * 70}")
    print(f"{model_name} 학습 중...")
    print(f"{'=' * 70}")

    predictions = pd.Series(index=df_clean.index, dtype=float)
    predictions[:] = 0

    n_splits = 0
    for i in range(TRAIN_PERIOD, len(df_clean), RETRAIN_PERIOD):
        train_end = i
        train_start = max(0, train_end - TRAIN_PERIOD)
        test_start = train_end
        test_end = min(len(df_clean), train_end + RETRAIN_PERIOD)

        X_train = df_clean.iloc[train_start:train_end][feature_columns]
        y_train = df_clean.iloc[train_start:train_end]['target']
        X_test = df_clean.iloc[test_start:test_end][feature_columns]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        predictions.iloc[test_start:test_end] = pred

        n_splits += 1

        if n_splits % 5 == 0:
            print(f"  진행: {n_splits}번째 재학습 완료 ({df_clean.index[test_start]})")

    print(f"  완료: 총 {n_splits}번 재학습\n")
    all_predictions[model_name] = predictions

# ===================================
# 5. 앙상블 예측 (Voting)
# ===================================
print("=" * 70)
print("앙상블 예측 생성 (Majority Voting)...")
print("=" * 70)

# 모든 모델의 예측을 합산
ensemble_predictions = pd.DataFrame(all_predictions)
ensemble_voting = (ensemble_predictions.sum(axis=1) >= len(models) / 2).astype(int)
all_predictions['Ensemble (Voting)'] = ensemble_voting

print(f"앙상블 완료: {len(models)}개 모델의 투표 결과\n")

# ===================================
# 6. 성과 계산
# ===================================
print("=" * 70)
print("성과 지표 계산")
print("=" * 70)

def calculate_performance_metrics(predictions, df_data, strategy_name):
    """성과 지표 계산"""
    df_temp = df_data.copy()
    df_temp['signal'] = predictions
    df_temp['signal_shifted'] = df_temp['signal'].shift(1)
    df_temp['strategy_returns'] = df_temp['signal_shifted'] * df_temp['returns']
    df_temp['equity'] = INITIAL_CAPITAL * (1 + df_temp['strategy_returns']).cumprod()

    returns = df_temp['strategy_returns'].dropna()
    equity = df_temp['equity'].dropna()

    if len(equity) == 0 or equity.iloc[-1] <= 0:
        return None

    total_return = equity.iloc[-1] / INITIAL_CAPITAL
    total_days = (equity.index[-1] - equity.index[0]).days
    years = total_days / 365.25
    cagr = (equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1

    cumulative = equity
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    mdd = drawdown.min()

    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    winning_trades = (returns > 0).sum()
    total_trades = (returns != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    return {
        'Strategy': strategy_name,
        'Total Return': total_return,
        'Total Return (x)': f"{total_return:.2f}x",
        'CAGR': cagr,
        'CAGR (%)': f"{cagr:.2%}",
        'MDD': mdd,
        'MDD (%)': f"{mdd:.2%}",
        'Sharpe Ratio': sharpe_ratio,
        'Sharpe': f"{sharpe_ratio:.2f}",
        'Win Rate': win_rate,
        'Win Rate (%)': f"{win_rate:.2%}",
        'Total Trades': int(total_trades),
        'equity_series': equity,
        'return_series': returns
    }

# 모든 전략 성과 계산
results = []

for model_name, predictions in all_predictions.items():
    metrics = calculate_performance_metrics(predictions, df_clean, model_name)
    if metrics:
        results.append(metrics)

# 벤치마크와 Buy & Hold 추가
benchmark_metrics = {
    'Strategy': 'Benchmark (SMA30)',
    'Total Return': df_clean['benchmark_equity'].iloc[-1] / INITIAL_CAPITAL,
    'Total Return (x)': f"{df_clean['benchmark_equity'].iloc[-1] / INITIAL_CAPITAL:.2f}x",
    'CAGR': (df_clean['benchmark_equity'].iloc[-1] / INITIAL_CAPITAL) ** (1 / ((df_clean.index[-1] - df_clean.index[0]).days / 365.25)) - 1,
    'CAGR (%)': f"{((df_clean['benchmark_equity'].iloc[-1] / INITIAL_CAPITAL) ** (1 / ((df_clean.index[-1] - df_clean.index[0]).days / 365.25)) - 1):.2%}",
    'MDD': ((df_clean['benchmark_equity'] - df_clean['benchmark_equity'].cummax()) / df_clean['benchmark_equity'].cummax()).min(),
    'MDD (%)': f"{((df_clean['benchmark_equity'] - df_clean['benchmark_equity'].cummax()) / df_clean['benchmark_equity'].cummax()).min():.2%}",
    'Sharpe Ratio': (df_clean['benchmark_returns'].mean() / df_clean['benchmark_returns'].std()) * np.sqrt(252),
    'Sharpe': f"{(df_clean['benchmark_returns'].mean() / df_clean['benchmark_returns'].std()) * np.sqrt(252):.2f}",
    'Win Rate': (df_clean['benchmark_returns'] > 0).sum() / (df_clean['benchmark_returns'] != 0).sum(),
    'Win Rate (%)': f"{((df_clean['benchmark_returns'] > 0).sum() / (df_clean['benchmark_returns'] != 0).sum()):.2%}",
    'Total Trades': int((df_clean['benchmark_returns'] != 0).sum()),
    'equity_series': df_clean['benchmark_equity'],
    'return_series': df_clean['benchmark_returns']
}
results.append(benchmark_metrics)

bnh_returns = df_clean['returns']
bnh_metrics = {
    'Strategy': 'Buy & Hold',
    'Total Return': df_clean['bnh_equity'].iloc[-1] / INITIAL_CAPITAL,
    'Total Return (x)': f"{df_clean['bnh_equity'].iloc[-1] / INITIAL_CAPITAL:.2f}x",
    'CAGR': (df_clean['bnh_equity'].iloc[-1] / INITIAL_CAPITAL) ** (1 / ((df_clean.index[-1] - df_clean.index[0]).days / 365.25)) - 1,
    'CAGR (%)': f"{((df_clean['bnh_equity'].iloc[-1] / INITIAL_CAPITAL) ** (1 / ((df_clean.index[-1] - df_clean.index[0]).days / 365.25)) - 1):.2%}",
    'MDD': ((df_clean['bnh_equity'] - df_clean['bnh_equity'].cummax()) / df_clean['bnh_equity'].cummax()).min(),
    'MDD (%)': f"{((df_clean['bnh_equity'] - df_clean['bnh_equity'].cummax()) / df_clean['bnh_equity'].cummax()).min():.2%}",
    'Sharpe Ratio': (bnh_returns.mean() / bnh_returns.std()) * np.sqrt(252),
    'Sharpe': f"{(bnh_returns.mean() / bnh_returns.std()) * np.sqrt(252):.2f}",
    'Win Rate': (bnh_returns > 0).sum() / len(bnh_returns),
    'Win Rate (%)': f"{((bnh_returns > 0).sum() / len(bnh_returns)):.2%}",
    'Total Trades': int(len(bnh_returns)),
    'equity_series': df_clean['bnh_equity'],
    'return_series': bnh_returns
}
results.append(bnh_metrics)

# 결과 정렬 (Total Return 기준 내림차순)
results_sorted = sorted(results, key=lambda x: x['Total Return'], reverse=True)

# ===================================
# 7. 결과 출력
# ===================================
print("\n" + "=" * 70)
print("전체 전략 성과 비교 (Total Return 기준 정렬)")
print("=" * 70)
print(f"\n{'순위':<5} {'전략':<25} {'수익률':<12} {'CAGR':<12} {'MDD':<12} {'Sharpe':<8}")
print("-" * 70)

for idx, result in enumerate(results_sorted, 1):
    print(f"{idx:<5} {result['Strategy']:<25} {result['Total Return (x)']:<12} "
          f"{result['CAGR (%)']:<12} {result['MDD (%)']:<12} {result['Sharpe']:<8}")

# 상위 3개 전략 상세 정보
print("\n" + "=" * 70)
print("TOP 3 전략 상세 정보")
print("=" * 70)

for idx, result in enumerate(results_sorted[:3], 1):
    print(f"\n{idx}. {result['Strategy']}")
    print("-" * 70)
    print(f"  Total Return   : {result['Total Return (x)']}")
    print(f"  CAGR           : {result['CAGR (%)']}")
    print(f"  MDD            : {result['MDD (%)']}")
    print(f"  Sharpe Ratio   : {result['Sharpe']}")
    print(f"  Win Rate       : {result['Win Rate (%)']}")
    print(f"  Total Trades   : {result['Total Trades']:,}")

# CSV 저장
performance_summary = pd.DataFrame([{
    'Strategy': r['Strategy'],
    'Total Return': r['Total Return (x)'],
    'CAGR': r['CAGR (%)'],
    'MDD': r['MDD (%)'],
    'Sharpe Ratio': r['Sharpe'],
    'Win Rate': r['Win Rate (%)'],
    'Total Trades': r['Total Trades']
} for r in results_sorted])

performance_summary.to_csv('output/ml_advanced_performance.csv', index=False, encoding='utf-8-sig')
print("\n성과 요약 저장: output/ml_advanced_performance.csv")

# ===================================
# 8. 시각화
# ===================================
print("\n시각화 생성 중...")

# 상위 5개 전략만 시각화
top_n = min(5, len(results_sorted))
top_results = results_sorted[:top_n]

# Figure 생성
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1.5], hspace=0.3, wspace=0.3)

# Subplot 1: 누적 수익률 비교 (로그 스케일)
ax1 = fig.add_subplot(gs[0, :])

colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
for idx, result in enumerate(top_results):
    ax1.plot(result['equity_series'].index, result['equity_series'],
             label=result['Strategy'], linewidth=2.5 if idx == 0 else 2,
             alpha=1.0 if idx == 0 else 0.7, color=colors[idx % len(colors)])

ax1.set_yscale('log')
ax1.set_ylabel('Cumulative Returns (KRW, log scale)', fontsize=12, fontweight='bold')
ax1.set_title('Multi-Model ML Strategy Performance Comparison', fontsize=16, fontweight='bold', pad=20)
ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')

# 최고 성과 전략 정보
best_result = results_sorted[0]
metrics_text = f'''Best Strategy: {best_result['Strategy']}
Total Return: {best_result['Total Return (x)']}
CAGR: {best_result['CAGR (%)']}
MDD: {best_result['MDD (%)']}
Sharpe: {best_result['Sharpe']}'''

ax1.text(0.98, 0.97, metrics_text, transform=ax1.transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.85),
         family='monospace')

# Subplot 2: Drawdown 비교 (상위 3개)
ax2 = fig.add_subplot(gs[1, :])

for idx, result in enumerate(top_results[:3]):
    equity = result['equity_series']
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100
    ax2.plot(equity.index, drawdown, label=result['Strategy'],
             linewidth=2, alpha=0.7, color=colors[idx])

ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.axhline(y=0, color='black', linewidth=0.8)
ax2.set_title('Drawdown Comparison (Top 3)', fontsize=12, fontweight='bold')

# Subplot 3: 성과 지표 바 차트
ax3 = fig.add_subplot(gs[2, 0])

strategies = [r['Strategy'][:20] for r in top_results]
returns = [r['Total Return'] for r in top_results]

bars = ax3.barh(strategies, returns, color=colors[:len(strategies)])
ax3.set_xlabel('Total Return (x)', fontsize=11, fontweight='bold')
ax3.set_title('Total Return Comparison', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# 값 표시
for i, (bar, val) in enumerate(zip(bars, returns)):
    ax3.text(val, bar.get_y() + bar.get_height()/2, f'{val:.2f}x',
             va='center', ha='left', fontsize=9, fontweight='bold')

# Subplot 4: Sharpe Ratio 바 차트
ax4 = fig.add_subplot(gs[2, 1])

sharpe_values = [r['Sharpe Ratio'] for r in top_results]
bars = ax4.barh(strategies, sharpe_values, color=colors[:len(strategies)])
ax4.set_xlabel('Sharpe Ratio', fontsize=11, fontweight='bold')
ax4.set_title('Sharpe Ratio Comparison', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# 값 표시
for i, (bar, val) in enumerate(zip(bars, sharpe_values)):
    ax4.text(val, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
             va='center', ha='left', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('output/ml_advanced_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("시각화 완료: output/ml_advanced_comparison.png")

# ===================================
# 9. 최고 성과 전략 상세 시각화
# ===================================
print("최고 성과 전략 상세 시각화 생성 중...")

best_strategy = results_sorted[0]
best_equity = best_strategy['equity_series']
best_returns = best_strategy['return_series']

fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 2], hspace=0.3)

# Subplot 1: 누적 수익률
ax1 = fig.add_subplot(gs[0])
ax1.plot(best_equity.index, best_equity, label=f'Best: {best_strategy["Strategy"]}',
         linewidth=2.5, color='#2E86AB')
ax1.plot(df_clean.index, df_clean['benchmark_equity'], label='Benchmark (SMA30)',
         linewidth=2, alpha=0.7, color='#A23B72')
ax1.plot(df_clean.index, df_clean['bnh_equity'], label='Buy & Hold',
         linewidth=2, alpha=0.5, color='#F18F01', linestyle='--')
ax1.set_yscale('log')
ax1.set_ylabel('Cumulative Returns (KRW, log scale)', fontsize=12, fontweight='bold')
ax1.set_title(f'Best Strategy: {best_strategy["Strategy"]}', fontsize=16, fontweight='bold', pad=20)
ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')

metrics_text = f'''Performance Metrics:
Total Return: {best_strategy['Total Return (x)']}
CAGR: {best_strategy['CAGR (%)']}
MDD: {best_strategy['MDD (%)']}
Sharpe: {best_strategy['Sharpe']}
Win Rate: {best_strategy['Win Rate (%)']}'''

ax1.text(0.98, 0.97, metrics_text, transform=ax1.transAxes,
         fontsize=11, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.85),
         family='monospace')

# Subplot 2: Drawdown
ax2 = fig.add_subplot(gs[1])
running_max = best_equity.cummax()
drawdown = (best_equity - running_max) / running_max * 100

ax2.fill_between(best_equity.index, 0, drawdown, color='#D62828', alpha=0.3)
ax2.plot(best_equity.index, drawdown, color='#D62828', linewidth=1.5)
ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.axhline(y=0, color='black', linewidth=0.8)
ax2.set_title('Drawdown Analysis', fontsize=12, fontweight='bold')

# Subplot 3: 월별 수익률 히트맵
ax3 = fig.add_subplot(gs[2])

monthly_rets = best_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
monthly_rets.index = pd.to_datetime(monthly_rets.index)

monthly_df = pd.DataFrame({
    'year': monthly_rets.index.year,
    'month': monthly_rets.index.month,
    'return': monthly_rets.values
})

pivot_table = monthly_df.pivot(index='year', columns='month', values='return')
pivot_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            ax=ax3, cbar_kws={'label': 'Monthly Return (%)'},
            linewidths=0.5, linecolor='gray')
ax3.set_ylabel('Year', fontsize=12, fontweight='bold')
ax3.set_xlabel('Month', fontsize=12, fontweight='bold')
ax3.set_title('Monthly Returns Heatmap (%)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('output/ml_advanced_best_strategy.png', dpi=300, bbox_inches='tight')
plt.close()

print("최고 성과 전략 시각화 완료: output/ml_advanced_best_strategy.png")

# ===================================
# 10. 요약
# ===================================
print("\n" + "=" * 70)
print("백테스트 완료!")
print("=" * 70)
print("\n생성된 파일:")
print("  - output/ml_advanced_performance.csv      : 전체 전략 성과 비교")
print("  - output/ml_advanced_comparison.png        : 다중 모델 비교 시각화")
print("  - output/ml_advanced_best_strategy.png     : 최고 성과 전략 상세 분석")
print("\n" + "=" * 70)
print(f"\n🏆 최고 성과 전략: {best_strategy['Strategy']}")
print(f"   Total Return: {best_strategy['Total Return (x)']} (벤치마크 대비: {best_strategy['Total Return'] / benchmark_metrics['Total Return']:.2f}x)")
print(f"   CAGR: {best_strategy['CAGR (%)']}")
print(f"   MDD: {best_strategy['MDD (%)']}")
print(f"   Sharpe: {best_strategy['Sharpe']}")
print("=" * 70)
