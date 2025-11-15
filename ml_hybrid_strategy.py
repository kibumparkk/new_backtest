"""
하이브리드 머신러닝 전략

ML 모델의 확률 예측 + 기술적 신호(SMA30)를 결합하여
벤치마크를 능가하는 전략을 개발합니다.

전략:
1. ML 확률 기반 필터링
2. SMA30 + ML 신뢰도 조합
3. 동적 임계값 설정
4. 다중 조건 필터링
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

os.makedirs('output', exist_ok=True)

print("=" * 70)
print("하이브리드 ML 전략 - 벤치마크 능가 목표")
print("=" * 70)
print("\n데이터 로딩 중...")

df = pd.read_parquet('chart_day/BTC_KRW.parquet')

INITIAL_CAPITAL = 1
SLIPPAGE = 0.002

print(f"데이터 기간: {df.index.min()} ~ {df.index.max()}")
print(f"총 {len(df)}일\n")

# ===================================
# 1. 피처 엔지니어링
# ===================================
print("피처 엔지니어링 중...")

# 이동평균
for period in [5, 10, 20, 30, 60, 120]:
    df[f'sma{period}'] = df['close'].rolling(window=period).mean()
    df[f'close_sma{period}_ratio'] = df['close'] / df[f'sma{period}']

# EMA
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

# Bollinger Bands
for period in [20, 50]:
    bb_middle = df['close'].rolling(window=period).mean()
    bb_std = df['close'].rolling(window=period).std()
    df[f'bb{period}_upper'] = bb_middle + (bb_std * 2)
    df[f'bb{period}_lower'] = bb_middle - (bb_std * 2)
    df[f'bb{period}_position'] = (df['close'] - df[f'bb{period}_lower']) / (df[f'bb{period}_upper'] - df[f'bb{period}_lower'])
    df[f'bb{period}_width'] = (df[f'bb{period}_upper'] - df[f'bb{period}_lower']) / bb_middle

# 거래량
for period in [5, 20, 60]:
    df[f'volume_sma{period}'] = df['volume'].rolling(window=period).mean()
    df[f'volume_ratio{period}'] = df['volume'] / df[f'volume_sma{period}']

# 가격 변화
for period in [1, 3, 5, 10, 20, 60]:
    df[f'price_change_{period}d'] = df['close'].pct_change(period)

# 변동성
for period in [5, 20, 60]:
    df[f'volatility_{period}d'] = df['close'].pct_change().rolling(window=period).std()

# 기타
df['hl_ratio'] = (df['high'] - df['low']) / df['close']
df['hl_ratio_sma'] = df['hl_ratio'].rolling(window=20).mean()

# 모멘텀
for period in [10, 20, 30]:
    df[f'momentum_{period}d_ratio'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
    df[f'roc_{period}d'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100

# Stochastic
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

# ATR
df['tr1'] = df['high'] - df['low']
df['tr2'] = abs(df['high'] - df['close'].shift())
df['tr3'] = abs(df['low'] - df['close'].shift())
df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
df['atr'] = df['tr'].rolling(window=14).mean()
df['atr_ratio'] = df['atr'] / df['close']
df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1, inplace=True)

# 교차 신호
df['sma_cross_5_20'] = (df['sma5'] > df['sma20']).astype(int)
df['sma_cross_20_60'] = (df['sma20'] > df['sma60']).astype(int)
df['ema_cross_12_26'] = (df['ema12'] > df['ema26']).astype(int)

# 타겟
df['returns'] = df['close'].pct_change()
df['next_day_return'] = df['returns'].shift(-1)
df['target'] = (df['next_day_return'] > 0).astype(int)

# 벤치마크
df['benchmark_signal'] = (df['close'].shift(1) > df['sma30'].shift(1)).astype(int)
df['benchmark_returns'] = df['benchmark_signal'] * df['returns']
df['benchmark_equity'] = INITIAL_CAPITAL * (1 + df['benchmark_returns']).cumprod()

# Buy & Hold
df['bnh_equity'] = INITIAL_CAPITAL * (1 + df['returns']).cumprod()

feature_columns = [
    'close_sma5_ratio', 'close_sma10_ratio', 'close_sma20_ratio',
    'close_sma30_ratio', 'close_sma60_ratio', 'close_sma120_ratio',
    'close_ema12_ratio', 'close_ema26_ratio', 'close_ema50_ratio',
    'rsi', 'rsi_sma', 'macd', 'macd_diff',
    'bb20_position', 'bb20_width', 'bb50_position', 'bb50_width',
    'volume_ratio5', 'volume_ratio20', 'volume_ratio60',
    'price_change_1d', 'price_change_3d', 'price_change_5d',
    'price_change_10d', 'price_change_20d', 'price_change_60d',
    'volatility_5d', 'volatility_20d', 'volatility_60d',
    'momentum_10d_ratio', 'momentum_20d_ratio', 'momentum_30d_ratio',
    'roc_10d', 'roc_20d', 'roc_30d',
    'hl_ratio', 'hl_ratio_sma', 'stoch_k', 'stoch_d', 'stoch_diff',
    'williams_r', 'atr_ratio',
    'sma_cross_5_20', 'sma_cross_20_60', 'ema_cross_12_26'
]

df_clean = df.dropna().copy()
print(f"학습 가능한 데이터: {len(df_clean)}일")
print(f"피처: {len(feature_columns)}개\n")

# ===================================
# 2. 모델 학습 및 확률 예측
# ===================================
print("=" * 70)
print("모델 학습 및 확률 예측")
print("=" * 70)

TRAIN_PERIOD = 365
RETRAIN_PERIOD = 90

# 여러 모델 사용
models_dict = {
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric='logloss', verbosity=0
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=200, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_split=10,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    ),
}

# 각 모델의 확률 예측 저장
all_probabilities = {}

for model_name, model in models_dict.items():
    print(f"\n{model_name} 학습 중...")

    probabilities = pd.Series(index=df_clean.index, dtype=float)
    probabilities[:] = 0.5

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
        # 상승 확률 예측
        prob = model.predict_proba(X_test_scaled)[:, 1]
        probabilities.iloc[test_start:test_end] = prob

        n_splits += 1

        if n_splits % 5 == 0:
            print(f"  진행: {n_splits}번째 재학습")

    print(f"  완료: 총 {n_splits}번 재학습")
    all_probabilities[model_name] = probabilities

# 앙상블 확률 (평균)
df_clean['ml_prob_ensemble'] = pd.DataFrame(all_probabilities).mean(axis=1)

print("\n앙상블 확률 계산 완료")

# ===================================
# 3. 하이브리드 전략 구현
# ===================================
print("\n" + "=" * 70)
print("하이브리드 전략 구현")
print("=" * 70)

# SMA30 기본 신호
df_clean['sma30_signal_raw'] = (df_clean['close'] > df_clean['sma30']).astype(int)

strategies = {}

# 전략 1: ML 확률 > 0.55 & SMA30 상승
df_clean['hybrid1_signal'] = ((df_clean['ml_prob_ensemble'] > 0.55) &
                               (df_clean['sma30_signal_raw'] == 1)).astype(int)
df_clean['hybrid1_signal_shifted'] = df_clean['hybrid1_signal'].shift(1)
df_clean['hybrid1_returns'] = df_clean['hybrid1_signal_shifted'] * df_clean['returns']
df_clean['hybrid1_equity'] = INITIAL_CAPITAL * (1 + df_clean['hybrid1_returns']).cumprod()
strategies['Hybrid 1 (Prob>0.55 & SMA30)'] = ('hybrid1_returns', 'hybrid1_equity')

# 전략 2: ML 확률 > 0.6 & SMA30 상승
df_clean['hybrid2_signal'] = ((df_clean['ml_prob_ensemble'] > 0.6) &
                               (df_clean['sma30_signal_raw'] == 1)).astype(int)
df_clean['hybrid2_signal_shifted'] = df_clean['hybrid2_signal'].shift(1)
df_clean['hybrid2_returns'] = df_clean['hybrid2_signal_shifted'] * df_clean['returns']
df_clean['hybrid2_equity'] = INITIAL_CAPITAL * (1 + df_clean['hybrid2_returns']).cumprod()
strategies['Hybrid 2 (Prob>0.6 & SMA30)'] = ('hybrid2_returns', 'hybrid2_equity')

# 전략 3: ML 확률 > 0.65 OR SMA30 상승
df_clean['hybrid3_signal'] = ((df_clean['ml_prob_ensemble'] > 0.65) |
                               (df_clean['sma30_signal_raw'] == 1)).astype(int)
df_clean['hybrid3_signal_shifted'] = df_clean['hybrid3_signal'].shift(1)
df_clean['hybrid3_returns'] = df_clean['hybrid3_signal_shifted'] * df_clean['returns']
df_clean['hybrid3_equity'] = INITIAL_CAPITAL * (1 + df_clean['hybrid3_returns']).cumprod()
strategies['Hybrid 3 (Prob>0.65 OR SMA30)'] = ('hybrid3_returns', 'hybrid3_equity')

# 전략 4: ML 확률 > 0.52 & (SMA30 OR RSI<70)
df_clean['hybrid4_signal'] = ((df_clean['ml_prob_ensemble'] > 0.52) &
                               ((df_clean['sma30_signal_raw'] == 1) | (df_clean['rsi'] < 70))).astype(int)
df_clean['hybrid4_signal_shifted'] = df_clean['hybrid4_signal'].shift(1)
df_clean['hybrid4_returns'] = df_clean['hybrid4_signal_shifted'] * df_clean['returns']
df_clean['hybrid4_equity'] = INITIAL_CAPITAL * (1 + df_clean['hybrid4_returns']).cumprod()
strategies['Hybrid 4 (Prob>0.52 & (SMA30 OR RSI<70))'] = ('hybrid4_returns', 'hybrid4_equity')

# 전략 5: ML 확률 가중치 적용 (0.5 이상일 때만)
df_clean['hybrid5_signal'] = (df_clean['ml_prob_ensemble'] > 0.5).astype(int)
df_clean['hybrid5_signal_shifted'] = df_clean['hybrid5_signal'].shift(1)
df_clean['hybrid5_returns'] = df_clean['hybrid5_signal_shifted'] * df_clean['returns']
df_clean['hybrid5_equity'] = INITIAL_CAPITAL * (1 + df_clean['hybrid5_returns']).cumprod()
strategies['Hybrid 5 (ML Prob>0.5)'] = ('hybrid5_returns', 'hybrid5_equity')

# 전략 6: 동적 임계값 (확률 상위 50% 매수)
prob_median = df_clean['ml_prob_ensemble'].rolling(window=90).median()
df_clean['hybrid6_signal'] = (df_clean['ml_prob_ensemble'] > prob_median).astype(int)
df_clean['hybrid6_signal_shifted'] = df_clean['hybrid6_signal'].shift(1)
df_clean['hybrid6_returns'] = df_clean['hybrid6_signal_shifted'] * df_clean['returns']
df_clean['hybrid6_equity'] = INITIAL_CAPITAL * (1 + df_clean['hybrid6_returns']).cumprod()
strategies['Hybrid 6 (Dynamic Threshold)'] = ('hybrid6_returns', 'hybrid6_equity')

# 전략 7: ML + Multiple SMA (5, 20, 60)
df_clean['hybrid7_signal'] = ((df_clean['ml_prob_ensemble'] > 0.55) &
                               (df_clean['sma5'] > df_clean['sma20']) &
                               (df_clean['sma20'] > df_clean['sma60'])).astype(int)
df_clean['hybrid7_signal_shifted'] = df_clean['hybrid7_signal'].shift(1)
df_clean['hybrid7_returns'] = df_clean['hybrid7_signal_shifted'] * df_clean['returns']
df_clean['hybrid7_equity'] = INITIAL_CAPITAL * (1 + df_clean['hybrid7_returns']).cumprod()
strategies['Hybrid 7 (ML + Multi-SMA Trend)'] = ('hybrid7_returns', 'hybrid7_equity')

# 전략 8: 보수적 전략 (높은 확률 + 강한 추세)
df_clean['hybrid8_signal'] = ((df_clean['ml_prob_ensemble'] > 0.6) &
                               (df_clean['close'] > df_clean['sma30']) &
                               (df_clean['rsi'] < 70) &
                               (df_clean['macd_diff'] > 0)).astype(int)
df_clean['hybrid8_signal_shifted'] = df_clean['hybrid8_signal'].shift(1)
df_clean['hybrid8_returns'] = df_clean['hybrid8_signal_shifted'] * df_clean['returns']
df_clean['hybrid8_equity'] = INITIAL_CAPITAL * (1 + df_clean['hybrid8_returns']).cumprod()
strategies['Hybrid 8 (Conservative)'] = ('hybrid8_returns', 'hybrid8_equity')

print(f"총 {len(strategies)}개 하이브리드 전략 생성\n")

# ===================================
# 4. 성과 계산
# ===================================
print("=" * 70)
print("성과 계산")
print("=" * 70)

def calculate_metrics(returns_col, equity_col, name):
    returns = df_clean[returns_col].dropna()
    equity = df_clean[equity_col].dropna()

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

    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    win_rate = (returns > 0).sum() / (returns != 0).sum() if (returns != 0).sum() > 0 else 0
    total_trades = (returns != 0).sum()

    return {
        'Strategy': name,
        'Total Return': total_return,
        'Total Return (x)': f"{total_return:.2f}x",
        'CAGR': cagr,
        'CAGR (%)': f"{cagr:.2%}",
        'MDD': mdd,
        'MDD (%)': f"{mdd:.2%}",
        'Sharpe': sharpe,
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Win Rate': win_rate,
        'Win Rate (%)': f"{win_rate:.2%}",
        'Total Trades': int(total_trades),
        'equity': equity,
        'returns': returns
    }

results = []

for name, (ret_col, eq_col) in strategies.items():
    metrics = calculate_metrics(ret_col, eq_col, name)
    if metrics:
        results.append(metrics)

# 벤치마크 추가
benchmark = calculate_metrics('benchmark_returns', 'benchmark_equity', 'Benchmark (SMA30)')
results.append(benchmark)

# Buy & Hold 추가
bnh_returns = df_clean['returns']
bnh_equity = df_clean['bnh_equity']
bnh_total_return = bnh_equity.iloc[-1] / INITIAL_CAPITAL
bnh_years = (bnh_equity.index[-1] - bnh_equity.index[0]).days / 365.25
bnh_cagr = (bnh_equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / bnh_years) - 1
bnh_mdd = ((bnh_equity - bnh_equity.cummax()) / bnh_equity.cummax()).min()
bnh_sharpe = (bnh_returns.mean() / bnh_returns.std()) * np.sqrt(252)

results.append({
    'Strategy': 'Buy & Hold',
    'Total Return': bnh_total_return,
    'Total Return (x)': f"{bnh_total_return:.2f}x",
    'CAGR': bnh_cagr,
    'CAGR (%)': f"{bnh_cagr:.2%}",
    'MDD': bnh_mdd,
    'MDD (%)': f"{bnh_mdd:.2%}",
    'Sharpe': bnh_sharpe,
    'Sharpe Ratio': f"{bnh_sharpe:.2f}",
    'Win Rate': (bnh_returns > 0).sum() / len(bnh_returns),
    'Win Rate (%)': f"{((bnh_returns > 0).sum() / len(bnh_returns)):.2%}",
    'Total Trades': len(bnh_returns),
    'equity': bnh_equity,
    'returns': bnh_returns
})

# 정렬
results_sorted = sorted(results, key=lambda x: x['Total Return'], reverse=True)

# ===================================
# 5. 결과 출력
# ===================================
print("\n전체 전략 성과 비교 (Total Return 기준)\n")
print(f"{'순위':<5} {'전략':<40} {'수익률':<12} {'CAGR':<12} {'MDD':<12} {'Sharpe':<8}")
print("-" * 85)

for idx, r in enumerate(results_sorted, 1):
    print(f"{idx:<5} {r['Strategy']:<40} {r['Total Return (x)']:<12} "
          f"{r['CAGR (%)']:<12} {r['MDD (%)']:<12} {r['Sharpe Ratio']:<8}")

print("\n" + "=" * 70)
print("TOP 5 전략 상세")
print("=" * 70)

for idx, r in enumerate(results_sorted[:5], 1):
    benchmark_ratio = r['Total Return'] / benchmark['Total Return']
    print(f"\n{idx}. {r['Strategy']}")
    print("-" * 70)
    print(f"  Total Return      : {r['Total Return (x)']} (벤치마크 대비: {benchmark_ratio:.2f}x)")
    print(f"  CAGR              : {r['CAGR (%)']}")
    print(f"  MDD               : {r['MDD (%)']}")
    print(f"  Sharpe Ratio      : {r['Sharpe Ratio']}")
    print(f"  Win Rate          : {r['Win Rate (%)']}")
    print(f"  Total Trades      : {r['Total Trades']:,}")

# CSV 저장
performance_df = pd.DataFrame([{
    'Strategy': r['Strategy'],
    'Total Return': r['Total Return (x)'],
    'CAGR': r['CAGR (%)'],
    'MDD': r['MDD (%)'],
    'Sharpe Ratio': r['Sharpe Ratio'],
    'Win Rate': r['Win Rate (%)'],
    'Total Trades': r['Total Trades']
} for r in results_sorted])

performance_df.to_csv('output/ml_hybrid_performance.csv', index=False, encoding='utf-8-sig')
print("\n성과 저장: output/ml_hybrid_performance.csv")

# ===================================
# 6. 시각화
# ===================================
print("\n시각화 생성 중...")

# 상위 5개 전략
top_5 = results_sorted[:5]

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 2, height_ratios=[2, 1, 1, 1], hspace=0.35, wspace=0.3)

# Subplot 1: 누적 수익률
ax1 = fig.add_subplot(gs[0, :])
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

for idx, r in enumerate(top_5):
    ax1.plot(r['equity'].index, r['equity'], label=r['Strategy'],
             linewidth=2.5 if idx == 0 else 2, alpha=1.0 if idx == 0 else 0.7,
             color=colors[idx % len(colors)])

ax1.set_yscale('log')
ax1.set_ylabel('Cumulative Returns (KRW, log scale)', fontsize=12, fontweight='bold')
ax1.set_title('Hybrid ML Strategy Performance - Top 5', fontsize=16, fontweight='bold', pad=20)
ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')

best = top_5[0]
metrics_text = f'''Best Strategy: {best['Strategy']}
Total Return: {best['Total Return (x)']}
CAGR: {best['CAGR (%)']}
MDD: {best['MDD (%)']}
Sharpe: {best['Sharpe Ratio']}
vs Benchmark: {best['Total Return'] / benchmark['Total Return']:.2f}x'''

ax1.text(0.98, 0.97, metrics_text, transform=ax1.transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.85),
         family='monospace')

# Subplot 2: Drawdown
ax2 = fig.add_subplot(gs[1, :])

for idx, r in enumerate(top_5[:3]):
    equity = r['equity']
    dd = (equity - equity.cummax()) / equity.cummax() * 100
    ax2.plot(equity.index, dd, label=r['Strategy'], linewidth=2,
             alpha=0.7, color=colors[idx])

ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.axhline(y=0, color='black', linewidth=0.8)
ax2.set_title('Drawdown Comparison (Top 3)', fontsize=12, fontweight='bold')

# Subplot 3: Total Return 비교
ax3 = fig.add_subplot(gs[2, 0])
strategies_short = [r['Strategy'][:35] for r in top_5]
returns_val = [r['Total Return'] for r in top_5]

bars = ax3.barh(strategies_short, returns_val, color=colors[:len(top_5)])
ax3.set_xlabel('Total Return (x)', fontsize=11, fontweight='bold')
ax3.set_title('Total Return Comparison', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

for bar, val in zip(bars, returns_val):
    ax3.text(val, bar.get_y() + bar.get_height()/2, f'{val:.2f}x',
             va='center', ha='left', fontsize=9, fontweight='bold')

# Subplot 4: Sharpe Ratio 비교
ax4 = fig.add_subplot(gs[2, 1])
sharpe_val = [r['Sharpe'] for r in top_5]

bars = ax4.barh(strategies_short, sharpe_val, color=colors[:len(top_5)])
ax4.set_xlabel('Sharpe Ratio', fontsize=11, fontweight='bold')
ax4.set_title('Sharpe Ratio Comparison', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

for bar, val in zip(bars, sharpe_val):
    ax4.text(val, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
             va='center', ha='left', fontsize=9, fontweight='bold')

# Subplot 5: Win Rate 비교
ax5 = fig.add_subplot(gs[3, 0])
win_rate_val = [r['Win Rate'] * 100 for r in top_5]

bars = ax5.barh(strategies_short, win_rate_val, color=colors[:len(top_5)])
ax5.set_xlabel('Win Rate (%)', fontsize=11, fontweight='bold')
ax5.set_title('Win Rate Comparison', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')

for bar, val in zip(bars, win_rate_val):
    ax5.text(val, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
             va='center', ha='left', fontsize=9, fontweight='bold')

# Subplot 6: MDD 비교
ax6 = fig.add_subplot(gs[3, 1])
mdd_val = [abs(r['MDD']) * 100 for r in top_5]

bars = ax6.barh(strategies_short, mdd_val, color=colors[:len(top_5)])
ax6.set_xlabel('Maximum Drawdown (%)', fontsize=11, fontweight='bold')
ax6.set_title('MDD Comparison (Lower is Better)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='x')
ax6.invert_xaxis()  # 낮을수록 좋으므로 반전

for bar, val in zip(bars, mdd_val):
    ax6.text(val, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
             va='center', ha='right', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('output/ml_hybrid_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("시각화 완료: output/ml_hybrid_results.png")

print("\n" + "=" * 70)
print("하이브리드 전략 백테스트 완료!")
print("=" * 70)
print(f"\n🏆 최고 성과: {best['Strategy']}")
print(f"   벤치마크 대비: {best['Total Return'] / benchmark['Total Return']:.2f}x")
if best['Total Return'] > benchmark['Total Return']:
    print(f"   ✅ 벤치마크 능가 성공!")
else:
    print(f"   ⚠️  벤치마크 미달 (추가 최적화 필요)")
print("=" * 70)
