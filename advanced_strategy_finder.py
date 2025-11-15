"""
Advanced Trend Following Strategy Finder
벤치마크(252.03x)를 능가하는 전략 탐색
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

os.makedirs('output', exist_ok=True)

df = pd.read_parquet('chart_day/BTC_KRW.parquet')
print(f"데이터 기간: {df.index.min()} ~ {df.index.max()}")
print(f"총 {len(df)}일\n")

INITIAL_CAPITAL = 1
SLIPPAGE = 0.002

# 기술적 지표 계산
print("기술적 지표 계산 중...")
for period in [5, 10, 15, 20, 30, 50, 60, 100, 120, 200]:
    df[f'sma{period}'] = df['close'].rolling(window=period).mean()
    df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()

# Bollinger Bands (multiple periods)
for period in [10, 20, 30]:
    df[f'bb_mid_{period}'] = df['close'].rolling(window=period).mean()
    df[f'bb_std_{period}'] = df['close'].rolling(window=period).std()
    df[f'bb_upper_{period}'] = df[f'bb_mid_{period}'] + 2 * df[f'bb_std_{period}']
    df[f'bb_lower_{period}'] = df[f'bb_mid_{period}'] - 2 * df[f'bb_std_{period}']

# Donchian Channel
for period in [10, 20, 30, 55]:
    df[f'dc_high_{period}'] = df['high'].rolling(window=period).max()
    df[f'dc_low_{period}'] = df['low'].rolling(window=period).min()

# ATR
df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    )
)
for period in [7, 14, 20, 30]:
    df[f'atr{period}'] = df['tr'].rolling(window=period).mean()

# MACD variations
ema12 = df['close'].ewm(span=12, adjust=False).mean()
ema26 = df['close'].ewm(span=26, adjust=False).mean()
df['macd'] = ema12 - ema26
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
df['macd_hist'] = df['macd'] - df['macd_signal']

# RSI
def calc_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

for period in [7, 14, 21]:
    df[f'rsi{period}'] = calc_rsi(df['close'], period)

# Momentum & ROC
for period in [3, 5, 7, 10, 15, 20, 30]:
    df[f'momentum{period}'] = df['close'] - df['close'].shift(period)
    df[f'roc{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100

# 벤치마크
df['sma30'] = df['close'].rolling(window=30).mean()
df['benchmark_signal'] = (df['close'] > df['sma30']).astype(int)

print("기술적 지표 계산 완료\n")

def backtest_strategy(signals, name):
    df_bt = df.copy()
    df_bt['signal'] = signals
    df_bt['position_change'] = df_bt['signal'].diff()
    df_bt['daily_return'] = df_bt['close'].pct_change()

    # 전략 수익률
    df_bt['strategy_return'] = df_bt['signal'].shift(1) * df_bt['daily_return'] - abs(df_bt['position_change']) * SLIPPAGE
    df_bt['strategy_equity'] = INITIAL_CAPITAL * (1 + df_bt['strategy_return']).cumprod()

    # 벤치마크 수익률
    df_bt['benchmark_position_change'] = df_bt['benchmark_signal'].diff()
    df_bt['benchmark_return'] = df_bt['benchmark_signal'].shift(1) * df_bt['daily_return'] - abs(df_bt['benchmark_position_change']) * SLIPPAGE
    df_bt['benchmark_equity'] = INITIAL_CAPITAL * (1 + df_bt['benchmark_return']).cumprod()

    df_bt = df_bt.dropna()

    if len(df_bt) == 0 or df_bt['strategy_equity'].iloc[-1] <= 0:
        return None

    total_return = df_bt['strategy_equity'].iloc[-1] / INITIAL_CAPITAL
    benchmark_return = df_bt['benchmark_equity'].iloc[-1] / INITIAL_CAPITAL

    years = (df_bt.index[-1] - df_bt.index[0]).days / 365.25
    cagr = (df_bt['strategy_equity'].iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1

    running_max = df_bt['strategy_equity'].cummax()
    drawdown = (df_bt['strategy_equity'] - running_max) / running_max * 100
    mdd = drawdown.min()

    sharpe = (df_bt['strategy_return'].mean() / df_bt['strategy_return'].std()) * np.sqrt(365) if df_bt['strategy_return'].std() != 0 else 0

    return {
        'name': name,
        'total_return': total_return,
        'benchmark_return': benchmark_return,
        'cagr': cagr,
        'mdd': mdd,
        'sharpe': sharpe,
        'df': df_bt
    }

strategies = []

print("전략 생성 중...\n")

# 1. 다양한 MA Cross 전략
for fast in [5, 10, 15, 20]:
    for slow in [30, 50, 60, 100]:
        if fast < slow:
            strategies.append((
                (df[f'sma{fast}'] > df[f'sma{slow}']).astype(int),
                f'SMA_{fast}_{slow}'
            ))
            strategies.append((
                (df[f'ema{fast}'] > df[f'ema{slow}']).astype(int),
                f'EMA_{fast}_{slow}'
            ))

# 2. Price vs MA 전략
for period in [10, 20, 30, 50, 60, 100]:
    strategies.append((
        (df['close'] > df[f'sma{period}']).astype(int),
        f'Price_Above_SMA{period}'
    ))

# 3. Multiple MA Alignment
strategies.append((
    ((df['sma5'] > df['sma10']) & (df['sma10'] > df['sma20']) & (df['sma20'] > df['sma50'])).astype(int),
    'SMA_Cascade_5_10_20_50'
))

strategies.append((
    ((df['sma10'] > df['sma20']) & (df['sma20'] > df['sma50']) & (df['sma50'] > df['sma100'])).astype(int),
    'SMA_Cascade_10_20_50_100'
))

strategies.append((
    ((df['ema5'] > df['ema10']) & (df['ema10'] > df['ema20']) & (df['ema20'] > df['ema50'])).astype(int),
    'EMA_Cascade_5_10_20_50'
))

# 4. Donchian Breakout with various periods
for period in [10, 20, 30, 55]:
    strategies.append((
        (df['close'] > df[f'dc_high_{period}'].shift(1)).astype(int),
        f'Donchian_{period}_Breakout'
    ))

# 5. Momentum strategies
for period in [5, 10, 15, 20, 30]:
    strategies.append((
        (df[f'momentum{period}'] > 0).astype(int),
        f'Momentum_{period}'
    ))

# 6. ROC strategies
for period in [5, 10, 15, 20]:
    strategies.append((
        (df[f'roc{period}'] > 0).astype(int),
        f'ROC_{period}'
    ))

# 7. Trend + Momentum combinations
for ma_period in [20, 30, 50]:
    for mom_period in [10, 20]:
        strategies.append((
            ((df['close'] > df[f'sma{ma_period}']) & (df[f'momentum{mom_period}'] > 0)).astype(int),
            f'Trend_SMA{ma_period}_Mom{mom_period}'
        ))

# 8. Dual Momentum (absolute + relative)
for period in [10, 15, 20]:
    strategies.append((
        ((df[f'momentum{period}'] > 0) & (df['close'] > df[f'sma{period}'])).astype(int),
        f'DualMomentum_{period}'
    ))

# 9. MACD + Trend
strategies.append((
    ((df['macd'] > df['macd_signal']) & (df['close'] > df['sma50'])).astype(int),
    'MACD_Trend_50'
))

strategies.append((
    ((df['macd_hist'] > 0) & (df['close'] > df['sma30'])).astype(int),
    'MACD_Hist_Trend_30'
))

# 10. Price momentum with trend filter
strategies.append((
    ((df['roc20'] > 0) & (df['sma20'] > df['sma50'])).astype(int),
    'ROC20_TrendFilter'
))

# 11. Breakout with trend confirmation
strategies.append((
    ((df['close'] > df['dc_high_20'].shift(1)) & (df['sma20'] > df['sma50'])).astype(int),
    'Donchian20_TrendConfirm'
))

# 12. Multi-timeframe trend
strategies.append((
    ((df['close'] > df['sma10']) & (df['sma10'] > df['sma30']) & (df['sma30'] > df['sma100'])).astype(int),
    'MultiTF_10_30_100'
))

# 13. Aggressive momentum
for period in [3, 7]:
    strategies.append((
        (df[f'roc{period}'] > 0).astype(int),
        f'FastROC_{period}'
    ))

# 14. Strong momentum (threshold based)
for period in [10, 20]:
    for threshold in [0, 5, 10]:
        strategies.append((
            (df[f'roc{period}'] > threshold).astype(int),
            f'ROC{period}_Thresh{threshold}'
        ))

print(f"총 {len(strategies)}개 전략 생성 완료\n")
print("="*100)
print("백테스트 실행 중...")
print("="*100)

results = []
for i, (signal, name) in enumerate(strategies, 1):
    result = backtest_strategy(signal, name)
    if result is not None:
        results.append(result)
        beats = "✓" if result['total_return'] > result['benchmark_return'] else " "
        print(f"{i:3d}. [{beats}] {name:35s} | Return: {result['total_return']:8.2f}x | Benchmark: {result['benchmark_return']:8.2f}x | CAGR: {result['cagr']:7.2%} | Sharpe: {result['sharpe']:6.2f}")

results.sort(key=lambda x: x['total_return'], reverse=True)

print("\n" + "="*100)
print("상위 20개 전략")
print("="*100)
for i, r in enumerate(results[:20], 1):
    beats = "✓" if r['total_return'] > r['benchmark_return'] else " "
    print(f"{i:2d}. [{beats}] {r['name']:35s} | Return: {r['total_return']:8.2f}x | Benchmark: {r['benchmark_return']:8.2f}x | CAGR: {r['cagr']:7.2%} | Sharpe: {r['sharpe']:6.2f}")

winning = [r for r in results if r['total_return'] > r['benchmark_return']]

print("\n" + "="*100)
print(f"벤치마크를 이기는 전략: {len(winning)}개")
print("="*100)

if winning:
    best = winning[0]
    print(f"\n🏆 최고 성과 전략: {best['name']}")
    print(f"   Total Return: {best['total_return']:.2f}x")
    print(f"   Benchmark: {best['benchmark_return']:.2f}x")
    print(f"   Outperformance: {(best['total_return'] / best['benchmark_return'] - 1) * 100:.2f}%")
    print(f"   CAGR: {best['cagr']:.2%}")
    print(f"   MDD: {best['mdd']:.2%}")
    print(f"   Sharpe: {best['sharpe']:.2f}")

    # 저장
    all_df = pd.DataFrame([{
        'Rank': i + 1,
        'Strategy': r['name'],
        'Total_Return_x': r['total_return'],
        'Benchmark_x': r['benchmark_return'],
        'Beats_Benchmark': 'Yes' if r['total_return'] > r['benchmark_return'] else 'No',
        'CAGR_%': r['cagr'] * 100,
        'MDD_%': r['mdd'],
        'Sharpe': r['sharpe']
    } for i, r in enumerate(results)])
    all_df.to_csv('output/advanced_strategies_results.csv', index=False)
    print(f"\n전체 결과 저장: output/advanced_strategies_results.csv")
else:
    print("\n⚠️  벤치마크를 이기는 전략을 찾지 못했습니다!")
    print(f"   벤치마크 성과: {results[0]['benchmark_return']:.2f}x")
    print(f"   최고 전략: {results[0]['name']} = {results[0]['total_return']:.2f}x")
    print(f"   Gap: {(results[0]['benchmark_return'] / results[0]['total_return'] - 1) * 100:.1f}% 부족")

print("\n백테스트 완료!")
