"""
Ultra Advanced Strategy Finder
목표: 벤치마크 252.03x 돌파
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('chart_day/BTC_KRW.parquet')
print(f"데이터: {df.index.min()} ~ {df.index.max()} ({len(df)}일)\n")

INITIAL_CAPITAL = 1
SLIPPAGE = 0.002

# 모든 지표 계산
print("지표 계산 중...")
for p in range(5, 201, 5):
    df[f'sma{p}'] = df['close'].rolling(window=p).mean()
    df[f'ema{p}'] = df['close'].ewm(span=p, adjust=False).mean()

for p in [5, 10, 15, 20, 25, 30]:
    df[f'mom{p}'] = df['close'] - df['close'].shift(p)
    df[f'roc{p}'] = (df['close'] - df['close'].shift(p)) / df['close'].shift(p) * 100

# Volatility
df['returns'] = df['close'].pct_change()
for p in [10, 20, 30]:
    df[f'vol{p}'] = df['returns'].rolling(window=p).std() * np.sqrt(365) * 100

# 벤치마크
df['benchmark_signal'] = (df['close'] > df['sma30']).astype(int)

def backtest(signal, name):
    d = df.copy()
    d['sig'] = signal
    d['pos_chg'] = d['sig'].diff()
    d['ret'] = d['close'].pct_change()

    d['strat_ret'] = d['sig'].shift(1) * d['ret'] - abs(d['pos_chg']) * SLIPPAGE
    d['strat_eq'] = INITIAL_CAPITAL * (1 + d['strat_ret']).cumprod()

    d['bench_pos_chg'] = d['benchmark_signal'].diff()
    d['bench_ret'] = d['benchmark_signal'].shift(1) * d['ret'] - abs(d['bench_pos_chg']) * SLIPPAGE
    d['bench_eq'] = INITIAL_CAPITAL * (1 + d['bench_ret']).cumprod()

    d = d.dropna()
    if len(d) == 0 or d['strat_eq'].iloc[-1] <= 0:
        return None

    tr = d['strat_eq'].iloc[-1] / INITIAL_CAPITAL
    br = d['bench_eq'].iloc[-1] / INITIAL_CAPITAL

    years = (d.index[-1] - d.index[0]).days / 365.25
    cagr = (d['strat_eq'].iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1

    mx = d['strat_eq'].cummax()
    dd = (d['strat_eq'] - mx) / mx * 100
    mdd = dd.min()

    sharpe = (d['strat_ret'].mean() / d['strat_ret'].std()) * np.sqrt(365) if d['strat_ret'].std() > 0 else 0

    return {'name': name, 'tr': tr, 'br': br, 'cagr': cagr, 'mdd': mdd, 'sharpe': sharpe}

results = []

print("전략 테스트 중...\n")

# 1. Price vs MA (모든 기간)
for p in range(5, 201, 5):
    r = backtest((df['close'] > df[f'sma{p}']).astype(int), f'Price>SMA{p}')
    if r: results.append(r)
    r = backtest((df['close'] > df[f'ema{p}']).astype(int), f'Price>EMA{p}')
    if r: results.append(r)

# 2. MA Cross (다양한 조합)
fast_periods = [5, 10, 15, 20, 25]
slow_periods = [30, 40, 50, 60, 80, 100, 120, 150, 200]
for fast in fast_periods:
    for slow in slow_periods:
        if fast < slow:
            r = backtest((df[f'sma{fast}'] > df[f'sma{slow}']).astype(int), f'SMA{fast}>{slow}')
            if r: results.append(r)
            r = backtest((df[f'ema{fast}'] > df[f'ema{slow}']).astype(int), f'EMA{fast}>{slow}')
            if r: results.append(r)

# 3. Momentum (다양한 기간)
for p in [5, 10, 15, 20, 25, 30]:
    r = backtest((df[f'mom{p}'] > 0).astype(int), f'Mom{p}')
    if r: results.append(r)
    r = backtest((df[f'roc{p}'] > 0).astype(int), f'ROC{p}')
    if r: results.append(r)

# 4. Trend + Momentum 조합
for trend_p in [20, 25, 30, 35, 40]:
    for mom_p in [10, 15, 20, 25]:
        r = backtest(
            ((df['close'] > df[f'sma{trend_p}']) & (df[f'mom{mom_p}'] > 0)).astype(int),
            f'Trend{trend_p}+Mom{mom_p}'
        )
        if r: results.append(r)

# 5. Double MA + Momentum
for ma1 in [15, 20, 25]:
    for ma2 in [30, 40, 50]:
        for mom_p in [10, 15, 20]:
            if ma1 < ma2:
                r = backtest(
                    ((df[f'sma{ma1}'] > df[f'sma{ma2}']) & (df[f'mom{mom_p}'] > 0)).astype(int),
                    f'SMA{ma1}>{ma2}+Mom{mom_p}'
                )
                if r: results.append(r)

# 6. Triple MA Alignment
for p1, p2, p3 in [(10, 20, 50), (10, 30, 60), (15, 30, 60), (20, 40, 80)]:
    r = backtest(
        ((df[f'sma{p1}'] > df[f'sma{p2}']) & (df[f'sma{p2}'] > df[f'sma{p3}']).astype(int)),
        f'3MA_{p1}_{p2}_{p3}'
    )
    if r: results.append(r)

# 7. Strong Momentum (threshold)
for mom_p in [10, 15, 20]:
    for thresh in [2, 3, 5, 7, 10]:
        r = backtest((df[f'roc{mom_p}'] > thresh).astype(int), f'ROC{mom_p}>={thresh}%')
        if r: results.append(r)

# 8. Trend + Strong Momentum
for trend_p in [25, 30, 35]:
    for mom_p in [15, 20]:
        for thresh in [3, 5, 7]:
            r = backtest(
                ((df['close'] > df[f'sma{trend_p}']) & (df[f'roc{mom_p}'] > thresh)).astype(int),
                f'Trend{trend_p}+ROC{mom_p}>{thresh}%'
            )
            if r: results.append(r)

# 9. Price Position (얼마나 MA위에 있는가)
for ma_p in [20, 25, 30, 35]:
    for pct in [0, 1, 2, 3, 5]:
        sig = ((df['close'] - df[f'sma{ma_p}']) / df[f'sma{ma_p}'] * 100 > pct).astype(int)
        r = backtest(sig, f'Price>{pct}%_above_SMA{ma_p}')
        if r: results.append(r)

# 10. Accelerating Momentum
for p1, p2 in [(5, 10), (10, 20), (15, 30)]:
    r = backtest((df[f'mom{p1}'] > df[f'mom{p2}']).astype(int), f'Mom{p1}>Mom{p2}')
    if r: results.append(r)

print(f"\n총 {len(results)}개 전략 테스트 완료\n")

results.sort(key=lambda x: x['tr'], reverse=True)

print("="*110)
print(f"{'Rank':<6} {'Strategy':<40} {'Return':>10} {'Benchmark':>10} {'Gap':>8} {'CAGR':>8} {'Sharpe':>7}")
print("="*110)

winners = []
for i, r in enumerate(results[:50], 1):
    gap = (r['tr'] / r['br'] - 1) * 100
    marker = "🏆" if r['tr'] > r['br'] else "  "
    print(f"{marker}{i:<5} {r['name']:<40} {r['tr']:>9.2f}x {r['br']:>9.2f}x {gap:>7.1f}% {r['cagr']:>7.2%} {r['sharpe']:>7.2f}")
    if r['tr'] > r['br']:
        winners.append(r)

print("="*110)
print(f"\n승자: {len(winners)}개")

if winners:
    best = winners[0]
    print(f"\n🎉 최고 전략: {best['name']}")
    print(f"   Return: {best['tr']:.2f}x (벤치마크: {best['br']:.2f}x)")
    print(f"   Outperformance: +{(best['tr']/best['br']-1)*100:.2f}%")
    print(f"   CAGR: {best['cagr']:.2%}")
    print(f"   MDD: {best['mdd']:.2%}")
    print(f"   Sharpe: {best['sharpe']:.2f}")
else:
    print(f"\n⚠️  벤치마크 {results[0]['br']:.2f}x를 이기는 전략 없음")
    print(f"   최고 전략: {results[0]['name']} = {results[0]['tr']:.2f}x")
    print(f"   부족: {(results[0]['br']/results[0]['tr']-1)*100:.1f}%")

# 저장
import pandas as pd
pd.DataFrame(results).to_csv('output/ultra_advanced_results.csv', index=False)
print(f"\n결과 저장: output/ultra_advanced_results.csv")
