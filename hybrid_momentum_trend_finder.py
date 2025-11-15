"""
Hybrid Momentum + Multi-Timeframe Trend Strategy
=================================================

목표:
- 벤치마크(252.03x) 초과
- MDD < 60%

전략:
- 멀티타임프레임 추세 확인
- 강한 모멘텀 확인
- 변동성 기반 필터
- 둘 다 충족 시에만 진입
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('chart_day/BTC_KRW.parquet')
print(f"데이터: {df.index.min()} ~ {df.index.max()} ({len(df)}일)\n")

INITIAL_CAPITAL = 1
SLIPPAGE = 0.002

print("지표 계산 중...")

# Moving Averages
for p in range(5, 121):
    df[f'sma{p}'] = df['close'].rolling(window=p).mean()
    df[f'ema{p}'] = df['close'].ewm(span=p, adjust=False).mean()

# Momentum & ROC
for p in [5, 10, 12, 15, 18, 20, 25, 30]:
    df[f'mom{p}'] = df['close'] - df['close'].shift(p)
    df[f'roc{p}'] = (df['close'] - df['close'].shift(p)) / df['close'].shift(p) * 100

# ATR (변동성 측정, 레버리지 아님)
df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    )
)
df['atr14'] = df['tr'].rolling(window=14).mean()
df['atr20'] = df['tr'].rolling(window=20).mean()

# Volatility
df['returns'] = df['close'].pct_change()
df['vol20'] = df['returns'].rolling(window=20).std() * np.sqrt(365)

# 벤치마크
df['benchmark_signal'] = (df['close'] > df['sma30']).astype(int)

print("지표 계산 완료\n")

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

print("="*100)
print("Hybrid Momentum + Trend 전략 테스트")
print("="*100)

# 1. Trend + Strong Momentum
print("\n1. Trend + Strong Momentum...")
for trend_ma in [25, 28, 30, 32, 35]:
    for mom_period in [10, 12, 15, 18, 20]:
        for mom_threshold in [0, 2, 3, 5]:
            sig = ((df['close'] > df[f'sma{trend_ma}']) &
                   (df[f'roc{mom_period}'] > mom_threshold)).astype(int)
            r = backtest(sig, f'Trend_SMA{trend_ma}+ROC{mom_period}>{mom_threshold}%')
            if r: results.append(r)

# 2. Multi-TF + Momentum
print("2. Multi-Timeframe + Momentum...")
tf_combos = [(10, 25, 50), (10, 28, 60), (15, 30, 60), (12, 30, 50)]
for p1, p2, p3 in tf_combos:
    for mom_p in [10, 15, 20]:
        for mom_th in [0, 2, 5]:
            sig = ((df[f'sma{p1}'] > df[f'sma{p2}']) &
                   (df[f'sma{p2}'] > df[f'sma{p3}']) &
                   (df[f'roc{mom_p}'] > mom_th)).astype(int)
            r = backtest(sig, f'MTF_{p1}_{p2}_{p3}+ROC{mom_p}>{mom_th}%')
            if r: results.append(r)

# 3. EMA + SMA + Momentum
print("3. EMA + SMA + Momentum...")
for ema_p in [8, 10, 12, 15]:
    for sma_p in [25, 28, 30, 32]:
        for mom_p in [10, 15, 20]:
            if ema_p < sma_p:
                sig = ((df[f'ema{ema_p}'] > df[f'sma{sma_p}']) &
                       (df[f'roc{mom_p}'] > 0)).astype(int)
                r = backtest(sig, f'EMA{ema_p}>SMA{sma_p}+ROC{mom_p}>0')
                if r: results.append(r)

# 4. Price Position + Momentum
print("4. Price Position + Momentum...")
for ma_p in [25, 28, 30, 32, 35]:
    for price_th in [0, 1, 2]:  # % above MA
        for mom_p in [10, 15, 20]:
            sig = (((df['close'] - df[f'sma{ma_p}']) / df[f'sma{ma_p}'] * 100 > price_th) &
                   (df[f'roc{mom_p}'] > 0)).astype(int)
            r = backtest(sig, f'Price>{price_th}%_SMA{ma_p}+ROC{mom_p}>0')
            if r: results.append(r)

# 5. Dual Momentum
print("5. Dual Momentum (Short + Long)...")
for short_mom in [5, 10, 15]:
    for long_mom in [20, 25, 30]:
        for trend_ma in [28, 30, 32]:
            if short_mom < long_mom:
                sig = ((df[f'roc{short_mom}'] > 0) &
                       (df[f'roc{long_mom}'] > 0) &
                       (df['close'] > df[f'sma{trend_ma}'])).astype(int)
                r = backtest(sig, f'DualMom_ROC{short_mom}&{long_mom}+SMA{trend_ma}')
                if r: results.append(r)

# 6. Accelerating Momentum
print("6. Accelerating Momentum + Trend...")
for ma_p in [25, 28, 30, 32]:
    for mom_p in [10, 15]:
        # Momentum이 가속 중 (현재 mom > 과거 mom)
        sig = ((df['close'] > df[f'sma{ma_p}']) &
               (df[f'mom{mom_p}'] > 0) &
               (df[f'mom{mom_p}'] > df[f'mom{mom_p}'].shift(5))).astype(int)
        r = backtest(sig, f'AccelMom_SMA{ma_p}+Mom{mom_p}↑')
        if r: results.append(r)

# 7. Low Volatility + Strong Trend
print("7. Low Volatility + Strong Trend...")
for ma_p in [25, 28, 30]:
    for vol_percentile in [30, 40, 50]:
        vol_threshold = df['vol20'].quantile(vol_percentile / 100)
        sig = ((df['close'] > df[f'sma{ma_p}']) &
               (df['vol20'] < vol_threshold) &
               (df['roc20'] > 0)).astype(int)
        r = backtest(sig, f'LowVol{vol_percentile}%_SMA{ma_p}+ROC20>0')
        if r: results.append(r)

# 8. MA Slope + Price Trend
print("8. MA Slope + Price Trend...")
for ma_p in [25, 28, 30, 32]:
    # MA 자체도 상승 중
    sig = ((df['close'] > df[f'sma{ma_p}']) &
           (df[f'sma{ma_p}'] > df[f'sma{ma_p}'].shift(5)) &
           (df['roc15'] > 0)).astype(int)
    r = backtest(sig, f'MA_Slope_SMA{ma_p}↑+ROC15>0')
    if r: results.append(r)

# 9. 변형된 벤치마크 (파라미터 최적화)
print("9. Optimized SMA...")
for ma_p in range(20, 41):
    sig = (df['close'] > df[f'sma{ma_p}']).astype(int)
    r = backtest(sig, f'Price>SMA{ma_p}')
    if r: results.append(r)

# 10. EMA 벤치마크
print("10. EMA Benchmark...")
for ema_p in range(20, 41):
    sig = (df['close'] > df[f'ema{ema_p}']).astype(int)
    r = backtest(sig, f'Price>EMA{ema_p}')
    if r: results.append(r)

print(f"\n총 {len(results)}개 전략 테스트 완료\n")

# 정렬 및 필터링
results.sort(key=lambda x: x['tr'], reverse=True)
acceptable = [r for r in results if r['mdd'] > -60]
winners = [r for r in acceptable if r['tr'] > r['br']]

print("="*115)
print(f"{'Rank':<6} {'Strategy':<50} {'Return':>10} {'Bench':>10} {'Gap':>8} {'CAGR':>8} {'MDD':>8} {'Sharpe':>7}")
print("="*115)

for i, r in enumerate(acceptable[:50], 1):
    gap = (r['tr'] / r['br'] - 1) * 100
    marker = "🏆" if r['tr'] > r['br'] else "  "
    print(f"{marker}{i:<5} {r['name']:<50} {r['tr']:>9.2f}x {r['br']:>9.2f}x {gap:>7.1f}% {r['cagr']:>7.2%} {r['mdd']:>7.1f}% {r['sharpe']:>7.2f}")

print("="*115)
print(f"\nMDD < 60% 전략: {len(acceptable)}개")
print(f"벤치마크 초과 (MDD < 60%): {len(winners)}개")
print("="*115)

if winners:
    best = winners[0]
    print(f"\n🎉 최고 전략: {best['name']}")
    print(f"   Total Return: {best['tr']:.2f}x")
    print(f"   Benchmark: {best['br']:.2f}x")
    print(f"   Outperformance: +{(best['tr']/best['br']-1)*100:.2f}%")
    print(f"   CAGR: {best['cagr']:.2%}")
    print(f"   MDD: {best['mdd']:.2%}")
    print(f"   Sharpe: {best['sharpe']:.2f}")

    import pandas as pd
    pd.DataFrame([{
        'Rank': i+1,
        'Strategy': r['name'],
        'Return_x': r['tr'],
        'Benchmark_x': r['br'],
        'Gap_%': (r['tr']/r['br']-1)*100,
        'CAGR_%': r['cagr']*100,
        'MDD_%': r['mdd'],
        'Sharpe': r['sharpe']
    } for i, r in enumerate(acceptable)]).to_csv('output/hybrid_strategy_results.csv', index=False)
    print(f"\n결과 저장: output/hybrid_strategy_results.csv")

else:
    print(f"\n⚠️  MDD < 60% 조건으로 벤치마크 초과 전략 없음")
    if acceptable:
        best = acceptable[0]
        print(f"   최고 전략: {best['name']}")
        print(f"   Return: {best['tr']:.2f}x (vs Benchmark: {best['br']:.2f}x)")
        print(f"   Gap: {(best['tr']/best['br']-1)*100:+.1f}%")
        print(f"   MDD: {best['mdd']:.2%}")

        # 상위 전략들의 gap 분석
        print(f"\n상위 10개 전략의 벤치마크 대비 성과:")
        for i, r in enumerate(acceptable[:10], 1):
            gap = (r['tr'] / r['br'] - 1) * 100
            print(f"   {i}. {r['name']:45s} {gap:+7.1f}%")

print("\n테스트 완료!")
