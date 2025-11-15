"""
Balanced Strategy Finder
=========================

목표: 일관성(월별 중앙값)과 성장성(Total Return)의 균형

평가 지표:
1. Consistency Score = 월별 중앙값 × 양수 월 비율
2. Growth Score = Total Return × (1 - |MDD|/100)
3. Combined Score = √(Consistency Score × Growth Score)

이를 통해 일관성과 성장성을 모두 고려
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

# 모든 지표 계산
for p in range(5, 121):
    df[f'sma{p}'] = df['close'].rolling(window=p).mean()
    df[f'ema{p}'] = df['close'].ewm(span=p, adjust=False).mean()

for p in [5, 10, 15, 20, 25, 30]:
    df[f'roc{p}'] = (df['close'] - df['close'].shift(p)) / df['close'].shift(p) * 100

df['returns'] = df['close'].pct_change()
df['vol20'] = df['returns'].rolling(window=20).std() * np.sqrt(365)

df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(
    abs(df['high'] - df['close'].shift(1)),
    abs(df['low'] - df['close'].shift(1))
))
df['atr14'] = df['tr'].rolling(window=14).mean()
df['atr_pct'] = df['atr14'] / df['close'] * 100

delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
rs = gain / loss
df['rsi14'] = 100 - (100 / (1 + rs))

df['benchmark_signal'] = (df['close'] > df['sma30']).astype(int)

print("지표 계산 완료\n")

def backtest_with_balanced_metrics(signal, name):
    """균형 잡힌 지표로 평가"""
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

    # 월별 수익률
    d['month'] = d.index.to_period('M')
    monthly_rets = d.groupby('month')['strat_ret'].apply(lambda x: (1 + x).prod() - 1) * 100
    bench_monthly_rets = d.groupby('month')['bench_ret'].apply(lambda x: (1 + x).prod() - 1) * 100

    # 지표 계산
    median_monthly = monthly_rets.median()
    bench_median = bench_monthly_rets.median()
    positive_months_pct = (monthly_rets > 0).sum() / len(monthly_rets) * 100

    tr = d['strat_eq'].iloc[-1] / INITIAL_CAPITAL
    br = d['bench_eq'].iloc[-1] / INITIAL_CAPITAL

    years = (d.index[-1] - d.index[0]).days / 365.25
    cagr = (tr) ** (1 / years) - 1

    mx = d['strat_eq'].cummax()
    dd = (d['strat_eq'] - mx) / mx * 100
    mdd = dd.min()

    sharpe = (d['strat_ret'].mean() / d['strat_ret'].std()) * np.sqrt(365) if d['strat_ret'].std() > 0 else 0

    # 균형 점수 계산
    # Consistency Score: 중앙값이 높고 양수 월이 많을수록 좋음
    consistency_score = max(0, median_monthly / 100) * (positive_months_pct / 100) * 100

    # Growth Score: Total Return이 높고 MDD가 작을수록 좋음
    growth_score = tr * (1 - abs(mdd) / 100)

    # Combined Score: 기하평균 (균형)
    if consistency_score > 0 and growth_score > 0:
        combined_score = np.sqrt(consistency_score * growth_score)
    else:
        combined_score = 0

    return {
        'name': name,
        'median_monthly': median_monthly,
        'bench_median': bench_median,
        'positive_months_pct': positive_months_pct,
        'tr': tr,
        'br': br,
        'cagr': cagr,
        'mdd': mdd,
        'sharpe': sharpe,
        'consistency_score': consistency_score,
        'growth_score': growth_score,
        'combined_score': combined_score
    }

results = []

print("="*100)
print("균형 전략 테스트 (일관성 × 성장성)")
print("="*100)

# 1. 기본 SMA 전략들 (비교 기준)
print("\n1. Baseline SMA Strategies...")
for p in range(25, 36):
    sig = (df['close'] > df[f'sma{p}']).astype(int)
    r = backtest_with_balanced_metrics(sig, f'SMA{p}')
    if r: results.append(r)

# 2. RSI 필터 전략
print("2. RSI-Filtered Strategies...")
for ma_p in range(25, 36):
    for rsi_low in [30, 35, 40, 45]:
        for rsi_high in [70, 75, 80, 85]:
            if rsi_low < rsi_high:
                sig = (
                    (df['close'] > df[f'sma{ma_p}']) &
                    (df['rsi14'] > rsi_low) &
                    (df['rsi14'] < rsi_high)
                ).astype(int)
                r = backtest_with_balanced_metrics(sig, f'SMA{ma_p}_RSI{rsi_low}_{rsi_high}')
                if r: results.append(r)

# 3. 모멘텀 + 추세 조합
print("3. Momentum + Trend Combinations...")
for ma_p in range(28, 33):
    for roc_p in [10, 15, 20]:
        for roc_th in [0, 1, 2, 3]:
            sig = (
                (df['close'] > df[f'sma{ma_p}']) &
                (df[f'roc{roc_p}'] > roc_th)
            ).astype(int)
            r = backtest_with_balanced_metrics(sig, f'SMA{ma_p}_ROC{roc_p}>{roc_th}')
            if r: results.append(r)

# 4. Dual MA + 모멘텀
print("4. Dual MA + Momentum...")
for short_ma in [20, 25, 30]:
    for long_ma in [40, 50, 60]:
        if short_ma < long_ma:
            sig = (
                (df['close'] > df[f'sma{short_ma}']) &
                (df[f'sma{short_ma}'] > df[f'sma{long_ma}']) &
                (df['roc20'] > 0)
            ).astype(int)
            r = backtest_with_balanced_metrics(sig, f'Dual_SMA{short_ma}_{long_ma}_ROC20')
            if r: results.append(r)

# 5. 변동성 필터
print("5. Volatility-Filtered Strategies...")
vol_percentiles = [50, 60, 70, 75, 80]
for ma_p in range(28, 33):
    for vol_pct in vol_percentiles:
        vol_threshold = df['vol20'].quantile(vol_pct / 100)
        sig = (
            (df['close'] > df[f'sma{ma_p}']) &
            (df['vol20'] < vol_threshold)
        ).astype(int)
        r = backtest_with_balanced_metrics(sig, f'SMA{ma_p}_LowVol{vol_pct}')
        if r: results.append(r)

# 6. 투표 시스템 (개선)
print("6. Voting Systems...")
for threshold in [3, 4, 5]:
    votes = (
        (df['close'] > df['sma25']).astype(int) +
        (df['close'] > df['sma30']).astype(int) +
        (df['close'] > df['sma35']).astype(int) +
        (df['roc15'] > 0).astype(int) +
        (df['roc20'] > 0).astype(int) +
        (df['sma25'] > df['sma50']).astype(int)
    )
    sig = (votes >= threshold).astype(int)
    r = backtest_with_balanced_metrics(sig, f'Vote6_{threshold}_of_6')
    if r: results.append(r)

# 7. 복합 필터 (RSI + 변동성)
print("7. Multi-Filter Strategies...")
for ma_p in [29, 30, 31]:
    sig = (
        (df['close'] > df[f'sma{ma_p}']) &
        (df['rsi14'] > 40) &
        (df['rsi14'] < 75) &
        (df['vol20'] < df['vol20'].quantile(0.7))
    ).astype(int)
    r = backtest_with_balanced_metrics(sig, f'MultiFilter_SMA{ma_p}_RSI_Vol')
    if r: results.append(r)

# 8. Adaptive 전략
print("8. Adaptive Strategies...")
vol_low = df['vol20'].quantile(0.33)
vol_high = df['vol20'].quantile(0.67)
for low_ma, mid_ma, high_ma in [(25, 30, 40), (25, 30, 50), (28, 31, 45)]:
    sig = np.where(
        df['vol20'] < vol_low,
        (df['close'] > df[f'sma{low_ma}']).astype(int),
        np.where(
            df['vol20'] > vol_high,
            (df['close'] > df[f'sma{high_ma}']).astype(int),
            (df['close'] > df[f'sma{mid_ma}']).astype(int)
        )
    )
    r = backtest_with_balanced_metrics(sig, f'Adaptive_{low_ma}_{mid_ma}_{high_ma}')
    if r: results.append(r)

print(f"\n총 {len(results)}개 전략 테스트 완료\n")

# 정렬: Combined Score 기준
results.sort(key=lambda x: x['combined_score'], reverse=True)
acceptable = [r for r in results if r['mdd'] > -60]
bench_winners = [r for r in acceptable if r['tr'] > r['br']]

print("="*140)
print(f"{'Rank':<6} {'Strategy':<40} {'Combined':>9} {'Median%':>8} {'WinMo%':>7} {'TotRet':>9} {'Bench':>9} {'MDD':>8} {'Sharpe':>7}")
print("="*140)

for i, r in enumerate(acceptable[:50], 1):
    marker = "🏆" if r['tr'] > r['br'] else "  "
    print(f"{marker}{i:<5} {r['name']:<40} {r['combined_score']:>8.2f} {r['median_monthly']:>7.2f}% {r['positive_months_pct']:>6.1f}% {r['tr']:>8.1f}x {r['br']:>8.1f}x {r['mdd']:>7.1f}% {r['sharpe']:>7.2f}")

print("="*140)
print(f"\nMDD < 60% 전략: {len(acceptable)}개")
print(f"벤치마크 초과 (Total Return): {len(bench_winners)}개")
print("="*140)

if bench_winners:
    best = bench_winners[0]
    print(f"\n🎉 최고 전략 (벤치마크 초과 + 최고 Combined Score): {best['name']}")
    print(f"   Combined Score: {best['combined_score']:.2f}")
    print(f"   월별 중앙값: {best['median_monthly']:.2f}% (벤치마크: {best['bench_median']:.2f}%)")
    print(f"   양수 월 비율: {best['positive_months_pct']:.1f}%")
    print(f"   Total Return: {best['tr']:.2f}x (벤치마크: {best['br']:.2f}x)")
    print(f"   Outperformance: {(best['tr']/best['br']-1)*100:+.2f}%")
    print(f"   CAGR: {best['cagr']:.2%}")
    print(f"   MDD: {best['mdd']:.2f}%")
    print(f"   Sharpe: {best['sharpe']:.2f}")

    pd.DataFrame([{
        'Rank': i+1,
        'Strategy': r['name'],
        'Combined_Score': r['combined_score'],
        'Median_Monthly_%': r['median_monthly'],
        'Positive_Months_%': r['positive_months_pct'],
        'Total_Return_x': r['tr'],
        'Benchmark_x': r['br'],
        'Outperformance_%': (r['tr']/r['br']-1)*100 if r['tr'] > r['br'] else -(r['br']/r['tr']-1)*100,
        'CAGR_%': r['cagr']*100,
        'MDD_%': r['mdd'],
        'Sharpe': r['sharpe']
    } for i, r in enumerate(acceptable)]).to_csv('output/balanced_strategy_results.csv', index=False)
    print(f"\n결과 저장: output/balanced_strategy_results.csv")

else:
    # Combined Score 최고 전략 (벤치마크 미달이라도)
    if acceptable:
        best = acceptable[0]
        print(f"\n⚠️  벤치마크 초과 전략 없음. Combined Score 최고:")
        print(f"   전략: {best['name']}")
        print(f"   Combined Score: {best['combined_score']:.2f}")
        print(f"   Total Return: {best['tr']:.2f}x (벤치마크: {best['br']:.2f}x)")
        print(f"   Gap: {(best['tr']/best['br']-1)*100:+.2f}%")
        print(f"   월별 중앙값: {best['median_monthly']:.2f}%")
        print(f"   MDD: {best['mdd']:.2f}%")

print("\n테스트 완료!")
