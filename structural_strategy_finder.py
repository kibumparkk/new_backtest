"""
Structural Trend Following Strategy Finder
===========================================

목표:
- 구조적으로 다른 전략 (단순 MA 파라미터 변경 아님)
- 월별 수익률의 중앙값(median)으로 평가
- 일관성 있는 성과 (과거 한두 번의 큰 수익에 의존 X)

평가 지표:
1. 월별 수익률 중앙값 (Median Monthly Return)
2. 양수 월 비율 (Win Rate %)
3. 월별 수익률 표준편차 (Consistency)
4. MDD < 60%
5. Total Return (참고용)

전략 유형:
- Regime-based: 시장 상태에 따라 전략 변경
- Adaptive: 변동성에 따라 파라미터 조정
- Composite: 여러 시그널의 조합/투표
- Risk-adjusted: 리스크 기반 포지션 조정
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

# Momentum
for p in [10, 20, 30, 50]:
    df[f'roc{p}'] = (df['close'] - df['close'].shift(p)) / df['close'].shift(p) * 100

# Volatility
df['returns'] = df['close'].pct_change()
df['vol20'] = df['returns'].rolling(window=20).std() * np.sqrt(365)
df['vol50'] = df['returns'].rolling(window=50).std() * np.sqrt(365)

# ATR
df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    )
)
df['atr14'] = df['tr'].rolling(window=14).mean()
df['atr_pct'] = df['atr14'] / df['close'] * 100

# RSI
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
rs = gain / loss
df['rsi14'] = 100 - (100 / (1 + rs))

# 벤치마크
df['benchmark_signal'] = (df['close'] > df['sma30']).astype(int)

# 시장 Regime 분류
df['regime_vol'] = pd.cut(df['vol20'], bins=3, labels=['low_vol', 'med_vol', 'high_vol'])
df['regime_trend'] = np.where(df['sma20'] > df['sma50'], 'uptrend', 'downtrend')

print("지표 계산 완료\n")

def backtest_with_median_metrics(signal, name):
    """월별 수익률 중앙값 기반 백테스트"""
    d = df.copy()
    d['sig'] = signal
    d['pos_chg'] = d['sig'].diff()
    d['ret'] = d['close'].pct_change()

    # 전략 수익률
    d['strat_ret'] = d['sig'].shift(1) * d['ret'] - abs(d['pos_chg']) * SLIPPAGE
    d['strat_eq'] = INITIAL_CAPITAL * (1 + d['strat_ret']).cumprod()

    # 벤치마크 수익률
    d['bench_pos_chg'] = d['benchmark_signal'].diff()
    d['bench_ret'] = d['benchmark_signal'].shift(1) * d['ret'] - abs(d['bench_pos_chg']) * SLIPPAGE
    d['bench_eq'] = INITIAL_CAPITAL * (1 + d['bench_ret']).cumprod()

    d = d.dropna()
    if len(d) == 0 or d['strat_eq'].iloc[-1] <= 0:
        return None

    # 월별 수익률 계산
    d['month'] = d.index.to_period('M')
    monthly_rets = d.groupby('month')['strat_ret'].apply(lambda x: (1 + x).prod() - 1) * 100
    bench_monthly_rets = d.groupby('month')['bench_ret'].apply(lambda x: (1 + x).prod() - 1) * 100

    # 핵심 지표: 월별 수익률 중앙값
    median_monthly_return = monthly_rets.median()
    bench_median_monthly = bench_monthly_rets.median()

    # 일관성 지표
    positive_months_pct = (monthly_rets > 0).sum() / len(monthly_rets) * 100
    monthly_std = monthly_rets.std()

    # 기존 지표
    tr = d['strat_eq'].iloc[-1] / INITIAL_CAPITAL
    br = d['bench_eq'].iloc[-1] / INITIAL_CAPITAL

    years = (d.index[-1] - d.index[0]).days / 365.25
    cagr = (d['strat_eq'].iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1

    mx = d['strat_eq'].cummax()
    dd = (d['strat_eq'] - mx) / mx * 100
    mdd = dd.min()

    sharpe = (d['strat_ret'].mean() / d['strat_ret'].std()) * np.sqrt(365) if d['strat_ret'].std() > 0 else 0

    return {
        'name': name,
        'median_monthly': median_monthly_return,  # 핵심 지표
        'bench_median_monthly': bench_median_monthly,
        'positive_months_pct': positive_months_pct,
        'monthly_std': monthly_std,
        'tr': tr,
        'br': br,
        'cagr': cagr,
        'mdd': mdd,
        'sharpe': sharpe,
        'num_months': len(monthly_rets)
    }

results = []

print("="*100)
print("구조적 전략 테스트 (월별 수익률 중앙값 기반)")
print("="*100)

# 1. ADAPTIVE VOLATILITY STRATEGY
print("\n1. Adaptive Volatility Strategy...")
# 변동성이 낮을 때는 짧은 MA, 높을 때는 긴 MA
vol_low_threshold = df['vol20'].quantile(0.33)
vol_high_threshold = df['vol20'].quantile(0.67)

sig = np.where(
    df['vol20'] < vol_low_threshold,
    (df['close'] > df['sma20']).astype(int),  # 낮은 변동성: 빠른 반응
    np.where(
        df['vol20'] > vol_high_threshold,
        (df['close'] > df['sma50']).astype(int),  # 높은 변동성: 느린 반응
        (df['close'] > df['sma30']).astype(int)   # 중간 변동성: 표준
    )
)
r = backtest_with_median_metrics(sig, 'Adaptive_Vol_MA20_30_50')
if r: results.append(r)

# 다양한 조합
for low_ma, mid_ma, high_ma in [(15, 30, 60), (20, 35, 50), (25, 30, 40)]:
    sig = np.where(
        df['vol20'] < vol_low_threshold,
        (df['close'] > df[f'sma{low_ma}']).astype(int),
        np.where(
            df['vol20'] > vol_high_threshold,
            (df['close'] > df[f'sma{high_ma}']).astype(int),
            (df['close'] > df[f'sma{mid_ma}']).astype(int)
        )
    )
    r = backtest_with_median_metrics(sig, f'Adaptive_Vol_{low_ma}_{mid_ma}_{high_ma}')
    if r: results.append(r)

# 2. COMPOSITE VOTING STRATEGY
print("2. Composite Voting Strategy...")
# 여러 지표의 투표
for threshold in [2, 3, 4]:
    votes = (
        (df['close'] > df['sma20']).astype(int) +
        (df['close'] > df['sma30']).astype(int) +
        (df['close'] > df['sma50']).astype(int) +
        (df['roc20'] > 0).astype(int) +
        (df['rsi14'] > 50).astype(int)
    )
    sig = (votes >= threshold).astype(int)
    r = backtest_with_median_metrics(sig, f'Composite_Vote_{threshold}_of_5')
    if r: results.append(r)

# 3개 지표 투표
for threshold in [2, 3]:
    votes = (
        (df['close'] > df['sma30']).astype(int) +
        (df['roc20'] > 0).astype(int) +
        (df['sma20'] > df['sma50']).astype(int)
    )
    sig = (votes >= threshold).astype(int)
    r = backtest_with_median_metrics(sig, f'Vote3_{threshold}_of_3')
    if r: results.append(r)

# 3. REGIME SWITCHING STRATEGY
print("3. Regime Switching Strategy...")
# 상승장/하락장에 따라 다른 전략
sig = np.where(
    df['sma20'] > df['sma50'],  # 상승장
    (df['close'] > df['sma20']).astype(int),  # 공격적
    (df['close'] > df['sma50']).astype(int)   # 보수적
)
r = backtest_with_median_metrics(sig, 'Regime_Switch_Trend')
if r: results.append(r)

# 변동성 regime
sig = np.where(
    df['vol20'] < df['vol20'].quantile(0.5),  # 낮은 변동성
    (df['close'] > df['sma25']).astype(int),  # 적극적
    (df['close'] > df['sma40']).astype(int)   # 보수적
)
r = backtest_with_median_metrics(sig, 'Regime_Switch_Vol')
if r: results.append(r)

# 4. MULTI-CONFIRMATION STRATEGY
print("4. Multi-Confirmation Strategy...")
# 여러 조건 동시 충족
for ma_p in [25, 30, 35]:
    sig = (
        (df['close'] > df[f'sma{ma_p}']) &  # 추세
        (df['roc20'] > 0) &  # 모멘텀
        (df['sma20'] > df['sma50'])  # 중기 추세
    ).astype(int)
    r = backtest_with_median_metrics(sig, f'MultiConfirm_SMA{ma_p}_ROC_Trend')
    if r: results.append(r)

# 5. FILTERED TREND STRATEGY
print("5. Filtered Trend Strategy...")
# 기본 추세에 필터 추가
for ma_p in [28, 30, 32]:
    # RSI 필터
    sig = (
        (df['close'] > df[f'sma{ma_p}']) &
        (df['rsi14'] > 40) &  # 과매도 필터
        (df['rsi14'] < 80)    # 과매수 필터
    ).astype(int)
    r = backtest_with_median_metrics(sig, f'Filtered_SMA{ma_p}_RSI40_80')
    if r: results.append(r)

    # 변동성 필터
    sig = (
        (df['close'] > df[f'sma{ma_p}']) &
        (df['vol20'] < df['vol20'].quantile(0.75))  # 높은 변동성 회피
    ).astype(int)
    r = backtest_with_median_metrics(sig, f'Filtered_SMA{ma_p}_LowVol75')
    if r: results.append(r)

# 6. DUAL TIMEFRAME CONFIRMATION
print("6. Dual Timeframe Confirmation...")
for short_p, long_p in [(20, 50), (25, 60), (30, 80)]:
    sig = (
        (df['close'] > df[f'sma{short_p}']) &  # 단기
        (df[f'sma{short_p}'] > df[f'sma{long_p}'])  # 장기 추세 확인
    ).astype(int)
    r = backtest_with_median_metrics(sig, f'DualTF_SMA{short_p}_{long_p}')
    if r: results.append(r)

# 7. MOMENTUM STRENGTH STRATEGY
print("7. Momentum Strength Strategy...")
# 강한 모멘텀일 때만
for ma_p in [28, 30, 32]:
    for roc_threshold in [2, 3, 5]:
        sig = (
            (df['close'] > df[f'sma{ma_p}']) &
            (df['roc20'] > roc_threshold)
        ).astype(int)
        r = backtest_with_median_metrics(sig, f'MomStrength_SMA{ma_p}_ROC{roc_threshold}')
        if r: results.append(r)

# 8. VOLATILITY-ADJUSTED ENTRY
print("8. Volatility-Adjusted Entry...")
# ATR이 적정 범위일 때만
for ma_p in [28, 30, 32]:
    atr_low = df['atr_pct'].quantile(0.2)
    atr_high = df['atr_pct'].quantile(0.8)
    sig = (
        (df['close'] > df[f'sma{ma_p}']) &
        (df['atr_pct'] > atr_low) &  # 너무 낮은 변동성 회피
        (df['atr_pct'] < atr_high)   # 너무 높은 변동성 회피
    ).astype(int)
    r = backtest_with_median_metrics(sig, f'VolAdjust_SMA{ma_p}_ATR20_80')
    if r: results.append(r)

# 9. BREAKOUT CONFIRMATION
print("9. Breakout Confirmation...")
# 고점 돌파 + 추세 확인
for ma_p in [28, 30, 32]:
    high_20 = df['high'].rolling(window=20).max()
    sig = (
        (df['close'] > high_20.shift(1)) &  # 고점 돌파
        (df['close'] > df[f'sma{ma_p}'])    # 추세 확인
    ).astype(int)
    r = backtest_with_median_metrics(sig, f'Breakout_High20_SMA{ma_p}')
    if r: results.append(r)

# 10. MEAN REVERSION PROTECTION
print("10. Mean Reversion Protection...")
# 급등 후 회피
for ma_p in [28, 30, 32]:
    sig = (
        (df['close'] > df[f'sma{ma_p}']) &
        (df['roc10'] < 20)  # 급등 회피 (10일 ROC < 20%)
    ).astype(int)
    r = backtest_with_median_metrics(sig, f'MeanRevProtect_SMA{ma_p}_ROC10_20')
    if r: results.append(r)

print(f"\n총 {len(results)}개 전략 테스트 완료\n")

# 정렬: 월별 수익률 중앙값 기준
results.sort(key=lambda x: x['median_monthly'], reverse=True)
acceptable = [r for r in results if r['mdd'] > -60]
winners = [r for r in acceptable if r['median_monthly'] > r['bench_median_monthly']]

print("="*130)
print(f"{'Rank':<6} {'Strategy':<45} {'MedianMo%':>10} {'BenchMed%':>10} {'Gap':>8} {'WinMo%':>8} {'MoStd':>8} {'TotRet':>9} {'MDD':>8}")
print("="*130)

for i, r in enumerate(acceptable[:50], 1):
    gap = r['median_monthly'] - r['bench_median_monthly']
    marker = "🏆" if r['median_monthly'] > r['bench_median_monthly'] else "  "
    print(f"{marker}{i:<5} {r['name']:<45} {r['median_monthly']:>9.2f}% {r['bench_median_monthly']:>9.2f}% {gap:>7.2f}% {r['positive_months_pct']:>7.1f}% {r['monthly_std']:>7.2f}% {r['tr']:>8.1f}x {r['mdd']:>7.1f}%")

print("="*130)
print(f"\nMDD < 60% 전략: {len(acceptable)}개")
print(f"월별 중앙값 벤치마크 초과: {len(winners)}개")
print("="*130)

if winners:
    best = winners[0]
    print(f"\n🎉 최고 전략: {best['name']}")
    print(f"   월별 수익률 중앙값: {best['median_monthly']:.2f}%")
    print(f"   벤치마크 중앙값: {best['bench_median_monthly']:.2f}%")
    print(f"   중앙값 차이: {best['median_monthly'] - best['bench_median_monthly']:+.2f}%p")
    print(f"   양수 월 비율: {best['positive_months_pct']:.1f}%")
    print(f"   월간 표준편차: {best['monthly_std']:.2f}%")
    print(f"   Total Return: {best['tr']:.2f}x (벤치마크: {best['br']:.2f}x)")
    print(f"   MDD: {best['mdd']:.2f}%")
    print(f"   Sharpe: {best['sharpe']:.2f}")

    # 저장
    pd.DataFrame([{
        'Rank': i+1,
        'Strategy': r['name'],
        'Median_Monthly_%': r['median_monthly'],
        'Bench_Median_%': r['bench_median_monthly'],
        'Gap_%p': r['median_monthly'] - r['bench_median_monthly'],
        'Positive_Months_%': r['positive_months_pct'],
        'Monthly_Std_%': r['monthly_std'],
        'Total_Return_x': r['tr'],
        'Benchmark_x': r['br'],
        'MDD_%': r['mdd'],
        'Sharpe': r['sharpe']
    } for i, r in enumerate(acceptable)]).to_csv('output/structural_strategy_results.csv', index=False)
    print(f"\n결과 저장: output/structural_strategy_results.csv")

else:
    print(f"\n⚠️  월별 중앙값 기준 벤치마크 초과 전략 없음")
    if acceptable:
        best = acceptable[0]
        print(f"   최고 전략: {best['name']}")
        print(f"   월별 중앙값: {best['median_monthly']:.2f}% (벤치마크: {best['bench_median_monthly']:.2f}%)")
        print(f"   부족: {best['bench_median_monthly'] - best['median_monthly']:.2f}%p")

        # 상위 10개 분석
        print(f"\n상위 10개 전략 (월별 중앙값 기준):")
        for i, r in enumerate(acceptable[:10], 1):
            gap = r['median_monthly'] - r['bench_median_monthly']
            print(f"   {i}. {r['name']:45s} {r['median_monthly']:6.2f}% (벤치: {r['bench_median_monthly']:5.2f}%, Gap: {gap:+6.2f}%p)")

print("\n테스트 완료!")
