"""
Volatility-Adjusted & Advanced Positioning Strategies
변동성 기반 포지션 크기 조정으로 벤치마크 돌파 시도
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('chart_day/BTC_KRW.parquet')
print(f"데이터: {df.index.min()} ~ {df.index.max()} ({len(df)}일)\n")

INITIAL_CAPITAL = 1
SLIPPAGE = 0.002

# 지표 계산
print("지표 계산 중...")
for p in [10, 15, 20, 25, 30, 35, 40, 50, 60]:
    df[f'sma{p}'] = df['close'].rolling(window=p).mean()
    df[f'ema{p}'] = df['close'].ewm(span=p, adjust=False).mean()

for p in [5, 10, 15, 20, 25, 30]:
    df[f'mom{p}'] = df['close'] - df['close'].shift(p)
    df[f'roc{p}'] = (df['close'] - df['close'].shift(p)) / df['close'].shift(p) * 100

# ATR
df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    )
)
for p in [10, 14, 20]:
    df[f'atr{p}'] = df['tr'].rolling(window=p).mean()

# Volatility
df['returns'] = df['close'].pct_change()
for p in [10, 20, 30]:
    df[f'vol{p}'] = df['returns'].rolling(window=p).std() * np.sqrt(365)

# 벤치마크
df['benchmark_signal'] = (df['close'] > df['sma30']).astype(int)

def backtest_variable_position(signal, position_size, name):
    """가변 포지션 크기로 백테스트"""
    d = df.copy()
    d['sig'] = signal
    d['position'] = d['sig'] * position_size  # 0 ~ position_size 범위
    d['position'] = d['position'].clip(0, 2)  # 최대 2x 레버리지
    d['pos_chg'] = d['position'].diff()
    d['ret'] = d['close'].pct_change()

    # 포지션 크기에 비례한 수익
    d['strat_ret'] = d['position'].shift(1) * d['ret'] - abs(d['pos_chg']) * SLIPPAGE
    d['strat_eq'] = INITIAL_CAPITAL * (1 + d['strat_ret']).cumprod()

    # 벤치마크 (고정 1x)
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

print("전략 테스트 시작...\n")

# 1. 역변동성 가중 (volatility targeting)
for vol_target in [0.2, 0.3, 0.4, 0.5]:
    for ma_period in [25, 30, 35]:
        signal = (df['close'] > df[f'sma{ma_period}']).astype(int)
        position = vol_target / df['vol20'].clip(0.05, 1.0)  # 변동성이 낮을 때 더 큰 포지션
        r = backtest_variable_position(signal, position, f'VolTarget{vol_target}_SMA{ma_period}')
        if r: results.append(r)

# 2. ATR 기반 포지션 크기
for atr_mult in [0.5, 1.0, 1.5, 2.0]:
    for ma_period in [25, 30, 35]:
        signal = (df['close'] > df[f'sma{ma_period}']).astype(int)
        position = 1 + (df[f'atr14'] / df['close'] * 100 * atr_mult)  # ATR이 클수록 더 큰 포지션
        r = backtest_variable_position(signal, position, f'ATR{atr_mult}x_SMA{ma_period}')
        if r: results.append(r)

# 3. Momentum 강도 기반 포지션
for mom_period in [15, 20, 25]:
    for ma_period in [25, 30, 35]:
        signal = (df['close'] > df[f'sma{ma_period}']).astype(int)
        # ROC이 클수록 더 큰 포지션 (최대 2x)
        position = 1 + (df[f'roc{mom_period}'].clip(-20, 50) / 50)
        r = backtest_variable_position(signal, position, f'MomScale_ROC{mom_period}_SMA{ma_period}')
        if r: results.append(r)

# 4. Trend 강도 기반
for ma_period in [25, 30, 35]:
    signal = (df['close'] > df[f'sma{ma_period}']).astype(int)
    # 가격이 MA 위에 많이 있을수록 큰 포지션
    trend_strength = ((df['close'] - df[f'sma{ma_period}']) / df[f'sma{ma_period}'] * 100).clip(0, 20)
    position = 1 + (trend_strength / 20)  # 최대 2x
    r = backtest_variable_position(signal, position, f'TrendStrength_SMA{ma_period}')
    if r: results.append(r)

# 5. 복합 조건 (Trend + Momentum + Volatility)
for ma_p in [25, 30, 35]:
    for mom_p in [15, 20]:
        signal = ((df['close'] > df[f'sma{ma_p}']) & (df[f'mom{mom_p}'] > 0)).astype(int)
        # 변동성 낮고, momentum 강할 때 큰 포지션
        vol_adj = 0.3 / df['vol20'].clip(0.05, 1.0)
        mom_adj = 1 + (df[f'roc{mom_p}'].clip(0, 30) / 30)
        position = vol_adj * mom_adj
        r = backtest_variable_position(signal, position, f'Complex_SMA{ma_p}_Mom{mom_p}')
        if r: results.append(r)

# 6. Kelly Criterion 기반
for ma_period in [25, 30, 35]:
    signal = (df['close'] > df[f'sma{ma_period}']).astype(int)
    # 승률과 평균 수익을 추정하여 kelly fraction 계산
    win_rate = 0.55  # 가정
    avg_win = 0.02  # 가정
    avg_loss = 0.015  # 가정
    kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    position = pd.Series(kelly * 2, index=df.index)  # kelly * 2 (절반 kelly)
    r = backtest_variable_position(signal, position, f'Kelly_SMA{ma_period}')
    if r: results.append(r)

# 7. 고정 레버리지
for leverage in [1.2, 1.5, 1.8, 2.0]:
    for ma_period in [25, 30, 35]:
        signal = (df['close'] > df[f'sma{ma_period}']).astype(int)
        position = pd.Series(leverage, index=df.index)
        r = backtest_variable_position(signal, position, f'Leverage{leverage}x_SMA{ma_period}')
        if r: results.append(r)

# 8. 조건부 레버리지 (강한 트렌드일 때만)
for ma_period in [25, 30, 35]:
    for threshold in [3, 5, 7]:
        signal = (df['close'] > df[f'sma{ma_period}']).astype(int)
        # ROC > threshold일 때만 1.5x 레버리지
        strong_trend = (df['roc20'] > threshold).astype(float) * 0.5 + 1
        position = signal * strong_trend
        r = backtest_variable_position(signal, position, f'CondLev_ROC>{threshold}%_SMA{ma_period}')
        if r: results.append(r)

print(f"총 {len(results)}개 전략 테스트 완료\n")

results.sort(key=lambda x: x['tr'], reverse=True)

print("="*115)
print(f"{'Rank':<6} {'Strategy':<45} {'Return':>10} {'Benchmark':>10} {'Gap':>8} {'CAGR':>8} {'MDD':>8} {'Sharpe':>7}")
print("="*115)

winners = []
for i, r in enumerate(results[:40], 1):
    gap = (r['tr'] / r['br'] - 1) * 100
    marker = "🏆" if r['tr'] > r['br'] else "  "
    print(f"{marker}{i:<5} {r['name']:<45} {r['tr']:>9.2f}x {r['br']:>9.2f}x {gap:>7.1f}% {r['cagr']:>7.2%} {r['mdd']:>7.1f}% {r['sharpe']:>7.2f}")
    if r['tr'] > r['br']:
        winners.append(r)

print("="*115)
print(f"\n승리 전략: {len(winners)}개\n")

if winners:
    best = winners[0]
    print(f"🎉 최고 전략: {best['name']}")
    print(f"   Return: {best['tr']:.2f}x (벤치마크: {best['br']:.2f}x)")
    print(f"   Outperformance: +{(best['tr']/best['br']-1)*100:.2f}%")
    print(f"   CAGR: {best['cagr']:.2%}")
    print(f"   MDD: {best['mdd']:.2%}")
    print(f"   Sharpe: {best['sharpe']:.2f}")

    pd.DataFrame(results).to_csv('output/volatility_adjusted_results.csv', index=False)
    print(f"\n결과 저장: output/volatility_adjusted_results.csv")
else:
    print(f"⚠️  벤치마크 {results[0]['br']:.2f}x를 이기는 전략 없음")
    print(f"   최고 전략: {results[0]['name']} = {results[0]['tr']:.2f}x")
    print(f"   부족: {(results[0]['br']/results[0]['tr']-1)*100:.1f}%")

print("\n테스트 완료!")
