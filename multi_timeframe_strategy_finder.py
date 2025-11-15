"""
Multi-Timeframe Trend Following Strategy Finder
================================================

목표:
- 벤치마크(252.03x) 초과
- MDD < 60% (실용적인 수준)
- 멀티타임프레임 정렬로 안전한 진입

전략 원리:
- 단기, 중기, 장기 추세가 모두 정렬될 때만 진입
- False signal 감소 → MDD 감소
- 강한 추세에만 진입 → 수익 극대화
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('chart_day/BTC_KRW.parquet')
print(f"데이터: {df.index.min()} ~ {df.index.max()} ({len(df)}일)\n")

INITIAL_CAPITAL = 1
SLIPPAGE = 0.002

# 다양한 시간프레임 지표 계산
print("멀티타임프레임 지표 계산 중...")
for p in range(5, 201, 5):
    df[f'sma{p}'] = df['close'].rolling(window=p).mean()
    df[f'ema{p}'] = df['close'].ewm(span=p, adjust=False).mean()

# 벤치마크 (수정된 방식 - shift는 백테스트에서만)
df['sma30'] = df['close'].rolling(window=30).mean()
df['benchmark_signal'] = (df['close'] > df['sma30']).astype(int)

print("지표 계산 완료\n")

def backtest(signal, name):
    """백테스트 함수 - shift는 여기서만 한 번 적용"""
    d = df.copy()
    d['sig'] = signal
    d['pos_chg'] = d['sig'].diff()
    d['ret'] = d['close'].pct_change()

    # 전략 수익률 (shift(1) 한 번만)
    d['strat_ret'] = d['sig'].shift(1) * d['ret'] - abs(d['pos_chg']) * SLIPPAGE
    d['strat_eq'] = INITIAL_CAPITAL * (1 + d['strat_ret']).cumprod()

    # 벤치마크 수익률 (shift(1) 한 번만)
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

    # 승률 계산
    trades = d[d['pos_chg'] != 0].copy()
    if len(trades) > 2:
        trade_returns = []
        in_position = False
        entry_price = 0
        for idx, row in d.iterrows():
            if row['sig'] == 1 and not in_position:
                entry_price = row['close']
                in_position = True
            elif row['sig'] == 0 and in_position:
                exit_price = row['close']
                trade_return = (exit_price - entry_price) / entry_price
                trade_returns.append(trade_return)
                in_position = False

        if trade_returns:
            win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns) * 100
            num_trades = len(trade_returns)
        else:
            win_rate = 0
            num_trades = 0
    else:
        win_rate = 0
        num_trades = 0

    return {
        'name': name,
        'tr': tr,
        'br': br,
        'cagr': cagr,
        'mdd': mdd,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'num_trades': num_trades,
        'df': d
    }

results = []

print("="*100)
print("멀티타임프레임 전략 테스트 시작")
print("="*100)
print()

# 1. Dual Timeframe (2개 시간프레임 정렬)
print("1. Dual Timeframe 전략 테스트...")
short_periods = [5, 10, 15, 20, 25]
long_periods = [30, 40, 50, 60, 80, 100]

for short in short_periods:
    for long in long_periods:
        if short < long:
            # Price > Short MA AND Short MA > Long MA
            sig = ((df['close'] > df[f'sma{short}']) & (df[f'sma{short}'] > df[f'sma{long}'])).astype(int)
            r = backtest(sig, f'Dual_P>SMA{short}>SMA{long}')
            if r: results.append(r)

# 2. Triple Timeframe (3개 시간프레임 정렬)
print("2. Triple Timeframe 전략 테스트...")
combinations = [
    (10, 20, 50),
    (10, 30, 60),
    (10, 30, 100),
    (15, 30, 60),
    (15, 40, 80),
    (20, 40, 100),
    (20, 50, 100),
    (25, 50, 100),
]

for p1, p2, p3 in combinations:
    # All MAs aligned
    sig = ((df[f'sma{p1}'] > df[f'sma{p2}']) &
           (df[f'sma{p2}'] > df[f'sma{p3}'])).astype(int)
    r = backtest(sig, f'Triple_SMA{p1}>{p2}>{p3}')
    if r: results.append(r)

    # Price above all
    sig = ((df['close'] > df[f'sma{p1}']) &
           (df['close'] > df[f'sma{p2}']) &
           (df['close'] > df[f'sma{p3}'])).astype(int)
    r = backtest(sig, f'Triple_P>SMA{p1},{p2},{p3}')
    if r: results.append(r)

# 3. Quad Timeframe (4개 시간프레임 정렬)
print("3. Quad Timeframe 전략 테스트...")
quad_combinations = [
    (10, 20, 50, 100),
    (10, 30, 60, 100),
    (15, 30, 60, 100),
    (20, 40, 60, 100),
]

for p1, p2, p3, p4 in quad_combinations:
    sig = ((df[f'sma{p1}'] > df[f'sma{p2}']) &
           (df[f'sma{p2}'] > df[f'sma{p3}']) &
           (df[f'sma{p3}'] > df[f'sma{p4}'])).astype(int)
    r = backtest(sig, f'Quad_SMA{p1}>{p2}>{p3}>{p4}')
    if r: results.append(r)

# 4. EMA 기반 멀티타임프레임
print("4. EMA 기반 멀티타임프레임 테스트...")
for short in [10, 15, 20]:
    for mid in [30, 40, 50]:
        for long in [60, 80, 100]:
            if short < mid < long:
                sig = ((df[f'ema{short}'] > df[f'ema{mid}']) &
                       (df[f'ema{mid}'] > df[f'ema{long}'])).astype(int)
                r = backtest(sig, f'EMA_Triple_{short}>{mid}>{long}')
                if r: results.append(r)

# 5. Price Position 기반 (모든 MA 위에)
print("5. Price Position 기반 전략 테스트...")
ma_sets = [
    [10, 20, 30],
    [10, 20, 50],
    [10, 30, 50],
    [15, 30, 60],
    [20, 30, 50],
    [20, 40, 60],
]

for ma_set in ma_sets:
    condition = df['close'] > df[f'sma{ma_set[0]}']
    for ma in ma_set[1:]:
        condition = condition & (df['close'] > df[f'sma{ma}'])
    sig = condition.astype(int)
    r = backtest(sig, f'Price_Above_All_{ma_set}')
    if r: results.append(r)

# 6. Hybrid (SMA + EMA 조합)
print("6. Hybrid SMA+EMA 전략 테스트...")
for sma_p in [20, 30, 40]:
    for ema_p in [10, 15, 20]:
        if ema_p < sma_p:
            sig = ((df[f'ema{ema_p}'] > df[f'sma{sma_p}']) &
                   (df['close'] > df[f'ema{ema_p}'])).astype(int)
            r = backtest(sig, f'Hybrid_EMA{ema_p}>SMA{sma_p}')
            if r: results.append(r)

# 7. 강한 추세 필터 (모든 MA가 상승 중)
print("7. 강한 추세 필터 전략 테스트...")
for p1, p2 in [(20, 50), (30, 60), (20, 60)]:
    # MA도 상승 추세여야 함
    sig = ((df['close'] > df[f'sma{p1}']) &
           (df[f'sma{p1}'] > df[f'sma{p1}'].shift(5)) &  # MA도 상승 중
           (df[f'sma{p1}'] > df[f'sma{p2}'])).astype(int)
    r = backtest(sig, f'StrongTrend_SMA{p1}↑>SMA{p2}')
    if r: results.append(r)

print(f"\n총 {len(results)}개 전략 테스트 완료\n")

# 결과 정렬 및 필터링
results.sort(key=lambda x: x['tr'], reverse=True)

# MDD < 60% 필터
acceptable_mdd_results = [r for r in results if r['mdd'] > -60]
winners = [r for r in acceptable_mdd_results if r['tr'] > r['br']]

print("="*120)
print(f"{'Rank':<6} {'Strategy':<45} {'Return':>10} {'Bench':>10} {'Gap':>8} {'CAGR':>8} {'MDD':>8} {'Sharpe':>7} {'WinRate':>8} {'Trades':>7}")
print("="*120)

# 상위 40개 전략 (MDD < 60%)
for i, r in enumerate(acceptable_mdd_results[:40], 1):
    gap = (r['tr'] / r['br'] - 1) * 100
    marker = "🏆" if r['tr'] > r['br'] else "  "
    print(f"{marker}{i:<5} {r['name']:<45} {r['tr']:>9.2f}x {r['br']:>9.2f}x {gap:>7.1f}% {r['cagr']:>7.2%} {r['mdd']:>7.1f}% {r['sharpe']:>7.2f} {r['win_rate']:>7.1f}% {r['num_trades']:>7d}")

print("="*120)
print(f"\nMDD < 60% 전략: {len(acceptable_mdd_results)}개")
print(f"벤치마크 초과 (MDD < 60%): {len(winners)}개")
print("="*120)

if winners:
    best = winners[0]
    print(f"\n🎉 최고 전략: {best['name']}")
    print(f"   Total Return: {best['tr']:.2f}x (벤치마크: {best['br']:.2f}x)")
    print(f"   Outperformance: +{(best['tr']/best['br']-1)*100:.2f}%")
    print(f"   CAGR: {best['cagr']:.2%}")
    print(f"   MDD: {best['mdd']:.2%}")
    print(f"   Sharpe: {best['sharpe']:.2f}")
    print(f"   Win Rate: {best['win_rate']:.1f}%")
    print(f"   Trades: {best['num_trades']}")

    # 저장
    import pandas as pd
    summary_df = pd.DataFrame([{
        'Rank': i + 1,
        'Strategy': r['name'],
        'Return_x': r['tr'],
        'Benchmark_x': r['br'],
        'Outperformance_%': (r['tr']/r['br']-1)*100,
        'CAGR_%': r['cagr']*100,
        'MDD_%': r['mdd'],
        'Sharpe': r['sharpe'],
        'Win_Rate_%': r['win_rate'],
        'Num_Trades': r['num_trades']
    } for i, r in enumerate(acceptable_mdd_results)])

    summary_df.to_csv('output/multi_timeframe_results.csv', index=False)
    print(f"\n전체 결과 저장: output/multi_timeframe_results.csv")

    # 최고 전략 상세 저장
    best_data = {
        'strategy': best['name'],
        'return': best['tr'],
        'benchmark': best['br'],
        'cagr': best['cagr'],
        'mdd': best['mdd'],
        'sharpe': best['sharpe'],
        'win_rate': best['win_rate'],
        'num_trades': best['num_trades']
    }

    import json
    with open('output/best_multi_timeframe_strategy.json', 'w') as f:
        json.dump(best_data, f, indent=2)

    print("최고 전략 저장: output/best_multi_timeframe_strategy.json")

else:
    print(f"\n⚠️  MDD < 60% 조건으로 벤치마크를 이기는 전략 없음")
    if acceptable_mdd_results:
        print(f"   MDD < 60% 최고 전략: {acceptable_mdd_results[0]['name']}")
        print(f"   Return: {acceptable_mdd_results[0]['tr']:.2f}x (벤치마크: {acceptable_mdd_results[0]['br']:.2f}x)")
        print(f"   MDD: {acceptable_mdd_results[0]['mdd']:.2%}")
        print(f"   부족: {(acceptable_mdd_results[0]['br']/acceptable_mdd_results[0]['tr']-1)*100:.1f}%")
    else:
        print("   모든 전략이 MDD > 60%")

print("\n테스트 완료!")
