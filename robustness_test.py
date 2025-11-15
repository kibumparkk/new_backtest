"""
Robustness Test: SMA30 vs SMA31
================================

목표: SMA31의 우위가 과적합인지 검증

테스트:
1. 시기별 성과 분석 (Walk-forward)
2. 통계적 유의성 검증 (t-test)
3. 파라미터 민감도 분석
4. 다른 기간 범위 테스트
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('chart_day/BTC_KRW.parquet')
print(f"전체 데이터: {df.index.min()} ~ {df.index.max()} ({len(df)}일)\n")

INITIAL_CAPITAL = 1
SLIPPAGE = 0.002

# 지표 계산
for p in range(25, 41):
    df[f'sma{p}'] = df['close'].rolling(window=p).mean()

def backtest_period(df_period, ma_period, name):
    """특정 기간 백테스트"""
    d = df_period.copy()

    signal = (d['close'] > d[f'sma{ma_period}']).astype(int)
    d['sig'] = signal
    d['pos_chg'] = d['sig'].diff()
    d['ret'] = d['close'].pct_change()

    d['strat_ret'] = d['sig'].shift(1) * d['ret'] - abs(d['pos_chg']) * SLIPPAGE
    d['strat_eq'] = INITIAL_CAPITAL * (1 + d['strat_ret']).cumprod()

    d = d.dropna()
    if len(d) == 0:
        return None

    tr = d['strat_eq'].iloc[-1] / INITIAL_CAPITAL
    years = (d.index[-1] - d.index[0]).days / 365.25
    cagr = (tr) ** (1 / years) - 1

    mx = d['strat_eq'].cummax()
    dd = (d['strat_eq'] - mx) / mx * 100
    mdd = dd.min()

    sharpe = (d['strat_ret'].mean() / d['strat_ret'].std()) * np.sqrt(365) if d['strat_ret'].std() > 0 else 0

    return {
        'name': name,
        'period': f"{d.index[0].date()} ~ {d.index[-1].date()}",
        'days': len(d),
        'tr': tr,
        'cagr': cagr,
        'mdd': mdd,
        'sharpe': sharpe
    }

print("="*100)
print("1. 시기별 성과 분석 (Walk-Forward Test)")
print("="*100)

# 전체 기간을 4개 구간으로 분할
total_days = len(df)
split_points = [0, total_days//4, total_days//2, 3*total_days//4, total_days]
period_names = ['Period 1 (Early)', 'Period 2', 'Period 3', 'Period 4 (Recent)']

print("\n각 시기별 SMA30 vs SMA31 성과:\n")

period_results = []
for i in range(len(split_points)-1):
    start_idx = split_points[i]
    end_idx = split_points[i+1]
    df_period = df.iloc[start_idx:end_idx]

    r30 = backtest_period(df_period, 30, 'SMA30')
    r31 = backtest_period(df_period, 31, 'SMA31')

    if r30 and r31:
        print(f"\n{period_names[i]}: {r30['period']}")
        print(f"  SMA30: Return={r30['tr']:7.2f}x, CAGR={r30['cagr']:6.2%}, MDD={r30['mdd']:6.2f}%, Sharpe={r30['sharpe']:.2f}")
        print(f"  SMA31: Return={r31['tr']:7.2f}x, CAGR={r31['cagr']:6.2%}, MDD={r31['mdd']:6.2f}%, Sharpe={r31['sharpe']:.2f}")

        winner = "SMA31" if r31['tr'] > r30['tr'] else "SMA30"
        diff = (r31['tr'] / r30['tr'] - 1) * 100
        print(f"  Winner: {winner} ({diff:+.2f}%)")

        period_results.append({
            'period': period_names[i],
            'sma30_tr': r30['tr'],
            'sma31_tr': r31['tr'],
            'winner': winner
        })

# 승률 계산
sma31_wins = sum(1 for r in period_results if r['winner'] == 'SMA31')
print(f"\n시기별 승률: SMA31 {sma31_wins}/{len(period_results)} = {sma31_wins/len(period_results)*100:.1f}%")

print("\n" + "="*100)
print("2. 통계적 유의성 검증 (월별 수익률 비교)")
print("="*100)

# 월별 수익률 계산
df['month'] = df.index.to_period('M')

for ma_p in [30, 31]:
    signal = (df['close'] > df[f'sma{ma_p}']).astype(int)
    df[f'sig{ma_p}'] = signal
    df[f'pos_chg{ma_p}'] = df[f'sig{ma_p}'].diff()
    df[f'ret{ma_p}'] = df[f'sig{ma_p}'].shift(1) * df['close'].pct_change() - abs(df[f'pos_chg{ma_p}']) * SLIPPAGE

monthly30 = df.groupby('month')[f'ret30'].apply(lambda x: (1 + x).prod() - 1) * 100
monthly31 = df.groupby('month')[f'ret31'].apply(lambda x: (1 + x).prod() - 1) * 100

# Paired t-test
t_stat, p_value = stats.ttest_rel(monthly31, monthly30)

print(f"\nSMA30 월평균: {monthly30.mean():.3f}% (중앙값: {monthly30.median():.3f}%)")
print(f"SMA31 월평균: {monthly31.mean():.3f}% (중앙값: {monthly31.median():.3f}%)")
print(f"차이: {monthly31.mean() - monthly30.mean():+.3f}%p")
print(f"\nPaired t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")

if p_value < 0.05:
    print(f"  결론: 통계적으로 유의미한 차이 있음 (p < 0.05) ✅")
else:
    print(f"  결론: 통계적으로 유의미한 차이 없음 (p >= 0.05) ⚠️")
    print(f"        → SMA31의 우위는 우연일 가능성 높음")

print("\n" + "="*100)
print("3. 파라미터 민감도 분석 (SMA 25~40)")
print("="*100)

sensitivity_results = []
for ma_p in range(25, 41):
    signal = (df['close'] > df[f'sma{ma_p}']).astype(int)
    d = df.copy()
    d['sig'] = signal
    d['pos_chg'] = d['sig'].diff()
    d['ret'] = d['close'].pct_change()
    d['strat_ret'] = d['sig'].shift(1) * d['ret'] - abs(d['pos_chg']) * SLIPPAGE
    d['strat_eq'] = INITIAL_CAPITAL * (1 + d['strat_ret']).cumprod()
    d = d.dropna()

    if len(d) > 0:
        tr = d['strat_eq'].iloc[-1] / INITIAL_CAPITAL
        mx = d['strat_eq'].cummax()
        dd = (d['strat_eq'] - mx) / mx * 100
        mdd = dd.min()

        sensitivity_results.append({
            'ma': ma_p,
            'tr': tr,
            'mdd': mdd
        })

print("\nSMA 파라미터별 성과:")
print(f"{'MA':<5} {'Total Return':>12} {'MDD':>10} {'Note':>20}")
print("-" * 60)
for r in sensitivity_results:
    note = ""
    if r['ma'] == 30:
        note = "← 벤치마크"
    elif r['ma'] == 31:
        note = "← 선택된 전략"
    elif r['tr'] > sensitivity_results[5]['tr']:  # SMA30 기준 (인덱스 5)
        note = "← 더 나음!"

    print(f"{r['ma']:<5} {r['tr']:>11.2f}x {r['mdd']:>9.2f}% {note:>20}")

# 최고 성과 찾기
best_ma = max(sensitivity_results, key=lambda x: x['tr'])
print(f"\n최고 성과: SMA{best_ma['ma']} = {best_ma['tr']:.2f}x")
print(f"SMA31 순위: {sorted(sensitivity_results, key=lambda x: x['tr'], reverse=True).index([r for r in sensitivity_results if r['ma']==31][0]) + 1}/{len(sensitivity_results)}")

print("\n" + "="*100)
print("4. 안정성 분석")
print("="*100)

# Return 변동성 (CV = Coefficient of Variation)
returns = [r['tr'] for r in sensitivity_results]
cv = np.std(returns) / np.mean(returns) * 100
print(f"\nSMA 25-40 범위에서 수익률 변동성 (CV): {cv:.2f}%")

# SMA 30 근처 (28-32) 분석
nearby_range = [r for r in sensitivity_results if 28 <= r['ma'] <= 32]
nearby_returns = [r['tr'] for r in nearby_range]
nearby_std = np.std(nearby_returns)
nearby_mean = np.mean(nearby_returns)

print(f"\nSMA 28-32 범위 분석:")
for r in nearby_range:
    print(f"  SMA{r['ma']}: {r['tr']:.2f}x")
print(f"  평균: {nearby_mean:.2f}x")
print(f"  표준편차: {nearby_std:.2f}x")
print(f"  SMA31과 평균 차이: {(sensitivity_results[6]['tr'] - nearby_mean):.2f}x")

print("\n" + "="*100)
print("최종 판단")
print("="*100)

print(f"""
분석 결과 요약:

1. 시기별 안정성:
   - SMA31 승률: {sma31_wins}/{len(period_results)} ({sma31_wins/len(period_results)*100:.0f}%)
   - 모든 시기에서 일관되게 우월하지 {'않음 ⚠️' if sma31_wins < len(period_results) else '함 ✅'}

2. 통계적 유의성:
   - p-value: {p_value:.4f}
   - {'통계적으로 유의미하지 않음 (과적합 가능성 높음) ⚠️' if p_value >= 0.05 else '통계적으로 유의미함 ✅'}

3. 파라미터 민감도:
   - 최고 성과: SMA{best_ma['ma']}
   - SMA31이 {'최선이 아님 ⚠️' if best_ma['ma'] != 31 else '최선임 ✅'}
   - SMA 28-32 범위에서 변동성: {nearby_std:.2f}x

결론:
""")

if p_value >= 0.05:
    print("⚠️  SMA31의 우위는 통계적으로 유의미하지 않습니다.")
    print("    이것은 8년 데이터에서 우연히 나온 결과일 가능성이 높습니다.")
    print("    미래에도 작동할 것이라는 근거가 약합니다.")
    print("\n권장사항:")
    print("    1. SMA30 (벤치마크)을 그대로 사용")
    print("    2. 또는 SMA 28-32 범위의 평균/중앙값 사용")
    print("    3. 단일 파라미터 최적화보다는 앙상블 접근")
else:
    print("✅ SMA31의 우위가 통계적으로 유의미합니다.")
    print("   미래에도 작동할 가능성이 있습니다.")

# 시각화
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 파라미터 민감도
ax1 = axes[0]
ma_values = [r['ma'] for r in sensitivity_results]
tr_values = [r['tr'] for r in sensitivity_results]
ax1.plot(ma_values, tr_values, 'b-o', linewidth=2, markersize=8)
ax1.axvline(x=30, color='red', linestyle='--', label='SMA30 (Benchmark)', linewidth=2)
ax1.axvline(x=31, color='green', linestyle='--', label='SMA31 (Selected)', linewidth=2)
ax1.axvline(x=best_ma['ma'], color='orange', linestyle=':', label=f"SMA{best_ma['ma']} (Best)", linewidth=2)
ax1.set_xlabel('SMA Period', fontsize=12)
ax1.set_ylabel('Total Return (x)', fontsize=12)
ax1.set_title('Parameter Sensitivity Analysis', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 월별 수익률 분포
ax2 = axes[1]
ax2.hist(monthly30, bins=30, alpha=0.5, label='SMA30', color='red')
ax2.hist(monthly31, bins=30, alpha=0.5, label='SMA31', color='green')
ax2.axvline(x=monthly30.mean(), color='red', linestyle='--', linewidth=2, label=f'SMA30 Mean: {monthly30.mean():.2f}%')
ax2.axvline(x=monthly31.mean(), color='green', linestyle='--', linewidth=2, label=f'SMA31 Mean: {monthly31.mean():.2f}%')
ax2.set_xlabel('Monthly Return (%)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title(f'Monthly Return Distribution (p-value: {p_value:.4f})', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/robustness_test.png', dpi=300, bbox_inches='tight')
print(f"\n시각화 저장: output/robustness_test.png")
plt.close()
