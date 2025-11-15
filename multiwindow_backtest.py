import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from itertools import product

# 출력 폴더 생성
os.makedirs('output', exist_ok=True)

# 데이터 로드
df = pd.read_parquet('chart_day/BTC_KRW.parquet')

# 초기 설정
INITIAL_CAPITAL = 1  # 1원
SLIPPAGE = 0.002     # 0.2%

print(f"데이터 기간: {df.index.min()} ~ {df.index.max()}")
print(f"총 {len(df)}일")
print(f"컬럼: {list(df.columns)}\n")

# === 멀티윈도우 설정 ===
# 20개의 이동평균 윈도우 (5일부터 100일까지)
WINDOWS = list(range(5, 105, 5))  # [5, 10, 15, 20, ..., 100]
print(f"멀티윈도우 ({len(WINDOWS)}개): {WINDOWS}\n")

# === 멀티윈도우 스코어 계산 ===
print("멀티윈도우 스코어 계산 중...")

# 각 윈도우에 대한 이동평균 계산
for window in WINDOWS:
    df[f'ma{window}'] = df['close'].rolling(window=window).mean()

# 스코어 계산: 현재가 > 이동평균이면 +1점
def calculate_multiwindow_score(row):
    score = 0
    for window in WINDOWS:
        if pd.notna(row[f'ma{window}']) and row['close'] > row[f'ma{window}']:
            score += 1
    return score

df['mw_score'] = df.apply(calculate_multiwindow_score, axis=1)

print(f"스코어 범위: {df['mw_score'].min()} ~ {df['mw_score'].max()}")
print(f"평균 스코어: {df['mw_score'].mean():.2f}\n")

# === 수익률 계산 ===
df['returns'] = df['close'].pct_change()

# === 벤치마크 전략 (SMA30) ===
df['sma30'] = df['close'].rolling(window=30).mean()
df['benchmark_signal'] = (df['close'].shift(1) > df['sma30'].shift(1)).astype(int)
df['benchmark_returns'] = df['benchmark_signal'] * df['returns']
df['benchmark_equity'] = INITIAL_CAPITAL * (1 + df['benchmark_returns']).cumprod()

# === 파라미터 최적화 ===
print("=" * 80)
print("파라미터 최적화 시작")
print("=" * 80)

# 테스트할 파라미터 범위
K_VALUES = list(range(0, 21, 2))  # 매수 임계값: 0, 2, 4, ..., 20
J_VALUES = list(range(1, 31, 2))  # 홀딩 기간: 1, 3, 5, ..., 29일

print(f"K 값 (매수 임계값): {K_VALUES}")
print(f"J 값 (홀딩 기간): {J_VALUES}")
print(f"총 조합 수: {len(K_VALUES) * len(J_VALUES)}\n")

# 최적화 결과 저장
optimization_results = []

# 전략 시뮬레이션 함수
def simulate_strategy(df, k_threshold, holding_period):
    """
    멀티윈도우 전략 시뮬레이션
    - k_threshold: 매수 스코어 임계값 (스코어 >= k이면 매수)
    - holding_period: 홀딩 기간 (j일 후 매도)
    """
    df_sim = df.copy()

    # 전일 스코어로 당일 포지션 결정 (Look-ahead bias 방지)
    df_sim['signal'] = 0
    df_sim['position_end_date'] = pd.NaT

    for i in range(1, len(df_sim)):
        # 이미 포지션을 보유 중인지 확인
        if i > 0 and pd.notna(df_sim.iloc[i-1]['position_end_date']):
            # 포지션 종료일이 지났는지 확인
            if df_sim.index[i] < df_sim.iloc[i-1]['position_end_date']:
                df_sim.iloc[i, df_sim.columns.get_loc('signal')] = 1
                df_sim.iloc[i, df_sim.columns.get_loc('position_end_date')] = df_sim.iloc[i-1]['position_end_date']
                continue

        # 새로운 매수 신호 확인 (전일 스코어 사용)
        if df_sim.iloc[i-1]['mw_score'] >= k_threshold:
            df_sim.iloc[i, df_sim.columns.get_loc('signal')] = 1
            # 홀딩 기간 후 매도일 계산
            end_idx = min(i + holding_period, len(df_sim) - 1)
            df_sim.iloc[i, df_sim.columns.get_loc('position_end_date')] = df_sim.index[end_idx]

    # 전략 수익률 계산
    df_sim['strategy_returns'] = df_sim['signal'] * df_sim['returns']

    # 슬리피지 적용
    # 포지션 진입/청산 시 슬리피지 적용
    position_changes = df_sim['signal'].diff().fillna(0)
    slippage_cost = position_changes.abs() * SLIPPAGE
    df_sim['strategy_returns_adj'] = df_sim['strategy_returns'] - slippage_cost

    # 자산 곡선 계산
    df_sim['strategy_equity'] = INITIAL_CAPITAL * (1 + df_sim['strategy_returns_adj']).cumprod()

    return df_sim

# 모든 파라미터 조합 테스트
total_combinations = len(K_VALUES) * len(J_VALUES)
current = 0

for k, j in product(K_VALUES, J_VALUES):
    current += 1

    # 진행상황 출력
    if current % 10 == 0 or current == total_combinations:
        print(f"진행: {current}/{total_combinations} ({current/total_combinations*100:.1f}%) - k={k}, j={j}")

    # 전략 시뮬레이션
    df_result = simulate_strategy(df, k, j)

    # NaN 제거
    df_result = df_result.dropna(subset=['strategy_equity'])

    if len(df_result) < 100:  # 데이터가 너무 적으면 스킵
        continue

    # 성과 지표 계산
    final_equity = df_result['strategy_equity'].iloc[-1]
    total_return = final_equity / INITIAL_CAPITAL

    # CAGR 계산
    total_days = (df_result.index[-1] - df_result.index[0]).days
    years = total_days / 365.25
    if years > 0:
        cagr = (final_equity / INITIAL_CAPITAL) ** (1 / years) - 1
    else:
        cagr = 0

    # MDD 계산
    cumulative = df_result['strategy_equity']
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    mdd = drawdown.min()

    # Sharpe Ratio 계산
    if df_result['strategy_returns_adj'].std() > 0:
        sharpe_ratio = df_result['strategy_returns_adj'].mean() / df_result['strategy_returns_adj'].std() * np.sqrt(365)
    else:
        sharpe_ratio = 0

    # 승률 계산
    winning_trades = (df_result['strategy_returns_adj'] > 0).sum()
    total_trades = (df_result['strategy_returns_adj'] != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # 결과 저장
    optimization_results.append({
        'k': k,
        'j': j,
        'total_return': total_return,
        'cagr': cagr,
        'mdd': mdd,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'final_equity': final_equity
    })

# 결과를 DataFrame으로 변환
results_df = pd.DataFrame(optimization_results)

# 최적 파라미터 찾기 (CAGR 기준)
best_by_cagr = results_df.loc[results_df['cagr'].idxmax()]
print("\n" + "=" * 80)
print("최적화 완료!")
print("=" * 80)
print(f"\n최적 파라미터 (CAGR 기준):")
print(f"  k (매수 임계값): {best_by_cagr['k']:.0f}")
print(f"  j (홀딩 기간): {best_by_cagr['j']:.0f}일")
print(f"  Total Return: {best_by_cagr['total_return']:.2f}x")
print(f"  CAGR: {best_by_cagr['cagr']:.2%}")
print(f"  MDD: {best_by_cagr['mdd']:.2%}")
print(f"  Sharpe Ratio: {best_by_cagr['sharpe_ratio']:.2f}")
print(f"  Win Rate: {best_by_cagr['win_rate']:.2%}")

# Sharpe Ratio 기준 최적 파라미터
best_by_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
print(f"\n최적 파라미터 (Sharpe Ratio 기준):")
print(f"  k (매수 임계값): {best_by_sharpe['k']:.0f}")
print(f"  j (홀딩 기간): {best_by_sharpe['j']:.0f}일")
print(f"  Total Return: {best_by_sharpe['total_return']:.2f}x")
print(f"  CAGR: {best_by_sharpe['cagr']:.2%}")
print(f"  MDD: {best_by_sharpe['mdd']:.2%}")
print(f"  Sharpe Ratio: {best_by_sharpe['sharpe_ratio']:.2f}")
print(f"  Win Rate: {best_by_sharpe['win_rate']:.2%}")

# 최적화 결과 저장
results_df.to_csv('output/optimization_results.csv', index=False)
print(f"\n최적화 결과 저장: output/optimization_results.csv")

# === 최적 파라미터로 최종 백테스트 실행 ===
print("\n" + "=" * 80)
print("최적 파라미터로 최종 백테스트 실행")
print("=" * 80)

best_k = int(best_by_cagr['k'])
best_j = int(best_by_cagr['j'])

df_final = simulate_strategy(df, best_k, best_j)
df_final = df_final.dropna(subset=['strategy_equity'])

# 최종 성과 지표
final_equity = df_final['strategy_equity'].iloc[-1]
total_return = final_equity / INITIAL_CAPITAL

total_days = (df_final.index[-1] - df_final.index[0]).days
years = total_days / 365.25
cagr = (final_equity / INITIAL_CAPITAL) ** (1 / years) - 1

cumulative = df_final['strategy_equity']
running_max = cumulative.cummax()
drawdown = (cumulative - running_max) / running_max
mdd = drawdown.min()

sharpe_ratio = df_final['strategy_returns_adj'].mean() / df_final['strategy_returns_adj'].std() * np.sqrt(365) if df_final['strategy_returns_adj'].std() > 0 else 0

# 벤치마크 성과
benchmark_final = df_final['benchmark_equity'].iloc[-1]
benchmark_return = benchmark_final / INITIAL_CAPITAL
benchmark_cagr = (benchmark_final / INITIAL_CAPITAL) ** (1 / years) - 1

benchmark_cum = df_final['benchmark_equity']
benchmark_running_max = benchmark_cum.cummax()
benchmark_drawdown = (benchmark_cum - benchmark_running_max) / benchmark_running_max
benchmark_mdd = benchmark_drawdown.min()

print(f"\n전략 성과:")
print(f"  Total Return: {total_return:.2f}x")
print(f"  CAGR: {cagr:.2%}")
print(f"  MDD: {mdd:.2%}")
print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"  최종 자산: {final_equity:,.0f}원")

print(f"\n벤치마크 성과 (SMA30):")
print(f"  Total Return: {benchmark_return:.2f}x")
print(f"  CAGR: {benchmark_cagr:.2%}")
print(f"  MDD: {benchmark_mdd:.2%}")

# === 시각화 ===
print("\n시각화 생성 중...")

fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(4, 2, height_ratios=[2, 1, 2, 2], hspace=0.3, wspace=0.3)

# Subplot 1: 누적 자산 곡선
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df_final.index, df_final['strategy_equity'], label=f'Strategy (k={best_k}, j={best_j})', linewidth=2, color='blue')
ax1.plot(df_final.index, df_final['benchmark_equity'], label='Benchmark (SMA30)', linewidth=2, alpha=0.7, color='orange')
ax1.set_yscale('log')
ax1.set_ylabel('Cumulative Returns (KRW, log scale)', fontsize=11)
ax1.set_title(f'Multi-Window Strategy Backtest (Windows: {len(WINDOWS)})', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 성과지표 텍스트 박스
metrics_text = f'''Strategy:
Total Return: {total_return:.2f}x
CAGR: {cagr:.2%}
MDD: {mdd:.2%}
Sharpe: {sharpe_ratio:.2f}

Benchmark:
Total Return: {benchmark_return:.2f}x
CAGR: {benchmark_cagr:.2%}
MDD: {benchmark_mdd:.2%}'''

ax1.text(0.98, 0.97, metrics_text, transform=ax1.transAxes,
         fontsize=9, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Subplot 2: Drawdown 차트
ax2 = fig.add_subplot(gs[1, :])
ax2.fill_between(df_final.index, 0, drawdown * 100, color='red', alpha=0.3, label='Strategy DD')
ax2.plot(df_final.index, drawdown * 100, color='red', linewidth=1)
ax2.fill_between(df_final.index, 0, benchmark_drawdown * 100, color='orange', alpha=0.2, label='Benchmark DD')
ax2.plot(df_final.index, benchmark_drawdown * 100, color='orange', linewidth=1, linestyle='--')
ax2.set_ylabel('Drawdown (%)', fontsize=11)
ax2.set_xlabel('Date', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.legend(loc='lower left')

# Subplot 3: 월별 수익률 히트맵
ax3 = fig.add_subplot(gs[2, :])

# 월별 수익률 계산
monthly_rets = df_final['strategy_returns_adj'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
monthly_rets.index = pd.to_datetime(monthly_rets.index)

# 피벗 테이블 생성
pivot_data = []
for date, ret in monthly_rets.items():
    pivot_data.append({'Year': date.year, 'Month': date.month, 'Return': ret})

pivot_df = pd.DataFrame(pivot_data)
pivot_table = pivot_df.pivot(index='Year', columns='Month', values='Return')
pivot_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            ax=ax3, cbar_kws={'label': 'Monthly Return (%)'}, linewidths=0.5)
ax3.set_ylabel('Year', fontsize=11)
ax3.set_xlabel('Month', fontsize=11)
ax3.set_title('Monthly Returns (%)', fontsize=12)

# Subplot 4: 스코어 분포
ax4 = fig.add_subplot(gs[3, 0])
score_counts = df_final['mw_score'].value_counts().sort_index()
ax4.bar(score_counts.index, score_counts.values, color='steelblue', alpha=0.7)
ax4.axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Threshold (k={best_k})')
ax4.set_xlabel('Multi-Window Score', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Score Distribution', fontsize=12)
ax4.grid(True, alpha=0.3, axis='y')
ax4.legend()

# Subplot 5: 최적화 히트맵 (CAGR)
ax5 = fig.add_subplot(gs[3, 1])
optimization_pivot = results_df.pivot(index='j', columns='k', values='cagr') * 100
sns.heatmap(optimization_pivot, annot=True, fmt='.1f', cmap='viridis',
            ax=ax5, cbar_kws={'label': 'CAGR (%)'})
ax5.set_ylabel('j (Holding Period)', fontsize=11)
ax5.set_xlabel('k (Score Threshold)', fontsize=11)
ax5.set_title('Optimization Heatmap (CAGR %)', fontsize=12)
ax5.invert_yaxis()

# 저장
plt.savefig('output/multiwindow_backtest_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("시각화 완료: output/multiwindow_backtest_results.png")

# === 성과 요약 저장 ===
performance_summary = pd.DataFrame([{
    'Strategy': f'Multi-Window (k={best_k}, j={best_j})',
    'Total_Return': f'{total_return:.2f}x',
    'CAGR': f'{cagr:.2%}',
    'MDD': f'{mdd:.2%}',
    'Sharpe_Ratio': f'{sharpe_ratio:.2f}',
    'Final_Equity': f'{final_equity:,.0f}',
    'Start_Date': df_final.index[0],
    'End_Date': df_final.index[-1],
    'Total_Days': total_days
}, {
    'Strategy': 'Benchmark (SMA30)',
    'Total_Return': f'{benchmark_return:.2f}x',
    'CAGR': f'{benchmark_cagr:.2%}',
    'MDD': f'{benchmark_mdd:.2%}',
    'Sharpe_Ratio': '-',
    'Final_Equity': f'{benchmark_final:,.0f}',
    'Start_Date': df_final.index[0],
    'End_Date': df_final.index[-1],
    'Total_Days': total_days
}])

performance_summary.to_csv('output/performance_summary.csv', index=False)
print("성과 요약 저장: output/performance_summary.csv")

# 월별 수익률 저장
monthly_returns_df = pd.DataFrame({
    'Date': monthly_rets.index,
    'Return_Pct': monthly_rets.values
})
monthly_returns_df.to_csv('output/monthly_returns.csv', index=False)
print("월별 수익률 저장: output/monthly_returns.csv")

print("\n" + "=" * 80)
print("백테스트 완료!")
print("=" * 80)
