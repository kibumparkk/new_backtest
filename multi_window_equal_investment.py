"""
멀티윈도우 균등투자 전략 백테스트

전략 설명:
- 여러 이동평균 윈도우(5, 10, 20, 30, 60일)를 사용
- 각 윈도우별로 시그널 생성: 전일 종가 > 전일 SMA
- 활성화된 시그널의 비율만큼 투자 (예: 5개 중 3개 활성 → 60% 투자)
- 슬리피지 0.2% 적용
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 출력 폴더 생성
os.makedirs('output', exist_ok=True)

# ===== 초기 설정 =====
INITIAL_CAPITAL = 1  # 1원
SLIPPAGE = 0.002     # 0.2%
WINDOWS = [5, 10, 20, 30, 60]  # 멀티윈도우

print("=" * 60)
print("멀티윈도우 균등투자 전략 백테스트")
print("=" * 60)
print(f"초기 자본: {INITIAL_CAPITAL}원")
print(f"슬리피지: {SLIPPAGE * 100}%")
print(f"윈도우: {WINDOWS}")
print()

# ===== 데이터 로드 =====
df = pd.read_parquet('chart_day/BTC_KRW.parquet')
print(f"데이터 기간: {df.index.min()} ~ {df.index.max()}")
print(f"총 {len(df)}일")
print(f"컬럼: {list(df.columns)}")
print()

# ===== 수익률 계산 =====
df['returns'] = df['close'].pct_change()

# ===== 벤치마크 전략 (SMA30) =====
df['sma30'] = df['close'].rolling(window=30).mean()
df['benchmark_signal'] = (df['close'].shift(1) > df['sma30'].shift(1)).astype(int)
df['benchmark_returns'] = df['benchmark_signal'] * df['returns']
df['benchmark_equity'] = INITIAL_CAPITAL * (1 + df['benchmark_returns']).cumprod()

# ===== 멀티윈도우 전략 =====
# 각 윈도우별 SMA 계산 및 시그널 생성
signals = []
for window in WINDOWS:
    sma_col = f'sma{window}'
    signal_col = f'signal{window}'

    # SMA 계산
    df[sma_col] = df['close'].rolling(window=window).mean()

    # 시그널 생성: 전일 종가 > 전일 SMA
    df[signal_col] = (df['close'].shift(1) > df[sma_col].shift(1)).astype(int)

    signals.append(signal_col)

# 활성 시그널 개수 계산
df['active_signals'] = df[signals].sum(axis=1)

# 포지션 비중 계산 (0.0 ~ 1.0)
df['position'] = df['active_signals'] / len(WINDOWS)

# 전략 수익률 계산
# 포지션 변화가 있을 때만 슬리피지 적용
df['position_change'] = df['position'].diff().abs()
df['slippage_cost'] = df['position_change'] * SLIPPAGE

# 전략 수익률 = 포지션 비중 × 시장 수익률 - 슬리피지
df['strategy_returns'] = df['position'] * df['returns'] - df['slippage_cost']

# 자산 곡선 계산
df['strategy_equity'] = INITIAL_CAPITAL * (1 + df['strategy_returns']).cumprod()

# Buy & Hold 전략
df['buyhold_equity'] = INITIAL_CAPITAL * (1 + df['returns']).cumprod()

# ===== Drawdown 계산 =====
# 전략 Drawdown
strategy_cumulative = df['strategy_equity']
strategy_running_max = strategy_cumulative.cummax()
df['strategy_drawdown'] = (strategy_cumulative - strategy_running_max) / strategy_running_max * 100

# 벤치마크 Drawdown
benchmark_cumulative = df['benchmark_equity']
benchmark_running_max = benchmark_cumulative.cummax()
df['benchmark_drawdown'] = (benchmark_cumulative - benchmark_running_max) / benchmark_running_max * 100

# ===== 성과 지표 계산 =====
# 전략 성과
final_equity = df['strategy_equity'].iloc[-1]
total_return = final_equity / INITIAL_CAPITAL

total_days = (df.index[-1] - df.index[0]).days
years = total_days / 365.25
cagr = (final_equity / INITIAL_CAPITAL) ** (1 / years) - 1

mdd = df['strategy_drawdown'].min()

# 샤프 비율 (연율화)
strategy_returns_clean = df['strategy_returns'].dropna()
sharpe_ratio = np.sqrt(365) * strategy_returns_clean.mean() / strategy_returns_clean.std() if strategy_returns_clean.std() > 0 else 0

# 승률 계산
win_days = (strategy_returns_clean > 0).sum()
total_days_traded = len(strategy_returns_clean)
win_rate = win_days / total_days_traded if total_days_traded > 0 else 0

# 벤치마크 성과
benchmark_final = df['benchmark_equity'].iloc[-1]
benchmark_total_return = benchmark_final / INITIAL_CAPITAL
benchmark_cagr = (benchmark_final / INITIAL_CAPITAL) ** (1 / years) - 1
benchmark_mdd = df['benchmark_drawdown'].min()

# Buy & Hold 성과
buyhold_final = df['buyhold_equity'].iloc[-1]
buyhold_total_return = buyhold_final / INITIAL_CAPITAL
buyhold_cagr = (buyhold_final / INITIAL_CAPITAL) ** (1 / years) - 1

# ===== 성과 출력 =====
print("=" * 60)
print("백테스트 결과")
print("=" * 60)
print("\n[멀티윈도우 균등투자 전략]")
print(f"  Total Return: {total_return:.2f}x")
print(f"  CAGR: {cagr:.2%}")
print(f"  MDD: {mdd:.2f}%")
print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"  Win Rate: {win_rate:.2%}")
print(f"  최종 자산: {final_equity:,.0f}원")

print("\n[벤치마크 (SMA30)]")
print(f"  Total Return: {benchmark_total_return:.2f}x")
print(f"  CAGR: {benchmark_cagr:.2%}")
print(f"  MDD: {benchmark_mdd:.2f}%")
print(f"  최종 자산: {benchmark_final:,.0f}원")

print("\n[Buy & Hold]")
print(f"  Total Return: {buyhold_total_return:.2f}x")
print(f"  CAGR: {buyhold_cagr:.2%}")
print(f"  최종 자산: {buyhold_final:,.0f}원")

print("\n[전략 vs 벤치마크]")
print(f"  초과 수익률: {(total_return - benchmark_total_return):.2f}x")
print(f"  초과 CAGR: {(cagr - benchmark_cagr):.2%}")
print()

# ===== 월별 수익률 계산 =====
monthly_returns = df['strategy_returns'].resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100
monthly_returns_df = monthly_returns.to_frame(name='Monthly Return (%)')
monthly_returns_df.to_csv('output/monthly_returns.csv')

# 월별 수익률 히트맵용 피벗 테이블
monthly_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).sum()
monthly_pivot_table = monthly_pivot.unstack(fill_value=np.nan)
if len(monthly_pivot_table) > 0:
    monthly_pivot_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# ===== 성과 요약 저장 (CSV) =====
performance_summary = pd.DataFrame({
    'Metric': ['Total Return', 'CAGR', 'MDD', 'Sharpe Ratio', 'Win Rate',
               'Final Equity (KRW)', 'Start Date', 'End Date', 'Total Days',
               'Benchmark Return', 'Benchmark CAGR', 'Benchmark MDD'],
    'Value': [f'{total_return:.2f}x', f'{cagr:.2%}', f'{mdd:.2f}%', f'{sharpe_ratio:.2f}', f'{win_rate:.2%}',
              f'{final_equity:,.0f}', str(df.index.min().date()), str(df.index.max().date()), total_days,
              f'{benchmark_total_return:.2f}x', f'{benchmark_cagr:.2%}', f'{benchmark_mdd:.2f}%']
})
performance_summary.to_csv('output/performance_summary.csv', index=False)

# ===== 시각화 =====
fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(4, 1, height_ratios=[2.5, 1.5, 1, 2], hspace=0.35)

# Subplot 1: 누적 수익률 (로그 스케일)
ax1 = fig.add_subplot(gs[0])
ax1.plot(df.index, df['strategy_equity'], label='Multi-Window Equal Investment',
         linewidth=2, color='#2E86AB')
ax1.plot(df.index, df['benchmark_equity'], label='Benchmark (SMA30)',
         linewidth=2, alpha=0.7, color='#A23B72')
ax1.plot(df.index, df['buyhold_equity'], label='Buy & Hold',
         linewidth=1.5, alpha=0.5, linestyle='--', color='#F18F01')
ax1.set_yscale('log')
ax1.set_ylabel('Cumulative Returns (KRW, log scale)', fontsize=12, fontweight='bold')
ax1.set_title('Multi-Window Equal Investment Strategy - Backtest Performance Analysis',
              fontsize=15, fontweight='bold', pad=20)
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(True, alpha=0.3, linestyle='--')

# 성과지표 텍스트 박스
metrics_text = f'''Strategy Performance:
Total Return: {total_return:.2f}x
CAGR: {cagr:.2%}
MDD: {mdd:.2f}%
Sharpe: {sharpe_ratio:.2f}
Win Rate: {win_rate:.2%}

Benchmark (SMA30):
Total Return: {benchmark_total_return:.2f}x
CAGR: {benchmark_cagr:.2%}
MDD: {benchmark_mdd:.2f}%'''

ax1.text(0.98, 0.97, metrics_text, transform=ax1.transAxes,
         fontsize=9.5, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
         family='monospace')

# Subplot 2: 포지션 비중 (Position Weight)
ax2 = fig.add_subplot(gs[1])
ax2.fill_between(df.index, 0, df['position'] * 100, alpha=0.4, color='#06A77D', label='Position Weight')
ax2.plot(df.index, df['position'] * 100, linewidth=1, color='#06A77D', alpha=0.8)
ax2.set_ylabel('Position Weight (%)', fontsize=11, fontweight='bold')
ax2.set_ylim(-5, 105)
ax2.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax2.axhline(y=100, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper left', fontsize=10)

# 평균 포지션 비중 표시
avg_position = df['position'].mean() * 100
ax2.text(0.02, 0.95, f'Avg: {avg_position:.1f}%', transform=ax2.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# Subplot 3: Drawdown (%)
ax3 = fig.add_subplot(gs[2])
ax3.fill_between(df.index, 0, df['strategy_drawdown'], color='#C73E1D', alpha=0.4, label='Strategy DD')
ax3.plot(df.index, df['strategy_drawdown'], color='#C73E1D', linewidth=1, alpha=0.8)
ax3.fill_between(df.index, 0, df['benchmark_drawdown'], color='#F18F01', alpha=0.3, label='Benchmark DD')
ax3.plot(df.index, df['benchmark_drawdown'], color='#F18F01', linewidth=1, alpha=0.6)
ax3.set_ylabel('Drawdown (%)', fontsize=11, fontweight='bold')
ax3.set_xlabel('Date', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.axhline(y=0, color='black', linewidth=0.8)
ax3.legend(loc='lower left', fontsize=10)

# Subplot 4: 월별 수익률 히트맵
ax4 = fig.add_subplot(gs[3])
if len(monthly_pivot_table) > 0:
    # NaN 값을 가진 셀은 회색으로 표시
    sns.heatmap(monthly_pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                ax=ax4, cbar_kws={'label': 'Monthly Return (%)'},
                linewidths=0.5, linecolor='gray', vmin=-30, vmax=30,
                cbar=True, square=False)
    ax4.set_ylabel('Year', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax4.set_title('Monthly Returns Heatmap (%)', fontsize=12, fontweight='bold')
else:
    ax4.text(0.5, 0.5, 'Insufficient data for monthly heatmap',
             ha='center', va='center', fontsize=12)
    ax4.axis('off')

# 저장
plt.savefig('output/backtest_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("=" * 60)
print("결과 저장 완료")
print("=" * 60)
print("  - output/backtest_results.png")
print("  - output/performance_summary.csv")
print("  - output/monthly_returns.csv")
print("=" * 60)
print("\n백테스트 완료!")
