"""
ATR-Based Leveraged Trend Following Strategy
==============================================

벤치마크를 5,340% 초과 달성한 최고 성과 전략

전략 설명:
- 기본 시그널: 종가 > SMA30
- 포지션 크기: ATR 기반 동적 레버리지 조정
- ATR이 클수록 변동성이 크므로 더 큰 포지션 진입
- 슬리피지: 0.2% (매수/매도 각각)

백테스트 성과 (2017-09-25 ~ 2025-11-10):
- Total Return: 13,711.24x
- Benchmark Return: 252.03x (SMA30 Fixed 1x)
- Outperformance: +5,340%
- CAGR: 230.71%
- MDD: -73.8%
- Sharpe Ratio: 1.70
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('output', exist_ok=True)

print("="*80)
print("ATR-Based Leveraged Trend Following Strategy")
print("="*80)

# 데이터 로드
df = pd.read_parquet('chart_day/BTC_KRW.parquet')
print(f"\n데이터 기간: {df.index.min()} ~ {df.index.max()}")
print(f"총 {len(df)}일\n")

INITIAL_CAPITAL = 1
SLIPPAGE = 0.002
ATR_MULTIPLIER = 1.0  # ATR 승수

# 지표 계산
print("지표 계산 중...")
df['sma30'] = df['close'].rolling(window=30).mean()

# ATR 계산
df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    )
)
df['atr14'] = df['tr'].rolling(window=14).mean()

# 전략 시그널
df['signal'] = (df['close'] > df['sma30']).astype(int)

# ATR 기반 포지션 크기 (ATR이 클수록 더 큰 포지션)
df['position_size'] = df['signal'] * (1 + (df['atr14'] / df['close'] * 100 * ATR_MULTIPLIER))
df['position_size'] = df['position_size'].clip(0, 2)  # 최대 2x 레버리지

# 벤치마크 (고정 1x)
df['benchmark_signal'] = df['signal']  # 동일한 시그널, 고정 1x 포지션

# 일일 수익률
df['daily_return'] = df['close'].pct_change()

# 전략 수익률 (가변 포지션)
df['position_change'] = df['position_size'].diff()
df['strategy_return'] = (
    df['position_size'].shift(1) * df['daily_return'] -
    abs(df['position_change']) * SLIPPAGE
)
df['strategy_equity'] = INITIAL_CAPITAL * (1 + df['strategy_return']).cumprod()

# 벤치마크 수익률 (고정 1x)
df['benchmark_position_change'] = df['benchmark_signal'].diff()
df['benchmark_return'] = (
    df['benchmark_signal'].shift(1) * df['daily_return'] -
    abs(df['benchmark_position_change']) * SLIPPAGE
)
df['benchmark_equity'] = INITIAL_CAPITAL * (1 + df['benchmark_return']).cumprod()

# Buy & Hold
df['bh_return'] = df['daily_return']
df['bh_equity'] = INITIAL_CAPITAL * (1 + df['bh_return']).cumprod()

df = df.dropna()

# 성과 지표 계산
print("성과 지표 계산 중...")

# Total Return
strategy_total_return = df['strategy_equity'].iloc[-1] / INITIAL_CAPITAL
benchmark_total_return = df['benchmark_equity'].iloc[-1] / INITIAL_CAPITAL
bh_total_return = df['bh_equity'].iloc[-1] / INITIAL_CAPITAL

# CAGR
years = (df.index[-1] - df.index[0]).days / 365.25
strategy_cagr = (df['strategy_equity'].iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1
benchmark_cagr = (df['benchmark_equity'].iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1

# MDD
strategy_running_max = df['strategy_equity'].cummax()
strategy_drawdown = (df['strategy_equity'] - strategy_running_max) / strategy_running_max * 100
strategy_mdd = strategy_drawdown.min()

benchmark_running_max = df['benchmark_equity'].cummax()
benchmark_drawdown = (df['benchmark_equity'] - benchmark_running_max) / benchmark_running_max * 100
benchmark_mdd = benchmark_drawdown.min()

# Sharpe Ratio
strategy_sharpe = (df['strategy_return'].mean() / df['strategy_return'].std()) * np.sqrt(365) if df['strategy_return'].std() > 0 else 0
benchmark_sharpe = (df['benchmark_return'].mean() / df['benchmark_return'].std()) * np.sqrt(365) if df['benchmark_return'].std() > 0 else 0

# 결과 출력
print("\n" + "="*80)
print("백테스트 결과 - ATR Leveraged Strategy")
print("="*80)

print(f"\n전략 성과:")
print(f"  Total Return: {strategy_total_return:,.2f}x")
print(f"  CAGR: {strategy_cagr:.2%}")
print(f"  MDD: {strategy_mdd:.2%}")
print(f"  Sharpe Ratio: {strategy_sharpe:.2f}")

print(f"\n벤치마크 성과 (SMA30 Fixed 1x):")
print(f"  Total Return: {benchmark_total_return:.2f}x")
print(f"  CAGR: {benchmark_cagr:.2%}")
print(f"  MDD: {benchmark_mdd:.2%}")
print(f"  Sharpe Ratio: {benchmark_sharpe:.2f}")

outperformance = (strategy_total_return / benchmark_total_return - 1) * 100
print(f"\n💰 Outperformance: {outperformance:+,.1f}%")

print(f"\nBuy & Hold: {bh_total_return:.2f}x")
print(f"전략 vs Buy & Hold: {strategy_total_return / bh_total_return:.1f}배")

# 포지션 크기 분석
avg_position = df[df['signal'] == 1]['position_size'].mean()
max_position = df['position_size'].max()
print(f"\n포지션 크기 통계:")
print(f"  평균 포지션: {avg_position:.2f}x")
print(f"  최대 포지션: {max_position:.2f}x")

# 월별 수익률
print("\n월별 수익률 계산 중...")
df['month'] = df.index.to_period('M')
monthly_returns = df.groupby('month')['strategy_return'].apply(lambda x: (1 + x).prod() - 1) * 100
monthly_returns.index = monthly_returns.index.to_timestamp()

pivot_data = []
for date, ret in monthly_returns.items():
    pivot_data.append({'year': date.year, 'month': date.month, 'return': ret})

pivot_df = pd.DataFrame(pivot_data)
pivot_table = pivot_df.pivot(index='year', columns='month', values='return')
pivot_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Drawdown
df['drawdown_pct'] = strategy_drawdown

# 시각화
print("\n시각화 생성 중...")
fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(4, 1, height_ratios=[2.5, 1, 1, 2], hspace=0.35)

# Subplot 1: 누적 자산 곡선
ax1 = fig.add_subplot(gs[0])
ax1.plot(df.index, df['strategy_equity'], label='ATR Leveraged Strategy',
         linewidth=2.5, color='#2E86AB')
ax1.plot(df.index, df['benchmark_equity'], label='Benchmark (SMA30 1x)',
         linewidth=2, alpha=0.7, color='#A23B72')
ax1.plot(df.index, df['bh_equity'], label='Buy & Hold',
         linewidth=1.5, alpha=0.5, linestyle='--', color='#F18F01')
ax1.set_yscale('log')
ax1.set_ylabel('Cumulative Returns (KRW, log scale)', fontsize=12)
ax1.set_title('ATR-Based Leveraged Trend Following Strategy - Performance Analysis',
              fontsize=15, fontweight='bold')
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(True, alpha=0.3)

# 성과지표
metrics_text = f'''Strategy Performance:
Total Return: {strategy_total_return:,.0f}x
CAGR: {strategy_cagr:.1%}
MDD: {strategy_mdd:.1%}
Sharpe: {strategy_sharpe:.2f}

Benchmark: {benchmark_total_return:.0f}x
Outperformance: {outperformance:+,.0f}%

Avg Position: {avg_position:.2f}x
Max Position: {max_position:.2f}x'''

ax1.text(0.98, 0.97, metrics_text, transform=ax1.transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
         family='monospace')

# Subplot 2: Drawdown
ax2 = fig.add_subplot(gs[1])
ax2.fill_between(df.index, 0, df['drawdown_pct'], color='red', alpha=0.3)
ax2.plot(df.index, df['drawdown_pct'], color='darkred', linewidth=1)
ax2.set_ylabel('Drawdown (%)', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_title('Strategy Drawdown', fontsize=12)

# Subplot 3: 포지션 크기
ax3 = fig.add_subplot(gs[2])
ax3.plot(df.index, df['position_size'], color='green', linewidth=1, alpha=0.6)
ax3.fill_between(df.index, 0, df['position_size'], color='green', alpha=0.2)
ax3.set_ylabel('Position Size (x)', fontsize=11)
ax3.set_xlabel('Date', fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.axhline(y=1, color='blue', linewidth=1, linestyle='--', alpha=0.5, label='1x (Benchmark)')
ax3.legend(loc='upper left', fontsize=9)
ax3.set_title('Dynamic Position Sizing (ATR-based)', fontsize=12)

# Subplot 4: 월별 수익률 히트맵
ax4 = fig.add_subplot(gs[3])
sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            ax=ax4, cbar_kws={'label': 'Monthly Return (%)'},
            linewidths=0.5, linecolor='gray')
ax4.set_ylabel('Year', fontsize=11)
ax4.set_xlabel('Month', fontsize=11)
ax4.set_title('Monthly Returns Heatmap (%)', fontsize=12)

plt.savefig('output/atr_leveraged_strategy.png', dpi=300, bbox_inches='tight')
print("시각화 저장: output/atr_leveraged_strategy.png")
plt.close()

# CSV 저장
summary_df = pd.DataFrame([{
    'Strategy': 'ATR_Leveraged_SMA30',
    'Total_Return_x': strategy_total_return,
    'CAGR_%': strategy_cagr * 100,
    'MDD_%': strategy_mdd,
    'Sharpe_Ratio': strategy_sharpe,
    'Avg_Position_Size': avg_position,
    'Max_Position_Size': max_position,
    'Benchmark_Return_x': benchmark_total_return,
    'Outperformance_%': outperformance,
    'Buy_Hold_Return_x': bh_total_return
}])
summary_df.to_csv('output/atr_leveraged_summary.csv', index=False)
print("성과 요약: output/atr_leveraged_summary.csv")

pivot_table.to_csv('output/atr_leveraged_monthly_returns.csv')
print("월별 수익률: output/atr_leveraged_monthly_returns.csv")

# 포지션 변경 로그
trades = df[abs(df['position_change']) > 0.01].copy()
trades_log = trades[['close', 'position_size', 'position_change', 'atr14']].copy()
trades_log.to_csv('output/atr_leveraged_position_changes.csv')
print(f"포지션 변경 로그: output/atr_leveraged_position_changes.csv ({len(trades_log)} 변경)")

print("\n" + "="*80)
print("백테스트 완료!")
print("="*80)
print(f"\n🎉 ATR Leveraged 전략이 벤치마크를 {outperformance:+,.1f}% 초과 달성!")
print(f"   최종 자산: 1원 → {strategy_total_return:,.0f}원")
