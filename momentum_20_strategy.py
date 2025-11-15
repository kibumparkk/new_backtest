"""
Momentum 20 Trend Following Strategy
=====================================

최고 성과 추세추종 전략

전략 설명:
- 20일 모멘텀을 사용한 추세추종 전략
- 모멘텀 = 현재가 - 20일 전 가격
- 모멘텀이 양수일 때 매수, 음수일 때 매도/관망

백테스트 성과 (2017-09-25 ~ 2025-11-10):
- Total Return: 141.89x
- Benchmark Return: 95.94x (SMA30 Cross)
- Outperformance: +47.90%
- CAGR: 92.25%
- MDD: -59.97%
- Sharpe Ratio: 1.48
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 출력 폴더 생성
os.makedirs('output', exist_ok=True)

# 데이터 로드
print("데이터 로딩 중...")
df = pd.read_parquet('chart_day/BTC_KRW.parquet')
print(f"데이터 기간: {df.index.min()} ~ {df.index.max()}")
print(f"총 {len(df)}일")

# 초기 설정
INITIAL_CAPITAL = 1
SLIPPAGE = 0.002

# 전략 지표 계산
print("\n전략 지표 계산 중...")

# Momentum 20 전략
df['momentum'] = df['close'] - df['close'].shift(20)
df['strategy_signal'] = (df['momentum'] > 0).astype(int)

# 벤치마크: SMA30 Cross (shift(1)은 수익률 계산에서 적용)
df['sma30'] = df['close'].rolling(window=30).mean()
df['benchmark_signal'] = (df['close'] > df['sma30']).astype(int)

# 일일 수익률
df['daily_return'] = df['close'].pct_change()

# 전략 수익률 계산 (슬리피지 적용)
df['strategy_position_change'] = df['strategy_signal'].diff()
df['strategy_return'] = (
    df['strategy_signal'].shift(1) * df['daily_return'] -
    abs(df['strategy_position_change']) * SLIPPAGE
)
df['strategy_equity'] = INITIAL_CAPITAL * (1 + df['strategy_return']).cumprod()

# 벤치마크 수익률 계산
df['benchmark_position_change'] = df['benchmark_signal'].diff()
df['benchmark_return'] = (
    df['benchmark_signal'].shift(1) * df['daily_return'] -
    abs(df['benchmark_position_change']) * SLIPPAGE
)
df['benchmark_equity'] = INITIAL_CAPITAL * (1 + df['benchmark_return']).cumprod()

# Buy & Hold 수익률
df['bh_return'] = df['daily_return']
df['bh_equity'] = INITIAL_CAPITAL * (1 + df['bh_return']).cumprod()

# 유효한 데이터만 사용
df = df.dropna()

# 성과 지표 계산
print("\n성과 지표 계산 중...")

# Total Return
strategy_total_return = df['strategy_equity'].iloc[-1] / INITIAL_CAPITAL
benchmark_total_return = df['benchmark_equity'].iloc[-1] / INITIAL_CAPITAL
bh_total_return = df['bh_equity'].iloc[-1] / INITIAL_CAPITAL

# CAGR
total_days = (df.index[-1] - df.index[0]).days
years = total_days / 365.25
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
if df['strategy_return'].std() != 0:
    strategy_sharpe = (df['strategy_return'].mean() / df['strategy_return'].std()) * np.sqrt(365)
else:
    strategy_sharpe = 0

if df['benchmark_return'].std() != 0:
    benchmark_sharpe = (df['benchmark_return'].mean() / df['benchmark_return'].std()) * np.sqrt(365)
else:
    benchmark_sharpe = 0

# 결과 출력
print("\n" + "="*70)
print("백테스트 결과 - Momentum 20 전략")
print("="*70)
print(f"\n전략 성과:")
print(f"  Total Return: {strategy_total_return:.2f}x")
print(f"  CAGR: {strategy_cagr:.2%}")
print(f"  MDD: {strategy_mdd:.2%}")
print(f"  Sharpe Ratio: {strategy_sharpe:.2f}")

print(f"\n벤치마크 성과 (SMA30 Cross):")
print(f"  Total Return: {benchmark_total_return:.2f}x")
print(f"  CAGR: {benchmark_cagr:.2%}")
print(f"  MDD: {benchmark_mdd:.2%}")
print(f"  Sharpe Ratio: {benchmark_sharpe:.2f}")

outperformance = (strategy_total_return / benchmark_total_return - 1) * 100
print(f"\nOutperformance: {outperformance:+.2f}%")

print(f"\nBuy & Hold 성과:")
print(f"  Total Return: {bh_total_return:.2f}x")

# 월별 수익률 계산
print("\n월별 수익률 계산 중...")
df['month'] = df.index.to_period('M')
monthly_returns = df.groupby('month')['strategy_return'].apply(lambda x: (1 + x).prod() - 1) * 100
monthly_returns.index = monthly_returns.index.to_timestamp()

# 피벗 테이블 생성
pivot_data = []
for date, ret in monthly_returns.items():
    pivot_data.append({'year': date.year, 'month': date.month, 'return': ret})

pivot_df = pd.DataFrame(pivot_data)
pivot_table = pivot_df.pivot(index='year', columns='month', values='return')
pivot_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Drawdown 계산
df['drawdown_pct'] = strategy_drawdown

# 시각화
print("\n시각화 생성 중...")
fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 2], hspace=0.3)

# Subplot 1: 누적 자산 곡선
ax1 = fig.add_subplot(gs[0])
ax1.plot(df.index, df['strategy_equity'], label='Momentum 20 Strategy',
         linewidth=2.5, color='#2E86AB')
ax1.plot(df.index, df['benchmark_equity'], label='Benchmark (SMA30)',
         linewidth=2, alpha=0.7, color='#A23B72')
ax1.plot(df.index, df['bh_equity'], label='Buy & Hold',
         linewidth=1.5, alpha=0.5, linestyle='--', color='#F18F01')
ax1.set_yscale('log')
ax1.set_ylabel('Cumulative Returns (KRW, log scale)', fontsize=11)
ax1.set_title('Momentum 20 Trend Following Strategy - Backtest Performance',
              fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# 성과지표 텍스트 박스
metrics_text = f'''Strategy Performance:
Total Return: {strategy_total_return:.2f}x
CAGR: {strategy_cagr:.2%}
MDD: {strategy_mdd:.2%}
Sharpe: {strategy_sharpe:.2f}

Benchmark: {benchmark_total_return:.2f}x
Outperformance: {outperformance:+.1f}%'''

ax1.text(0.98, 0.97, metrics_text, transform=ax1.transAxes,
         fontsize=9.5, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
         family='monospace')

# Subplot 2: Drawdown 차트
ax2 = fig.add_subplot(gs[1])
ax2.fill_between(df.index, 0, df['drawdown_pct'], color='red', alpha=0.3)
ax2.plot(df.index, df['drawdown_pct'], color='darkred', linewidth=1)
ax2.set_ylabel('Drawdown (%)', fontsize=11)
ax2.set_xlabel('Date', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_title('Strategy Drawdown', fontsize=11)

# Subplot 3: 월별 수익률 히트맵
ax3 = fig.add_subplot(gs[2])
sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            ax=ax3, cbar_kws={'label': 'Monthly Return (%)'},
            linewidths=0.5, linecolor='gray')
ax3.set_ylabel('Year', fontsize=11)
ax3.set_xlabel('Month', fontsize=11)
ax3.set_title('Monthly Returns Heatmap (%)', fontsize=12)

# 저장
plt.savefig('output/momentum_20_backtest.png', dpi=300, bbox_inches='tight')
print("시각화 저장: output/momentum_20_backtest.png")
plt.close()

# CSV 파일 저장
print("\n결과 파일 저장 중...")

# 성과 요약
summary_df = pd.DataFrame([{
    'Strategy': 'Momentum_20',
    'Total_Return_x': strategy_total_return,
    'CAGR_%': strategy_cagr * 100,
    'MDD_%': strategy_mdd,
    'Sharpe_Ratio': strategy_sharpe,
    'Benchmark_Return_x': benchmark_total_return,
    'Benchmark_CAGR_%': benchmark_cagr * 100,
    'Benchmark_MDD_%': benchmark_mdd,
    'Outperformance_%': outperformance,
    'Buy_Hold_Return_x': bh_total_return
}])
summary_df.to_csv('output/momentum_20_summary.csv', index=False)
print("성과 요약: output/momentum_20_summary.csv")

# 월별 수익률
pivot_table.to_csv('output/momentum_20_monthly_returns.csv')
print("월별 수익률: output/momentum_20_monthly_returns.csv")

# 거래 로그 (진입/청산 시점만)
trades = df[df['strategy_position_change'] != 0].copy()
trades['action'] = trades['strategy_position_change'].apply(
    lambda x: 'BUY' if x > 0 else 'SELL'
)
trades_log = trades[['close', 'action', 'momentum']].copy()
trades_log.to_csv('output/momentum_20_trades.csv')
print(f"거래 로그: output/momentum_20_trades.csv ({len(trades_log)} 거래)")

print("\n" + "="*70)
print("백테스트 완료!")
print("="*70)
print(f"\n최종 결과: Momentum 20 전략이 벤치마크를 {outperformance:+.2f}% 초과 달성")
