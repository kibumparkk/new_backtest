import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# 출력 폴더 생성
os.makedirs('output', exist_ok=True)

# 데이터 로드
df = pd.read_parquet('chart_day/BTC_KRW.parquet')

# 초기 설정
INITIAL_CAPITAL = 1  # 1원
SLIPPAGE = 0.002     # 0.2%

# 데이터 확인
print("=" * 60)
print("데이터 정보")
print("=" * 60)
print(f"데이터 기간: {df.index.min()} ~ {df.index.max()}")
print(f"총 {len(df)}일")
print(f"컬럼: {list(df.columns)}")
print()

# === 멀티윈도우 20개 설정 ===
# 5일부터 100일까지 5일 간격으로 20개 윈도우
windows = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
print("=" * 60)
print("멀티윈도우 전략 설정")
print("=" * 60)
print(f"윈도우 개수: {len(windows)}개")
print(f"윈도우 기간: {windows}")
print()

# === 벤치마크 전략 구현 (SMA30) ===
df['sma30'] = df['close'].rolling(window=30).mean()
df['benchmark_signal'] = (df['close'].shift(1) > df['sma30'].shift(1)).astype(int)
df['returns'] = df['close'].pct_change()
df['benchmark_returns'] = df['benchmark_signal'] * df['returns']
df['benchmark_equity'] = INITIAL_CAPITAL * (1 + df['benchmark_returns']).cumprod()

# === 멀티윈도우 스코어 계산 ===
# 각 윈도우에 대해 SMA 계산 및 스코어 부여
scores = pd.DataFrame(index=df.index)

for window in windows:
    sma_col = f'sma_{window}'
    score_col = f'score_{window}'

    # SMA 계산
    df[sma_col] = df['close'].rolling(window=window).mean()

    # 스코어: close > SMA이면 1, 아니면 0
    scores[score_col] = (df['close'] > df[sma_col]).astype(int)

# 전체 스코어 평균 계산 (0~1 사이)
df['multi_window_score'] = scores.mean(axis=1)

print("=" * 60)
print("스코어 통계")
print("=" * 60)
print(f"스코어 범위: {df['multi_window_score'].min():.3f} ~ {df['multi_window_score'].max():.3f}")
print(f"스코어 평균: {df['multi_window_score'].mean():.3f}")
print(f"스코어 중앙값: {df['multi_window_score'].median():.3f}")
print()

# === 전략 시그널 생성 ===
# 전일 스코어가 0.5 이상이면 당일 매수
df['strategy_signal'] = (df['multi_window_score'].shift(1) >= 0.5).astype(int)

# 전략 수익률 계산
df['strategy_returns'] = df['strategy_signal'] * df['returns']

# 슬리피지 적용
# 포지션 변화가 있을 때만 슬리피지 적용
df['position_change'] = df['strategy_signal'].diff().abs()
df['slippage_cost'] = df['position_change'] * SLIPPAGE
df['strategy_returns_with_slippage'] = df['strategy_returns'] - df['slippage_cost']

# 자산 곡선 계산
df['strategy_equity'] = INITIAL_CAPITAL * (1 + df['strategy_returns_with_slippage']).cumprod()

# Buy & Hold 전략
df['buyhold_returns'] = df['returns']
df['buyhold_equity'] = INITIAL_CAPITAL * (1 + df['buyhold_returns']).cumprod()

# === 성과 지표 계산 ===
def calculate_metrics(equity_series, name="Strategy"):
    """성과 지표 계산"""
    # 유효한 데이터만 사용
    valid_equity = equity_series.dropna()

    if len(valid_equity) == 0:
        return {}

    # Total Return (배수)
    total_return = valid_equity.iloc[-1] / INITIAL_CAPITAL

    # CAGR
    start_date = valid_equity.index[0]
    end_date = valid_equity.index[-1]
    total_days = (end_date - start_date).days
    years = total_days / 365.25
    cagr = (valid_equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1

    # MDD 계산
    cummax = valid_equity.cummax()
    drawdown = (valid_equity - cummax) / cummax
    mdd = drawdown.min()

    # Sharpe Ratio (일별 수익률 기준)
    if name == "Strategy":
        daily_returns = df['strategy_returns_with_slippage'].dropna()
    elif name == "Benchmark":
        daily_returns = df['benchmark_returns'].dropna()
    else:  # Buy & Hold
        daily_returns = df['buyhold_returns'].dropna()

    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else 0

    # Win Rate
    winning_days = (daily_returns > 0).sum()
    total_trading_days = len(daily_returns[daily_returns != 0])
    win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0

    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'MDD': mdd,
        'Sharpe Ratio': sharpe_ratio,
        'Win Rate': win_rate,
        'Final Equity': valid_equity.iloc[-1]
    }

# 각 전략 성과 계산
strategy_metrics = calculate_metrics(df['strategy_equity'], "Strategy")
benchmark_metrics = calculate_metrics(df['benchmark_equity'], "Benchmark")
buyhold_metrics = calculate_metrics(df['buyhold_equity'], "Buy & Hold")

print("=" * 60)
print("성과 지표 비교")
print("=" * 60)
print(f"\n{'Metric':<20} {'Multi-Window':<15} {'Benchmark':<15} {'Buy & Hold':<15}")
print("-" * 65)
print(f"{'Total Return':<20} {strategy_metrics['Total Return']:<15.2f}x {benchmark_metrics['Total Return']:<15.2f}x {buyhold_metrics['Total Return']:<15.2f}x")
print(f"{'CAGR':<20} {strategy_metrics['CAGR']:<15.2%} {benchmark_metrics['CAGR']:<15.2%} {buyhold_metrics['CAGR']:<15.2%}")
print(f"{'MDD':<20} {strategy_metrics['MDD']:<15.2%} {benchmark_metrics['MDD']:<15.2%} {buyhold_metrics['MDD']:<15.2%}")
print(f"{'Sharpe Ratio':<20} {strategy_metrics['Sharpe Ratio']:<15.2f} {benchmark_metrics['Sharpe Ratio']:<15.2f} {buyhold_metrics['Sharpe Ratio']:<15.2f}")
print(f"{'Win Rate':<20} {strategy_metrics['Win Rate']:<15.2%} {benchmark_metrics['Win Rate']:<15.2%} {buyhold_metrics['Win Rate']:<15.2%}")
print()

# === 성과 요약 CSV 저장 ===
performance_df = pd.DataFrame({
    'Strategy': ['Multi-Window Score', 'Benchmark (SMA30)', 'Buy & Hold'],
    'Total Return (x)': [strategy_metrics['Total Return'], benchmark_metrics['Total Return'], buyhold_metrics['Total Return']],
    'CAGR (%)': [strategy_metrics['CAGR'] * 100, benchmark_metrics['CAGR'] * 100, buyhold_metrics['CAGR'] * 100],
    'MDD (%)': [strategy_metrics['MDD'] * 100, benchmark_metrics['MDD'] * 100, buyhold_metrics['MDD'] * 100],
    'Sharpe Ratio': [strategy_metrics['Sharpe Ratio'], benchmark_metrics['Sharpe Ratio'], buyhold_metrics['Sharpe Ratio']],
    'Win Rate (%)': [strategy_metrics['Win Rate'] * 100, benchmark_metrics['Win Rate'] * 100, buyhold_metrics['Win Rate'] * 100],
    'Final Equity (KRW)': [strategy_metrics['Final Equity'], benchmark_metrics['Final Equity'], buyhold_metrics['Final Equity']]
})
performance_df.to_csv('output/performance_summary.csv', index=False, encoding='utf-8-sig')
print("성과 요약 저장: output/performance_summary.csv")

# === 월별 수익률 계산 ===
monthly_returns = df['strategy_returns_with_slippage'].resample('M').apply(lambda x: (1 + x).prod() - 1)
monthly_returns_df = pd.DataFrame({
    'Date': monthly_returns.index,
    'Monthly Return (%)': monthly_returns.values * 100
})
monthly_returns_df.to_csv('output/monthly_returns.csv', index=False, encoding='utf-8-sig')
print("월별 수익률 저장: output/monthly_returns.csv")

# === 시각화 ===
fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(4, 1, height_ratios=[2.5, 1, 1.5, 2], hspace=0.3)

# Subplot 1: 누적 자산 곡선 (로그 스케일)
ax1 = fig.add_subplot(gs[0])
ax1.plot(df.index, df['strategy_equity'], label='Multi-Window Strategy (Score >= 0.5)', linewidth=2, color='#2E86AB')
ax1.plot(df.index, df['benchmark_equity'], label='Benchmark (SMA30)', linewidth=2, alpha=0.7, color='#A23B72')
ax1.plot(df.index, df['buyhold_equity'], label='Buy & Hold', linewidth=2, alpha=0.5, color='#F18F01', linestyle='--')
ax1.set_yscale('log')
ax1.set_ylabel('Cumulative Returns (KRW, log scale)', fontsize=12)
ax1.set_title('Multi-Window Score Strategy Backtest Performance (20 Windows, Threshold = 0.5)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# 성과지표 텍스트 박스
metrics_text = f'''Multi-Window Strategy
Total Return: {strategy_metrics['Total Return']:.2f}x
CAGR: {strategy_metrics['CAGR']:.2%}
MDD: {strategy_metrics['MDD']:.2%}
Sharpe: {strategy_metrics['Sharpe Ratio']:.2f}
Win Rate: {strategy_metrics['Win Rate']:.2%}'''

ax1.text(0.98, 0.03, metrics_text, transform=ax1.transAxes,
         fontsize=9, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

# Subplot 2: 멀티윈도우 스코어 시계열
ax2 = fig.add_subplot(gs[1])
ax2.plot(df.index, df['multi_window_score'], linewidth=1, color='#2E86AB', alpha=0.7)
ax2.axhline(y=0.5, color='red', linewidth=2, linestyle='--', label='Threshold (0.5)', alpha=0.8)
ax2.fill_between(df.index, 0, df['multi_window_score'], where=(df['multi_window_score'] >= 0.5),
                  alpha=0.3, color='green', label='Buy Signal')
ax2.fill_between(df.index, 0, df['multi_window_score'], where=(df['multi_window_score'] < 0.5),
                  alpha=0.3, color='red', label='Sell Signal')
ax2.set_ylabel('Score', fontsize=11)
ax2.set_ylim(-0.05, 1.05)
ax2.set_title('Multi-Window Score (Average of 20 Windows)', fontsize=12)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

# Subplot 3: Drawdown 차트 (%)
ax3 = fig.add_subplot(gs[2])
cummax = df['strategy_equity'].cummax()
drawdown_pct = (df['strategy_equity'] - cummax) / cummax * 100
ax3.fill_between(df.index, 0, drawdown_pct, color='red', alpha=0.3)
ax3.plot(df.index, drawdown_pct, color='red', linewidth=1)
ax3.set_ylabel('Drawdown (%)', fontsize=11)
ax3.set_xlabel('Date', fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='black', linewidth=0.5)
ax3.set_title(f'Drawdown (MDD: {strategy_metrics["MDD"]:.2%})', fontsize=12)

# Subplot 4: 월별 수익률 히트맵
ax4 = fig.add_subplot(gs[3])

# 월별 수익률을 피벗 테이블로 변환
monthly_rets = df['strategy_returns_with_slippage'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
if len(monthly_rets) > 0:
    monthly_pivot = pd.DataFrame({
        'Year': monthly_rets.index.year,
        'Month': monthly_rets.index.month,
        'Return': monthly_rets.values
    })
    pivot_table = monthly_pivot.pivot(index='Year', columns='Month', values='Return')

    # 월 이름으로 컬럼명 변경
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot_table.columns = [month_names[int(m)-1] if m in pivot_table.columns else m for m in range(1, 13)]
    pivot_table = pivot_table.reindex(columns=month_names, fill_value=np.nan)

    # 히트맵 그리기
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                ax=ax4, cbar_kws={'label': 'Monthly Return (%)'},
                linewidths=0.5, linecolor='gray')
    ax4.set_ylabel('Year', fontsize=11)
    ax4.set_xlabel('Month', fontsize=11)
    ax4.set_title('Monthly Returns Heatmap (%)', fontsize=12)

# 저장
plt.savefig('output/backtest_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("시각화 저장: output/backtest_results.png")
print()

print("=" * 60)
print("백테스트 완료!")
print("=" * 60)
print(f"전략: 멀티윈도우 스코어 (20개 윈도우)")
print(f"윈도우: {windows}")
print(f"매수 조건: 전일 스코어 >= 0.5")
print(f"Total Return: {strategy_metrics['Total Return']:.2f}x")
print(f"CAGR: {strategy_metrics['CAGR']:.2%}")
print(f"MDD: {strategy_metrics['MDD']:.2%}")
print(f"Sharpe Ratio: {strategy_metrics['Sharpe Ratio']:.2f}")
print(f"최종 자산: {strategy_metrics['Final Equity']:,.4f}원")
print()
print("결과 파일:")
print("  - output/backtest_results.png")
print("  - output/performance_summary.csv")
print("  - output/monthly_returns.csv")
print("=" * 60)
