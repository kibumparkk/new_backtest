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

print("=" * 80)
print("멀티윈도우 스코어 임계값 캘리브레이션")
print("=" * 80)
print(f"데이터 기간: {df.index.min()} ~ {df.index.max()}")
print(f"총 {len(df)}일")
print()

# === 멀티윈도우 20개 설정 ===
windows = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
print(f"윈도우 개수: {len(windows)}개")
print(f"윈도우 기간: {windows}")
print()

# === 기본 계산 ===
df['returns'] = df['close'].pct_change()

# 각 윈도우에 대해 SMA 계산 및 스코어 부여
scores = pd.DataFrame(index=df.index)

for window in windows:
    sma_col = f'sma_{window}'
    score_col = f'score_{window}'
    df[sma_col] = df['close'].rolling(window=window).mean()
    scores[score_col] = (df['close'] > df[sma_col]).astype(int)

# 전체 스코어 평균 계산 (0~1 사이)
df['multi_window_score'] = scores.mean(axis=1)

# === 벤치마크 전략 (SMA30) ===
df['sma30'] = df['close'].rolling(window=30).mean()
df['benchmark_signal'] = (df['close'].shift(1) > df['sma30'].shift(1)).astype(int)
df['benchmark_returns'] = df['benchmark_signal'] * df['returns']
df['benchmark_equity'] = INITIAL_CAPITAL * (1 + df['benchmark_returns']).cumprod()

# === 다양한 임계값으로 백테스트 ===
thresholds = np.arange(0.0, 1.05, 0.05)  # 0.0, 0.05, 0.10, ..., 1.00
results = []

print("=" * 80)
print("임계값별 백테스트 진행 중...")
print("=" * 80)

for threshold in thresholds:
    # 전략 시그널 생성
    signal = (df['multi_window_score'].shift(1) >= threshold).astype(int)

    # 전략 수익률 계산
    strategy_returns = signal * df['returns']

    # 슬리피지 적용
    position_change = signal.diff().abs()
    slippage_cost = position_change * SLIPPAGE
    strategy_returns_with_slippage = strategy_returns - slippage_cost

    # 자산 곡선 계산
    equity = INITIAL_CAPITAL * (1 + strategy_returns_with_slippage).cumprod()
    valid_equity = equity.dropna()

    if len(valid_equity) == 0:
        continue

    # 성과 지표 계산
    total_return = valid_equity.iloc[-1] / INITIAL_CAPITAL

    # CAGR
    start_date = valid_equity.index[0]
    end_date = valid_equity.index[-1]
    total_days = (end_date - start_date).days
    years = total_days / 365.25
    cagr = (valid_equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1

    # MDD
    cummax = valid_equity.cummax()
    drawdown = (valid_equity - cummax) / cummax
    mdd = drawdown.min()

    # Sharpe Ratio
    daily_returns = strategy_returns_with_slippage.dropna()
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else 0

    # 거래 일수
    trading_days = (signal != 0).sum()

    results.append({
        'Threshold': threshold,
        'Total Return': total_return,
        'CAGR': cagr,
        'MDD': mdd,
        'Sharpe Ratio': sharpe_ratio,
        'Trading Days': trading_days,
        'Equity': equity
    })

    print(f"Threshold: {threshold:.2f} | Total Return: {total_return:>8.2f}x | CAGR: {cagr:>7.2%} | MDD: {mdd:>7.2%} | Sharpe: {sharpe_ratio:>5.2f}")

# 결과 DataFrame 생성
results_df = pd.DataFrame(results)

# 최적값 및 최저값 찾기
best_idx = results_df['Total Return'].idxmax()
worst_idx = results_df['Total Return'].idxmin()
best_sharpe_idx = results_df['Sharpe Ratio'].idxmax()
best_mdd_idx = results_df['MDD'].idxmax()  # MDD는 음수이므로 max가 최소 손실

print()
print("=" * 80)
print("캘리브레이션 결과")
print("=" * 80)
print()

print("📈 최고 수익률 (Total Return 기준):")
print(f"  - Threshold: {results_df.loc[best_idx, 'Threshold']:.2f}")
print(f"  - Total Return: {results_df.loc[best_idx, 'Total Return']:.2f}x")
print(f"  - CAGR: {results_df.loc[best_idx, 'CAGR']:.2%}")
print(f"  - MDD: {results_df.loc[best_idx, 'MDD']:.2%}")
print(f"  - Sharpe Ratio: {results_df.loc[best_idx, 'Sharpe Ratio']:.2f}")
print()

print("📉 최저 수익률 (Total Return 기준):")
print(f"  - Threshold: {results_df.loc[worst_idx, 'Threshold']:.2f}")
print(f"  - Total Return: {results_df.loc[worst_idx, 'Total Return']:.2f}x")
print(f"  - CAGR: {results_df.loc[worst_idx, 'CAGR']:.2%}")
print(f"  - MDD: {results_df.loc[worst_idx, 'MDD']:.2%}")
print(f"  - Sharpe Ratio: {results_df.loc[worst_idx, 'Sharpe Ratio']:.2f}")
print()

print("⚡ 최고 샤프 비율:")
print(f"  - Threshold: {results_df.loc[best_sharpe_idx, 'Threshold']:.2f}")
print(f"  - Total Return: {results_df.loc[best_sharpe_idx, 'Total Return']:.2f}x")
print(f"  - CAGR: {results_df.loc[best_sharpe_idx, 'CAGR']:.2%}")
print(f"  - MDD: {results_df.loc[best_sharpe_idx, 'MDD']:.2%}")
print(f"  - Sharpe Ratio: {results_df.loc[best_sharpe_idx, 'Sharpe Ratio']:.2f}")
print()

print("🛡️  최소 낙폭 (Best MDD):")
print(f"  - Threshold: {results_df.loc[best_mdd_idx, 'Threshold']:.2f}")
print(f"  - Total Return: {results_df.loc[best_mdd_idx, 'Total Return']:.2f}x")
print(f"  - CAGR: {results_df.loc[best_mdd_idx, 'CAGR']:.2%}")
print(f"  - MDD: {results_df.loc[best_mdd_idx, 'MDD']:.2%}")
print(f"  - Sharpe Ratio: {results_df.loc[best_mdd_idx, 'Sharpe Ratio']:.2f}")
print()

# 벤치마크와 비교
benchmark_total_return = df['benchmark_equity'].iloc[-1] / INITIAL_CAPITAL
print(f"📊 벤치마크 (SMA30): {benchmark_total_return:.2f}x")
print()

# CSV 저장
calibration_results = results_df.drop('Equity', axis=1)
calibration_results.to_csv('output/threshold_calibration.csv', index=False, encoding='utf-8-sig')
print("임계값 캘리브레이션 결과 저장: output/threshold_calibration.csv")

# === 시각화 ===
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Total Return vs Threshold
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(results_df['Threshold'], results_df['Total Return'], marker='o', linewidth=2, markersize=4, color='#2E86AB')
ax1.axhline(y=benchmark_total_return, color='red', linestyle='--', linewidth=2, label=f'Benchmark (SMA30): {benchmark_total_return:.2f}x', alpha=0.7)
ax1.scatter(results_df.loc[best_idx, 'Threshold'], results_df.loc[best_idx, 'Total Return'],
            color='green', s=200, zorder=5, marker='*', label=f'Best: {results_df.loc[best_idx, "Threshold"]:.2f}')
ax1.scatter(results_df.loc[worst_idx, 'Threshold'], results_df.loc[worst_idx, 'Total Return'],
            color='red', s=200, zorder=5, marker='X', label=f'Worst: {results_df.loc[worst_idx, "Threshold"]:.2f}')
ax1.set_xlabel('Threshold', fontsize=11)
ax1.set_ylabel('Total Return (x)', fontsize=11)
ax1.set_title('Total Return vs Threshold', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# 2. CAGR vs Threshold
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(results_df['Threshold'], results_df['CAGR'] * 100, marker='o', linewidth=2, markersize=4, color='#A23B72')
ax2.scatter(results_df.loc[best_idx, 'Threshold'], results_df.loc[best_idx, 'CAGR'] * 100,
            color='green', s=200, zorder=5, marker='*')
ax2.scatter(results_df.loc[worst_idx, 'Threshold'], results_df.loc[worst_idx, 'CAGR'] * 100,
            color='red', s=200, zorder=5, marker='X')
ax2.set_xlabel('Threshold', fontsize=11)
ax2.set_ylabel('CAGR (%)', fontsize=11)
ax2.set_title('CAGR vs Threshold', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. MDD vs Threshold
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(results_df['Threshold'], results_df['MDD'] * 100, marker='o', linewidth=2, markersize=4, color='#F18F01')
ax3.scatter(results_df.loc[best_mdd_idx, 'Threshold'], results_df.loc[best_mdd_idx, 'MDD'] * 100,
            color='green', s=200, zorder=5, marker='*', label=f'Best MDD: {results_df.loc[best_mdd_idx, "Threshold"]:.2f}')
ax3.set_xlabel('Threshold', fontsize=11)
ax3.set_ylabel('MDD (%)', fontsize=11)
ax3.set_title('Maximum Drawdown vs Threshold', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9)

# 4. Sharpe Ratio vs Threshold
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(results_df['Threshold'], results_df['Sharpe Ratio'], marker='o', linewidth=2, markersize=4, color='#6A994E')
ax4.scatter(results_df.loc[best_sharpe_idx, 'Threshold'], results_df.loc[best_sharpe_idx, 'Sharpe Ratio'],
            color='green', s=200, zorder=5, marker='*', label=f'Best Sharpe: {results_df.loc[best_sharpe_idx, "Threshold"]:.2f}')
ax4.set_xlabel('Threshold', fontsize=11)
ax4.set_ylabel('Sharpe Ratio', fontsize=11)
ax4.set_title('Sharpe Ratio vs Threshold', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=9)

# 5. 최고/최저/벤치마크 누적 수익률 비교
ax5 = fig.add_subplot(gs[2, :])
ax5.plot(df.index, results[best_idx]['Equity'], label=f'Best (Threshold={results_df.loc[best_idx, "Threshold"]:.2f})',
         linewidth=2, color='green')
ax5.plot(df.index, results[worst_idx]['Equity'], label=f'Worst (Threshold={results_df.loc[worst_idx, "Threshold"]:.2f})',
         linewidth=2, color='red', alpha=0.7)
ax5.plot(df.index, df['benchmark_equity'], label='Benchmark (SMA30)',
         linewidth=2, color='blue', alpha=0.5, linestyle='--')
ax5.set_yscale('log')
ax5.set_xlabel('Date', fontsize=11)
ax5.set_ylabel('Cumulative Returns (KRW, log scale)', fontsize=11)
ax5.set_title('Equity Curves Comparison: Best vs Worst vs Benchmark', fontsize=12, fontweight='bold')
ax5.legend(loc='upper left', fontsize=10)
ax5.grid(True, alpha=0.3)

plt.suptitle('Multi-Window Score Threshold Calibration Analysis (20 Windows)',
             fontsize=14, fontweight='bold', y=0.995)

# 저장
plt.savefig('output/threshold_calibration.png', dpi=300, bbox_inches='tight')
plt.close()
print("시각화 저장: output/threshold_calibration.png")

# === 상세 결과 표 출력 ===
print()
print("=" * 80)
print("상세 결과 테이블")
print("=" * 80)
print(calibration_results.to_string(index=False))
print("=" * 80)
