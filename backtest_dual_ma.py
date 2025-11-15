import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# 출력 폴더 생성
os.makedirs('output', exist_ok=True)

# 데이터 로드
print("=" * 80)
print("이동평균선 매수/매도 분리 전략 백테스트")
print("=" * 80)
df = pd.read_parquet('chart_day/BTC_KRW.parquet')

# 초기 설정
INITIAL_CAPITAL = 1  # 1원
SLIPPAGE = 0.002     # 0.2%

# 데이터 확인
print(f"\n데이터 기간: {df.index.min()} ~ {df.index.max()}")
print(f"총 {len(df)}일")
print(f"컬럼: {list(df.columns)}")

# === 벤치마크 전략 구현 (SMA30) ===
print("\n" + "=" * 80)
print("1. 벤치마크 전략 (전일종가 > SMA30)")
print("=" * 80)

# SMA 계산
df['sma30'] = df['close'].rolling(window=30).mean()

# 시그널 생성: 전일 데이터로 당일 포지션 결정
df['benchmark_signal'] = (df['close'].shift(1) > df['sma30'].shift(1)).astype(int)

# 수익률 계산
df['returns'] = df['close'].pct_change()

# 벤치마크 전략 수익률 (슬리피지 적용)
df['benchmark_returns'] = df['benchmark_signal'] * df['returns']

# 포지션 변화 감지
df['benchmark_position_change'] = df['benchmark_signal'].diff().abs()

# 슬리피지 적용: 포지션 변경 시 -0.2%
df['benchmark_slippage'] = -SLIPPAGE * df['benchmark_position_change']
df['benchmark_returns_with_slippage'] = df['benchmark_returns'] + df['benchmark_slippage']

# 벤치마크 자산 곡선 계산
df['benchmark_equity'] = INITIAL_CAPITAL * (1 + df['benchmark_returns_with_slippage']).cumprod()

# === 매수/매도 분리 전략 탐색 ===
print("\n" + "=" * 80)
print("2. 매수/매도 분리 전략 최적화")
print("=" * 80)

# 다양한 MA 조합 테스트
best_return = 0
best_params = None
best_results = None

# 매수 MA: 5, 10, 15, 20, 25
# 매도 MA: 30, 40, 50, 60, 70
buy_mas = [5, 10, 15, 20, 25]
sell_mas = [30, 40, 50, 60, 70]

print("\n매수/매도 MA 조합 테스트:")
print("-" * 80)
print(f"{'매수MA':<8} {'매도MA':<8} {'Total Return':<15} {'CAGR':<10} {'MDD':<10}")
print("-" * 80)

all_results = []

for buy_ma in buy_mas:
    for sell_ma in sell_mas:
        # 이동평균 계산
        df[f'sma_buy_{buy_ma}'] = df['close'].rolling(window=buy_ma).mean()
        df[f'sma_sell_{sell_ma}'] = df['close'].rolling(window=sell_ma).mean()

        # 매수/매도 시그널 생성
        # 매수 조건: 전일 종가가 전일 매수MA보다 위
        # 매도 조건: 전일 종가가 전일 매도MA보다 아래

        # 포지션 추적
        position = pd.Series(0, index=df.index)

        for i in range(1, len(df)):
            # 전일 데이터로 판단
            prev_close = df['close'].iloc[i-1]
            prev_buy_ma = df[f'sma_buy_{buy_ma}'].iloc[i-1]
            prev_sell_ma = df[f'sma_sell_{sell_ma}'].iloc[i-1]

            # 이전 포지션
            prev_position = position.iloc[i-1]

            # 매수 시그널: 현재 포지션이 0이고, 종가 > 매수MA
            if prev_position == 0 and not pd.isna(prev_buy_ma):
                if prev_close > prev_buy_ma:
                    position.iloc[i] = 1  # 매수
                else:
                    position.iloc[i] = 0
            # 매도 시그널: 현재 포지션이 1이고, 종가 < 매도MA
            elif prev_position == 1 and not pd.isna(prev_sell_ma):
                if prev_close < prev_sell_ma:
                    position.iloc[i] = 0  # 매도
                else:
                    position.iloc[i] = 1
            else:
                position.iloc[i] = prev_position

        # 전략 수익률 계산
        strategy_returns = position * df['returns']

        # 포지션 변화 감지 (슬리피지 적용)
        position_change = position.diff().abs()
        slippage_cost = -SLIPPAGE * position_change
        strategy_returns_with_slippage = strategy_returns + slippage_cost

        # 자산 곡선
        equity = INITIAL_CAPITAL * (1 + strategy_returns_with_slippage).cumprod()

        # 성과 지표 계산
        total_return = equity.iloc[-1] / INITIAL_CAPITAL

        # CAGR
        total_days = (df.index[-1] - df.index[0]).days
        years = total_days / 365.25
        cagr = (total_return) ** (1 / years) - 1

        # MDD
        cumulative = equity
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        mdd = drawdown.min()

        # 결과 저장
        result = {
            'buy_ma': buy_ma,
            'sell_ma': sell_ma,
            'total_return': total_return,
            'cagr': cagr,
            'mdd': mdd,
            'equity': equity,
            'position': position,
            'returns': strategy_returns_with_slippage
        }
        all_results.append(result)

        print(f"{buy_ma:<8} {sell_ma:<8} {total_return:<15.2f}x {cagr:<10.2%} {mdd:<10.2%}")

        # 최고 성과 추적
        if total_return > best_return:
            best_return = total_return
            best_params = (buy_ma, sell_ma)
            best_results = result

print("-" * 80)
print(f"\n최적 전략: 매수MA={best_params[0]}, 매도MA={best_params[1]}")
print(f"Total Return: {best_return:.2f}x")

# 벤치마크 성과 계산
bench_total_return = df['benchmark_equity'].iloc[-1] / INITIAL_CAPITAL
bench_total_days = (df.index[-1] - df.index[0]).days
bench_years = bench_total_days / 365.25
bench_cagr = (bench_total_return) ** (1 / bench_years) - 1
bench_cumulative = df['benchmark_equity']
bench_running_max = bench_cumulative.cummax()
bench_drawdown = (bench_cumulative - bench_running_max) / bench_running_max
bench_mdd = bench_drawdown.min()

print("\n" + "=" * 80)
print("3. 벤치마크 vs 최적 전략 비교")
print("=" * 80)
print(f"\n{'지표':<20} {'벤치마크 (SMA30)':<25} {'최적 전략':<25}")
print("-" * 80)
print(f"{'Total Return':<20} {bench_total_return:<25.2f}x {best_return:<25.2f}x")
print(f"{'CAGR':<20} {bench_cagr:<25.2%} {best_results['cagr']:<25.2%}")
print(f"{'MDD':<20} {bench_mdd:<25.2%} {best_results['mdd']:<25.2%}")
print("-" * 80)

improvement = ((best_return - bench_total_return) / bench_total_return) * 100
print(f"\n개선율: {improvement:+.2f}%")

# === 성과 지표 계산 ===
print("\n" + "=" * 80)
print("4. 상세 성과 분석")
print("=" * 80)

# 최적 전략 데이터
df['strategy_signal'] = best_results['position']
df['strategy_returns'] = best_results['returns']
df['strategy_equity'] = best_results['equity']

# Drawdown 계산
df['strategy_drawdown'] = (df['strategy_equity'] - df['strategy_equity'].cummax()) / df['strategy_equity'].cummax()
df['benchmark_drawdown'] = bench_drawdown

# 월별 수익률 계산
monthly_strategy = df['strategy_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
monthly_benchmark = df['benchmark_returns_with_slippage'].resample('M').apply(lambda x: (1 + x).prod() - 1)

# 승률 계산
strategy_trades = df['strategy_signal'].diff().abs()
winning_trades = ((df['strategy_returns'] > 0) & (strategy_trades > 0)).sum()
total_trades = (strategy_trades > 0).sum()
win_rate = winning_trades / total_trades if total_trades > 0 else 0

# Sharpe Ratio
strategy_sharpe = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(365) if df['strategy_returns'].std() > 0 else 0
benchmark_sharpe = df['benchmark_returns_with_slippage'].mean() / df['benchmark_returns_with_slippage'].std() * np.sqrt(365) if df['benchmark_returns_with_slippage'].std() > 0 else 0

print(f"\n최적 전략 상세:")
print(f"  - 매수 MA: {best_params[0]}")
print(f"  - 매도 MA: {best_params[1]}")
print(f"  - 총 거래 횟수: {total_trades}")
print(f"  - 승률: {win_rate:.2%}")
print(f"  - Sharpe Ratio: {strategy_sharpe:.2f}")
print(f"\n벤치마크:")
print(f"  - Sharpe Ratio: {benchmark_sharpe:.2f}")

# === 시각화 ===
print("\n" + "=" * 80)
print("5. 시각화 생성")
print("=" * 80)

# Figure 생성: 3개의 subplot
fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 2], hspace=0.35)

# Subplot 1: 누적 자산 곡선 (로그 스케일)
ax1 = fig.add_subplot(gs[0])
ax1.plot(df.index, df['strategy_equity'], label=f'Strategy (Buy MA={best_params[0]}, Sell MA={best_params[1]})', linewidth=2, color='#2E86AB')
ax1.plot(df.index, df['benchmark_equity'], label='Benchmark (SMA30)', linewidth=2, alpha=0.7, color='#A23B72')
ax1.set_yscale('log')
ax1.set_ylabel('Cumulative Returns (KRW, log scale)', fontsize=12, fontweight='bold')
ax1.set_title('이동평균선 매수/매도 분리 전략 백테스트', fontsize=16, fontweight='bold', pad=20)
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(True, alpha=0.3, linestyle='--')

# 성과지표 텍스트 박스
metrics_text = f'''[최적 전략]
Total Return: {best_return:.2f}x
CAGR: {best_results['cagr']:.2%}
MDD: {best_results['mdd']:.2%}
Sharpe: {strategy_sharpe:.2f}
Win Rate: {win_rate:.2%}

[벤치마크]
Total Return: {bench_total_return:.2f}x
CAGR: {bench_cagr:.2%}
MDD: {bench_mdd:.2%}
Sharpe: {benchmark_sharpe:.2f}'''

ax1.text(0.98, 0.97, metrics_text, transform=ax1.transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
         family='monospace')

# Subplot 2: Drawdown 차트
ax2 = fig.add_subplot(gs[1])
ax2.fill_between(df.index, 0, df['strategy_drawdown'] * 100, color='#E63946', alpha=0.3, label='Strategy DD')
ax2.plot(df.index, df['strategy_drawdown'] * 100, color='#E63946', linewidth=1.5)
ax2.fill_between(df.index, 0, df['benchmark_drawdown'] * 100, color='#F77F00', alpha=0.2, label='Benchmark DD')
ax2.plot(df.index, df['benchmark_drawdown'] * 100, color='#F77F00', linewidth=1.5, linestyle='--')
ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date', fontsize=11)
ax2.legend(loc='lower left', fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.axhline(y=0, color='black', linewidth=0.8)

# Subplot 3: 월별 수익률 히트맵
ax3 = fig.add_subplot(gs[2])

# 월별 수익률을 피벗 테이블로 변환
monthly_rets = monthly_strategy * 100  # 퍼센트로 변환
monthly_rets_df = pd.DataFrame({
    'year': monthly_rets.index.year,
    'month': monthly_rets.index.month,
    'return': monthly_rets.values
})

# 피벗 테이블 생성
pivot_table = monthly_rets_df.pivot_table(values='return', index='year', columns='month', aggfunc='sum')
pivot_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# 히트맵 생성
sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            ax=ax3, cbar_kws={'label': 'Monthly Return (%)'},
            linewidths=0.5, linecolor='gray')
ax3.set_ylabel('Year', fontsize=12, fontweight='bold')
ax3.set_xlabel('Month', fontsize=12, fontweight='bold')
ax3.set_title('Monthly Returns Heatmap (%)', fontsize=13, fontweight='bold')

# 저장
plt.savefig('output/backtest_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n시각화 완료: output/backtest_results.png")

# === 결과 저장 ===
print("\n" + "=" * 80)
print("6. 결과 저장")
print("=" * 80)

# Performance Summary
performance_summary = pd.DataFrame({
    'Strategy': ['Dual MA Strategy', 'Benchmark (SMA30)'],
    'Buy MA': [best_params[0], 'N/A'],
    'Sell MA': [best_params[1], 'N/A'],
    'Total Return (x)': [f"{best_return:.2f}x", f"{bench_total_return:.2f}x"],
    'CAGR (%)': [f"{best_results['cagr']:.2%}", f"{bench_cagr:.2%}"],
    'MDD (%)': [f"{best_results['mdd']:.2%}", f"{bench_mdd:.2%}"],
    'Sharpe Ratio': [f"{strategy_sharpe:.2f}", f"{benchmark_sharpe:.2f}"],
    'Win Rate (%)': [f"{win_rate:.2%}", 'N/A'],
    'Total Trades': [total_trades, (df['benchmark_signal'].diff().abs() > 0).sum()]
})

performance_summary.to_csv('output/performance_summary.csv', index=False, encoding='utf-8-sig')
print("✓ Performance Summary: output/performance_summary.csv")

# Monthly Returns
monthly_returns_df = pd.DataFrame({
    'Date': monthly_strategy.index,
    'Strategy Return (%)': monthly_strategy.values * 100,
    'Benchmark Return (%)': monthly_benchmark.values * 100
})
monthly_returns_df.to_csv('output/monthly_returns.csv', index=False, encoding='utf-8-sig')
print("✓ Monthly Returns: output/monthly_returns.csv")

# All tested strategies
all_strategies_df = pd.DataFrame([
    {
        'Buy MA': r['buy_ma'],
        'Sell MA': r['sell_ma'],
        'Total Return (x)': f"{r['total_return']:.2f}x",
        'CAGR (%)': f"{r['cagr']:.2%}",
        'MDD (%)': f"{r['mdd']:.2%}"
    }
    for r in all_results
]).sort_values('Total Return (x)', ascending=False)

all_strategies_df.to_csv('output/all_strategies_tested.csv', index=False, encoding='utf-8-sig')
print("✓ All Tested Strategies: output/all_strategies_tested.csv")

# === 최종 요약 ===
print("\n" + "=" * 80)
print("백테스트 완료!")
print("=" * 80)
print(f"\n최적 전략: 매수 MA={best_params[0]}, 매도 MA={best_params[1]}")
print(f"  • Total Return: {best_return:.2f}x")
print(f"  • CAGR: {best_results['cagr']:.2%}")
print(f"  • MDD: {best_results['mdd']:.2%}")
print(f"  • Sharpe Ratio: {strategy_sharpe:.2f}")
print(f"  • Win Rate: {win_rate:.2%}")
print(f"\n벤치마크 (SMA30):")
print(f"  • Total Return: {bench_total_return:.2f}x")
print(f"  • CAGR: {bench_cagr:.2%}")
print(f"  • MDD: {bench_mdd:.2%}")
print(f"\n개선율: {improvement:+.2f}%")
print(f"\n결과 저장:")
print(f"  • output/backtest_results.png")
print(f"  • output/performance_summary.csv")
print(f"  • output/monthly_returns.csv")
print(f"  • output/all_strategies_tested.csv")
print("=" * 80)
