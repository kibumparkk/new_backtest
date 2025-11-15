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
print("단순 비대칭 MA 전략 백테스트 (매수 MA ≠ 매도 MA)")
print("=" * 80)
df = pd.read_parquet('chart_day/BTC_KRW.parquet')

# 초기 설정
INITIAL_CAPITAL = 1  # 1원
SLIPPAGE = 0.002     # 0.2%

print(f"\n데이터 기간: {df.index.min()} ~ {df.index.max()}")
print(f"총 {len(df)}일")

# === 벤치마크 전략 구현 (SMA30) ===
print("\n" + "=" * 80)
print("1. 벤치마크 전략 (전일종가 > SMA30)")
print("=" * 80)

df['sma30'] = df['close'].rolling(window=30).mean()
df['benchmark_signal'] = (df['close'].shift(1) > df['sma30'].shift(1)).astype(int)
df['returns'] = df['close'].pct_change()
df['benchmark_returns'] = df['benchmark_signal'] * df['returns']
df['benchmark_position_change'] = df['benchmark_signal'].diff().abs()
df['benchmark_slippage'] = -SLIPPAGE * df['benchmark_position_change']
df['benchmark_returns_with_slippage'] = df['benchmark_returns'] + df['benchmark_slippage']
df['benchmark_equity'] = INITIAL_CAPITAL * (1 + df['benchmark_returns_with_slippage']).cumprod()

# === 단순 비대칭 전략: 매수 MA ≠ 매도 MA ===
print("\n" + "=" * 80)
print("2. 단순 비대칭 전략 최적화")
print("컨셉: 매수 조건(종가 > SMA_A), 매도 조건(종가 < SMA_B)")
print("=" * 80)

best_return = 0
best_params = None
best_results = None
all_results = []

# 매수/매도 MA 범위를 더 넓게 테스트
buy_ma_range = list(range(20, 51, 5))  # 20, 25, 30, 35, 40, 45, 50
sell_ma_range = list(range(20, 51, 5))  # 20, 25, 30, 35, 40, 45, 50

print(f"\n총 {len(buy_ma_range) * len(sell_ma_range)}개 조합 테스트")
print("-" * 80)
print(f"{'매수MA':<8} {'매도MA':<8} {'관계':<12} {'Return':<12} {'CAGR':<10} {'MDD':<10} {'거래수':<8}")
print("-" * 80)

for buy_ma in buy_ma_range:
    for sell_ma in sell_ma_range:
        # 동일한 MA는 벤치마크와 같으므로 스킵
        if buy_ma == sell_ma:
            continue

        # MA 계산
        df[f'sma_{buy_ma}'] = df['close'].rolling(window=buy_ma).mean()
        df[f'sma_{sell_ma}'] = df['close'].rolling(window=sell_ma).mean()

        # 포지션 추적
        position = pd.Series(0, index=df.index)

        for i in range(1, len(df)):
            prev_close = df['close'].iloc[i-1]
            prev_buy_ma = df[f'sma_{buy_ma}'].iloc[i-1]
            prev_sell_ma = df[f'sma_{sell_ma}'].iloc[i-1]
            prev_position = position.iloc[i-1]

            # 매수: 종가 > 매수 MA
            if prev_position == 0:
                if not pd.isna(prev_buy_ma):
                    if prev_close > prev_buy_ma:
                        position.iloc[i] = 1
                    else:
                        position.iloc[i] = 0
                else:
                    position.iloc[i] = 0
            # 매도: 종가 < 매도 MA
            else:
                if not pd.isna(prev_sell_ma):
                    if prev_close < prev_sell_ma:
                        position.iloc[i] = 0
                    else:
                        position.iloc[i] = 1
                else:
                    position.iloc[i] = prev_position

        # 전략 수익률 계산
        strategy_returns = position * df['returns']
        position_change = position.diff().abs()
        slippage_cost = -SLIPPAGE * position_change
        strategy_returns_with_slippage = strategy_returns + slippage_cost

        # 자산 곡선
        equity = INITIAL_CAPITAL * (1 + strategy_returns_with_slippage).cumprod()

        # 성과 지표
        total_return = equity.iloc[-1] / INITIAL_CAPITAL
        total_days = (df.index[-1] - df.index[0]).days
        years = total_days / 365.25
        cagr = (total_return) ** (1 / years) - 1

        cumulative = equity
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        mdd = drawdown.min()

        num_trades = (position_change > 0).sum()

        # 관계 파악
        if buy_ma < sell_ma:
            relation = "빠른매수-느린매도"
        else:
            relation = "느린매수-빠른매도"

        result = {
            'buy_ma': buy_ma,
            'sell_ma': sell_ma,
            'relation': relation,
            'total_return': total_return,
            'cagr': cagr,
            'mdd': mdd,
            'num_trades': num_trades,
            'equity': equity,
            'position': position,
            'returns': strategy_returns_with_slippage
        }
        all_results.append(result)

        print(f"{buy_ma:<8} {sell_ma:<8} {relation:<12} {total_return:<12.2f}x {cagr:<10.2%} {mdd:<10.2%} {num_trades:<8}")

        if total_return > best_return:
            best_return = total_return
            best_params = result
            best_results = result

print("-" * 80)

# 벤치마크 성과
bench_total_return = df['benchmark_equity'].iloc[-1] / INITIAL_CAPITAL
bench_years = (df.index[-1] - df.index[0]).days / 365.25
bench_cagr = (bench_total_return) ** (1 / bench_years) - 1
bench_cumulative = df['benchmark_equity']
bench_running_max = bench_cumulative.cummax()
bench_drawdown = (bench_cumulative - bench_running_max) / bench_running_max
bench_mdd = bench_drawdown.min()
bench_num_trades = (df['benchmark_position_change'] > 0).sum()

print(f"\n최적 전략 발견!")
print(f"  매수 MA: {best_params['buy_ma']}")
print(f"  매도 MA: {best_params['sell_ma']}")
print(f"  전략 유형: {best_params['relation']}")
print(f"  Total Return: {best_return:.2f}x")

print("\n" + "=" * 80)
print("3. 벤치마크 vs 최적 전략 비교")
print("=" * 80)
print(f"\n{'지표':<20} {'벤치마크 (SMA30)':<25} {'최적 전략':<25}")
print("-" * 80)
print(f"{'Total Return':<20} {bench_total_return:<25.2f}x {best_return:<25.2f}x")
print(f"{'CAGR':<20} {bench_cagr:<25.2%} {best_results['cagr']:<25.2%}")
print(f"{'MDD':<20} {bench_mdd:<25.2%} {best_results['mdd']:<25.2%}")
print(f"{'거래 횟수':<20} {bench_num_trades:<25} {best_results['num_trades']:<25}")
print("-" * 80)

improvement = ((best_return - bench_total_return) / bench_total_return) * 100
print(f"\n개선율: {improvement:+.2f}%")

if best_return > bench_total_return:
    print("✅ 벤치마크 능가 성공!")
else:
    print("⚠️  벤치마크 미달")
    # 벤치마크를 능가하는 전략이 있는지 확인
    outperform_count = sum(1 for r in all_results if r['total_return'] > bench_total_return)
    if outperform_count > 0:
        print(f"\n💡 벤치마크를 능가하는 전략 {outperform_count}개 발견!")
        print("\n상위 5개 전략:")
        sorted_results = sorted(all_results, key=lambda x: x['total_return'], reverse=True)
        for i, r in enumerate(sorted_results[:5]):
            status = "✅" if r['total_return'] > bench_total_return else "  "
            print(f"  {status} {i+1}. 매수MA={r['buy_ma']}, 매도MA={r['sell_ma']} ({r['relation']}): {r['total_return']:.2f}x")

# === 성과 지표 계산 ===
print("\n" + "=" * 80)
print("4. 상세 성과 분석")
print("=" * 80)

df['strategy_signal'] = best_results['position']
df['strategy_returns'] = best_results['returns']
df['strategy_equity'] = best_results['equity']
df['strategy_drawdown'] = (df['strategy_equity'] - df['strategy_equity'].cummax()) / df['strategy_equity'].cummax()
df['benchmark_drawdown'] = bench_drawdown

monthly_strategy = df['strategy_returns'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
monthly_benchmark = df['benchmark_returns_with_slippage'].resample('ME').apply(lambda x: (1 + x).prod() - 1)

strategy_trades = df['strategy_signal'].diff().abs()
winning_trades = ((df['strategy_returns'] > 0) & (strategy_trades > 0)).sum()
total_trades = (strategy_trades > 0).sum()
win_rate = winning_trades / total_trades if total_trades > 0 else 0

strategy_sharpe = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(365) if df['strategy_returns'].std() > 0 else 0
benchmark_sharpe = df['benchmark_returns_with_slippage'].mean() / df['benchmark_returns_with_slippage'].std() * np.sqrt(365) if df['benchmark_returns_with_slippage'].std() > 0 else 0

print(f"\n최적 전략:")
print(f"  - 매수 MA: {best_params['buy_ma']} (종가 > SMA{best_params['buy_ma']}일 때 매수)")
print(f"  - 매도 MA: {best_params['sell_ma']} (종가 < SMA{best_params['sell_ma']}일 때 매도)")
print(f"  - 전략 유형: {best_params['relation']}")
print(f"  - 총 거래 횟수: {total_trades}")
print(f"  - 승률: {win_rate:.2%}")
print(f"  - Sharpe Ratio: {strategy_sharpe:.2f}")
print(f"\n벤치마크:")
print(f"  - 총 거래 횟수: {bench_num_trades}")
print(f"  - Sharpe Ratio: {benchmark_sharpe:.2f}")

# === 시각화 ===
print("\n" + "=" * 80)
print("5. 시각화 생성")
print("=" * 80)

fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 2], hspace=0.35)

ax1 = fig.add_subplot(gs[0])
ax1.plot(df.index, df['strategy_equity'], label=f'Strategy (Buy: SMA{best_params["buy_ma"]}, Sell: SMA{best_params["sell_ma"]})', linewidth=2.5, color='#2E86AB')
ax1.plot(df.index, df['benchmark_equity'], label='Benchmark (SMA30)', linewidth=2, alpha=0.7, color='#A23B72', linestyle='--')
ax1.set_yscale('log')
ax1.set_ylabel('Cumulative Returns (KRW, log scale)', fontsize=12, fontweight='bold')
ax1.set_title(f'Simple Asymmetric MA Strategy: {best_params["relation"]}', fontsize=16, fontweight='bold', pad=20)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')

status = "OUTPERFORM ✅" if best_return > bench_total_return else "UNDERPERFORM ⚠️"
metrics_text = f'''[Strategy - {status}]
Buy: Close > SMA{best_params["buy_ma"]}
Sell: Close < SMA{best_params["sell_ma"]}
Type: {best_params["relation"]}
Total Return: {best_return:.2f}x
CAGR: {best_results['cagr']:.2%}
MDD: {best_results['mdd']:.2%}
Sharpe: {strategy_sharpe:.2f}
Trades: {total_trades}

[Benchmark]
Total Return: {bench_total_return:.2f}x
CAGR: {bench_cagr:.2%}
MDD: {bench_mdd:.2%}
Sharpe: {benchmark_sharpe:.2f}
Trades: {bench_num_trades}

Improvement: {improvement:+.2f}%'''

ax1.text(0.98, 0.97, metrics_text, transform=ax1.transAxes,
         fontsize=8.5, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightgreen' if best_return > bench_total_return else 'wheat', alpha=0.9),
         family='monospace')

ax2 = fig.add_subplot(gs[1])
ax2.fill_between(df.index, 0, df['strategy_drawdown'] * 100, color='#E63946', alpha=0.4, label='Strategy DD')
ax2.plot(df.index, df['strategy_drawdown'] * 100, color='#E63946', linewidth=1.5)
ax2.fill_between(df.index, 0, df['benchmark_drawdown'] * 100, color='#F77F00', alpha=0.2, label='Benchmark DD')
ax2.plot(df.index, df['benchmark_drawdown'] * 100, color='#F77F00', linewidth=1.5, linestyle='--')
ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date', fontsize=11)
ax2.legend(loc='lower left', fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.axhline(y=0, color='black', linewidth=0.8)

ax3 = fig.add_subplot(gs[2])
monthly_rets = monthly_strategy * 100
monthly_rets_df = pd.DataFrame({
    'year': monthly_rets.index.year,
    'month': monthly_rets.index.month,
    'return': monthly_rets.values
})

pivot_table = monthly_rets_df.pivot_table(values='return', index='year', columns='month', aggfunc='sum')
pivot_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            ax=ax3, cbar_kws={'label': 'Monthly Return (%)'},
            linewidths=0.5, linecolor='gray')
ax3.set_ylabel('Year', fontsize=12, fontweight='bold')
ax3.set_xlabel('Month', fontsize=12, fontweight='bold')
ax3.set_title('Monthly Returns Heatmap (%)', fontsize=13, fontweight='bold')

plt.savefig('output/backtest_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n시각화 완료: output/backtest_results.png")

# === 결과 저장 ===
print("\n" + "=" * 80)
print("6. 결과 저장")
print("=" * 80)

performance_summary = pd.DataFrame({
    'Strategy': [f'Asymmetric MA ({best_params["relation"]})', 'Benchmark (SMA30)'],
    'Buy MA': [best_params['buy_ma'], 30],
    'Sell MA': [best_params['sell_ma'], 30],
    'Total Return (x)': [f"{best_return:.2f}x", f"{bench_total_return:.2f}x"],
    'CAGR (%)': [f"{best_results['cagr']:.2%}", f"{bench_cagr:.2%}"],
    'MDD (%)': [f"{best_results['mdd']:.2%}", f"{bench_mdd:.2%}"],
    'Sharpe Ratio': [f"{strategy_sharpe:.2f}", f"{benchmark_sharpe:.2f}"],
    'Total Trades': [total_trades, bench_num_trades]
})

performance_summary.to_csv('output/performance_summary.csv', index=False, encoding='utf-8-sig')
print("✓ Performance Summary: output/performance_summary.csv")

monthly_returns_df = pd.DataFrame({
    'Date': monthly_strategy.index,
    'Strategy Return (%)': monthly_strategy.values * 100,
    'Benchmark Return (%)': monthly_benchmark.values * 100
})
monthly_returns_df.to_csv('output/monthly_returns.csv', index=False, encoding='utf-8-sig')
print("✓ Monthly Returns: output/monthly_returns.csv")

all_strategies_df = pd.DataFrame([
    {
        'Buy MA': r['buy_ma'],
        'Sell MA': r['sell_ma'],
        'Relation': r['relation'],
        'Total Return': r['total_return'],
        'Total Return (x)': f"{r['total_return']:.2f}x",
        'CAGR (%)': f"{r['cagr']:.2%}",
        'MDD (%)': f"{r['mdd']:.2%}",
        'Trades': r['num_trades']
    }
    for r in all_results
]).sort_values('Total Return', ascending=False)

all_strategies_df.to_csv('output/all_strategies_tested.csv', index=False, encoding='utf-8-sig')
print("✓ All Tested Strategies: output/all_strategies_tested.csv")

# === 최종 요약 ===
print("\n" + "=" * 80)
print("백테스트 완료!")
print("=" * 80)
print(f"\n최적 전략: {best_params['relation']}")
print(f"  • 매수 조건: 종가 > SMA{best_params['buy_ma']}")
print(f"  • 매도 조건: 종가 < SMA{best_params['sell_ma']}")
print(f"  • Total Return: {best_return:.2f}x")
print(f"  • CAGR: {best_results['cagr']:.2%}")
print(f"  • MDD: {best_results['mdd']:.2%}")
print(f"  • Sharpe Ratio: {strategy_sharpe:.2f}")
print(f"  • 거래 횟수: {total_trades}")
print(f"\n벤치마크 (SMA30):")
print(f"  • Total Return: {bench_total_return:.2f}x")
print(f"  • CAGR: {bench_cagr:.2%}")
print(f"  • MDD: {bench_mdd:.2%}")
print(f"  • 거래 횟수: {bench_num_trades}")
print(f"\n개선율: {improvement:+.2f}%")

if best_return > bench_total_return:
    print("\n✅ 성공: 벤치마크를 능가하는 전략을 찾았습니다!")
    print(f"\n전략 설명:")
    print(f"  매수와 매도의 기준을 분리하여 사용합니다:")
    print(f"  - 매수: 종가가 SMA{best_params['buy_ma']} 위로 올라갈 때 매수")
    print(f"  - 매도: 종가가 SMA{best_params['sell_ma']} 아래로 내려갈 때 매도")
else:
    print("\n⚠️  주의: 현재 전략이 벤치마크에 미달합니다.")

print(f"\n결과 파일:")
print(f"  • output/backtest_results.png")
print(f"  • output/performance_summary.csv")
print(f"  • output/monthly_returns.csv")
print(f"  • output/all_strategies_tested.csv")
print("=" * 80)
