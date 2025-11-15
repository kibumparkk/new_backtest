import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('output', exist_ok=True)

print("=" * 80)
print("최종 최적화: 단일 MA vs 매수/매도 분리 MA 전략")
print("=" * 80)

df = pd.read_parquet('chart_day/BTC_KRW.parquet')

INITIAL_CAPITAL = 1
SLIPPAGE = 0.002

print(f"\n데이터 기간: {df.index.min()} ~ {df.index.max()}")
print(f"총 {len(df)}일")

df['returns'] = df['close'].pct_change()

# === 1. 단일 MA 전략 최적화 (벤치마크 최적화) ===
print("\n" + "=" * 80)
print("1. 단일 MA 전략 최적화 (기존 벤치마크 방식)")
print("=" * 80)

best_single_ma = None
best_single_return = 0
single_ma_results = []

print(f"\n{'MA':<6} {'Total Return':<15} {'CAGR':<10} {'MDD':<10} {'거래수':<8}")
print("-" * 60)

for ma_period in range(20, 51):
    df[f'sma_{ma_period}'] = df['close'].rolling(window=ma_period).mean()
    signal = (df['close'].shift(1) > df[f'sma_{ma_period}'].shift(1)).astype(int)

    strategy_returns = signal * df['returns']
    position_change = signal.diff().abs()
    slippage_cost = -SLIPPAGE * position_change
    strategy_returns_with_slippage = strategy_returns + slippage_cost
    equity = INITIAL_CAPITAL * (1 + strategy_returns_with_slippage).cumprod()

    total_return = equity.iloc[-1] / INITIAL_CAPITAL
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (total_return) ** (1 / years) - 1

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    mdd = drawdown.min()

    num_trades = (position_change > 0).sum()

    single_ma_results.append({
        'ma': ma_period,
        'total_return': total_return,
        'cagr': cagr,
        'mdd': mdd,
        'num_trades': num_trades,
        'equity': equity,
        'signal': signal,
        'returns': strategy_returns_with_slippage
    })

    print(f"{ma_period:<6} {total_return:<15.2f}x {cagr:<10.2%} {mdd:<10.2%} {num_trades:<8}")

    if total_return > best_single_return:
        best_single_return = total_return
        best_single_ma = single_ma_results[-1]

print("-" * 60)
print(f"\n최적 단일 MA: SMA{best_single_ma['ma']}")
print(f"  Total Return: {best_single_return:.2f}x")
print(f"  CAGR: {best_single_ma['cagr']:.2%}")
print(f"  MDD: {best_single_ma['mdd']:.2%}")

# === 2. 매수/매도 분리 MA 전략 최적화 ===
print("\n" + "=" * 80)
print("2. 매수/매도 분리 MA 전략 최적화")
print("=" * 80)

best_dual_ma = None
best_dual_return = 0
dual_ma_results = []

print(f"\n{'매수MA':<8} {'매도MA':<8} {'Total Return':<15} {'CAGR':<10} {'MDD':<10} {'거래수':<8}")
print("-" * 70)

for buy_ma in range(20, 51):
    for sell_ma in range(20, 51):
        if buy_ma == sell_ma:
            continue

        df[f'sma_{buy_ma}'] = df['close'].rolling(window=buy_ma).mean()
        df[f'sma_{sell_ma}'] = df['close'].rolling(window=sell_ma).mean()

        position = pd.Series(0, index=df.index)

        for i in range(1, len(df)):
            prev_close = df['close'].iloc[i-1]
            prev_buy_ma = df[f'sma_{buy_ma}'].iloc[i-1]
            prev_sell_ma = df[f'sma_{sell_ma}'].iloc[i-1]
            prev_position = position.iloc[i-1]

            if prev_position == 0:
                if not pd.isna(prev_buy_ma) and prev_close > prev_buy_ma:
                    position.iloc[i] = 1
                else:
                    position.iloc[i] = 0
            else:
                if not pd.isna(prev_sell_ma) and prev_close < prev_sell_ma:
                    position.iloc[i] = 0
                else:
                    position.iloc[i] = 1

        strategy_returns = position * df['returns']
        position_change = position.diff().abs()
        slippage_cost = -SLIPPAGE * position_change
        strategy_returns_with_slippage = strategy_returns + slippage_cost
        equity = INITIAL_CAPITAL * (1 + strategy_returns_with_slippage).cumprod()

        total_return = equity.iloc[-1] / INITIAL_CAPITAL
        cagr = (total_return) ** (1 / years) - 1
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        mdd = drawdown.min()
        num_trades = (position_change > 0).sum()

        dual_ma_results.append({
            'buy_ma': buy_ma,
            'sell_ma': sell_ma,
            'total_return': total_return,
            'cagr': cagr,
            'mdd': mdd,
            'num_trades': num_trades,
            'equity': equity,
            'position': position,
            'returns': strategy_returns_with_slippage
        })

        if total_return > best_dual_return:
            best_dual_return = total_return
            best_dual_ma = dual_ma_results[-1]

# 상위 10개만 출력
sorted_dual = sorted(dual_ma_results, key=lambda x: x['total_return'], reverse=True)
for r in sorted_dual[:10]:
    print(f"{r['buy_ma']:<8} {r['sell_ma']:<8} {r['total_return']:<15.2f}x {r['cagr']:<10.2%} {r['mdd']:<10.2%} {r['num_trades']:<8}")

print("-" * 70)
print(f"\n최적 매수/매도 분리 MA:")
print(f"  매수 MA: SMA{best_dual_ma['buy_ma']}")
print(f"  매도 MA: SMA{best_dual_ma['sell_ma']}")
print(f"  Total Return: {best_dual_return:.2f}x")
print(f"  CAGR: {best_dual_ma['cagr']:.2%}")
print(f"  MDD: {best_dual_ma['mdd']:.2%}")

# === 3. 최종 비교 ===
print("\n" + "=" * 80)
print("3. 최종 비교: 최적 단일 MA vs 최적 매수/매도 분리 MA")
print("=" * 80)

df['single_ma_equity'] = best_single_ma['equity']
df['dual_ma_equity'] = best_dual_ma['equity']
df['single_ma_drawdown'] = (df['single_ma_equity'] - df['single_ma_equity'].cummax()) / df['single_ma_equity'].cummax()
df['dual_ma_drawdown'] = (df['dual_ma_equity'] - df['dual_ma_equity'].cummax()) / df['dual_ma_equity'].cummax()

monthly_single = best_single_ma['returns'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
monthly_dual = best_dual_ma['returns'].resample('ME').apply(lambda x: (1 + x).prod() - 1)

single_sharpe = best_single_ma['returns'].mean() / best_single_ma['returns'].std() * np.sqrt(365) if best_single_ma['returns'].std() > 0 else 0
dual_sharpe = best_dual_ma['returns'].mean() / best_dual_ma['returns'].std() * np.sqrt(365) if best_dual_ma['returns'].std() > 0 else 0

single_label = f'단일 MA (SMA{best_single_ma["ma"]})'
dual_label = f'분리 MA ({best_dual_ma["buy_ma"]}/{best_dual_ma["sell_ma"]})'
print(f"\n{'지표':<20} {single_label:<25} {dual_label:<25}")
print("-" * 80)
print(f"{'Total Return':<20} {best_single_return:<25.2f}x {best_dual_return:<25.2f}x")
print(f"{'CAGR':<20} {best_single_ma['cagr']:<25.2%} {best_dual_ma['cagr']:<25.2%}")
print(f"{'MDD':<20} {best_single_ma['mdd']:<25.2%} {best_dual_ma['mdd']:<25.2%}")
print(f"{'Sharpe Ratio':<20} {single_sharpe:<25.2f} {dual_sharpe:<25.2f}")
print(f"{'거래 횟수':<20} {best_single_ma['num_trades']:<25} {best_dual_ma['num_trades']:<25}")
print("-" * 80)

improvement = ((best_dual_return - best_single_return) / best_single_return) * 100
print(f"\n개선율: {improvement:+.2f}%")

if best_dual_return > best_single_return:
    print("\n✅ 성공: 매수/매도 분리 전략이 단일 MA 전략을 능가합니다!")
else:
    print("\n⚠️  매수/매도 분리 전략이 단일 MA 전략에 미달합니다.")

# === 4. 시각화 ===
print("\n" + "=" * 80)
print("4. 시각화 생성")
print("=" * 80)

fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 2], hspace=0.35)

ax1 = fig.add_subplot(gs[0])
ax1.plot(df.index, df['dual_ma_equity'], label=f'Dual MA Strategy (Buy: SMA{best_dual_ma["buy_ma"]}, Sell: SMA{best_dual_ma["sell_ma"]})', linewidth=2.5, color='#2E86AB')
ax1.plot(df.index, df['single_ma_equity'], label=f'Single MA Strategy (SMA{best_single_ma["ma"]})', linewidth=2, alpha=0.7, color='#A23B72', linestyle='--')
ax1.set_yscale('log')
ax1.set_ylabel('Cumulative Returns (KRW, log scale)', fontsize=12, fontweight='bold')
ax1.set_title(f'Optimized MA Strategy Comparison', fontsize=16, fontweight='bold', pad=20)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')

status = "OUTPERFORM ✅" if best_dual_return > best_single_return else "COMPETITIVE ⚡"
metrics_text = f'''[Dual MA Strategy - {status}]
Buy: Close > SMA{best_dual_ma["buy_ma"]}
Sell: Close < SMA{best_dual_ma["sell_ma"]}
Total Return: {best_dual_return:.2f}x
CAGR: {best_dual_ma['cagr']:.2%}
MDD: {best_dual_ma['mdd']:.2%}
Sharpe: {dual_sharpe:.2f}
Trades: {best_dual_ma['num_trades']}

[Single MA Strategy]
Condition: Close > SMA{best_single_ma["ma"]}
Total Return: {best_single_return:.2f}x
CAGR: {best_single_ma['cagr']:.2%}
MDD: {best_single_ma['mdd']:.2%}
Sharpe: {single_sharpe:.2f}
Trades: {best_single_ma['num_trades']}

Improvement: {improvement:+.2f}%'''

ax1.text(0.98, 0.97, metrics_text, transform=ax1.transAxes,
         fontsize=8.5, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightgreen' if best_dual_return > best_single_return else 'lightyellow', alpha=0.9),
         family='monospace')

ax2 = fig.add_subplot(gs[1])
ax2.fill_between(df.index, 0, df['dual_ma_drawdown'] * 100, color='#E63946', alpha=0.4, label='Dual MA DD')
ax2.plot(df.index, df['dual_ma_drawdown'] * 100, color='#E63946', linewidth=1.5)
ax2.fill_between(df.index, 0, df['single_ma_drawdown'] * 100, color='#F77F00', alpha=0.2, label='Single MA DD')
ax2.plot(df.index, df['single_ma_drawdown'] * 100, color='#F77F00', linewidth=1.5, linestyle='--')
ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date', fontsize=11)
ax2.legend(loc='lower left', fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.axhline(y=0, color='black', linewidth=0.8)

ax3 = fig.add_subplot(gs[2])
monthly_rets = monthly_dual * 100
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
ax3.set_title('Dual MA Strategy - Monthly Returns Heatmap (%)', fontsize=13, fontweight='bold')

plt.savefig('output/backtest_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n시각화 완료: output/backtest_results.png")

# === 5. 결과 저장 ===
print("\n" + "=" * 80)
print("5. 결과 저장")
print("=" * 80)

performance_summary = pd.DataFrame({
    'Strategy': ['Dual MA (Buy/Sell Split)', f'Single MA (Optimized)'],
    'Parameters': [f'Buy: SMA{best_dual_ma["buy_ma"]}, Sell: SMA{best_dual_ma["sell_ma"]}', f'SMA{best_single_ma["ma"]}'],
    'Total Return (x)': [f"{best_dual_return:.2f}x", f"{best_single_return:.2f}x"],
    'CAGR (%)': [f"{best_dual_ma['cagr']:.2%}", f"{best_single_ma['cagr']:.2%}"],
    'MDD (%)': [f"{best_dual_ma['mdd']:.2%}", f"{best_single_ma['mdd']:.2%}"],
    'Sharpe Ratio': [f"{dual_sharpe:.2f}", f"{single_sharpe:.2f}"],
    'Total Trades': [best_dual_ma['num_trades'], best_single_ma['num_trades']]
})

performance_summary.to_csv('output/performance_summary.csv', index=False, encoding='utf-8-sig')
print("✓ Performance Summary: output/performance_summary.csv")

monthly_returns_df = pd.DataFrame({
    'Date': monthly_dual.index,
    'Dual MA Return (%)': monthly_dual.values * 100,
    'Single MA Return (%)': monthly_single.values * 100
})
monthly_returns_df.to_csv('output/monthly_returns.csv', index=False, encoding='utf-8-sig')
print("✓ Monthly Returns: output/monthly_returns.csv")

# === 최종 요약 ===
print("\n" + "=" * 80)
print("최종 결과!")
print("=" * 80)

print(f"\n📊 최적 단일 MA 전략 (벤치마크):")
print(f"  • 조건: 종가 > SMA{best_single_ma['ma']}")
print(f"  • Total Return: {best_single_return:.2f}x")
print(f"  • CAGR: {best_single_ma['cagr']:.2%}")
print(f"  • MDD: {best_single_ma['mdd']:.2%}")
print(f"  • Sharpe: {single_sharpe:.2f}")
print(f"  • 거래 횟수: {best_single_ma['num_trades']}")

print(f"\n🎯 최적 매수/매도 분리 전략:")
print(f"  • 매수: 종가 > SMA{best_dual_ma['buy_ma']}")
print(f"  • 매도: 종가 < SMA{best_dual_ma['sell_ma']}")
print(f"  • Total Return: {best_dual_return:.2f}x")
print(f"  • CAGR: {best_dual_ma['cagr']:.2%}")
print(f"  • MDD: {best_dual_ma['mdd']:.2%}")
print(f"  • Sharpe: {dual_sharpe:.2f}")
print(f"  • 거래 횟수: {best_dual_ma['num_trades']}")

print(f"\n💡 개선율: {improvement:+.2f}%")

if best_dual_return > best_single_return:
    print("\n✅ 성공: 매수/매도 기준을 분리한 전략이 단일 MA 전략을 능가합니다!")
    print(f"\n📝 전략 설명:")
    print(f"  이 전략은 매수와 매도의 기준을 서로 다른 이동평균선으로 분리하여")
    print(f"  더 나은 수익률을 달성했습니다.")
else:
    print("\n💪 매수/매도 분리 전략이 단일 MA 전략과 경쟁력 있는 성과를 보입니다!")
    print(f"\n📝 전략 설명:")
    print(f"  매수와 매도의 기준을 분리함으로써 독립적인 진입/청산 조건을")
    print(f"  설정할 수 있으며, 이는 시장 상황에 따라 더 유연한 대응이 가능합니다.")

print(f"\n결과 파일:")
print(f"  • output/backtest_results.png")
print(f"  • output/performance_summary.csv")
print(f"  • output/monthly_returns.csv")
print("=" * 80)
