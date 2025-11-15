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
df = pd.read_parquet('chart_day/BTC_KRW.parquet')
print(f"데이터 기간: {df.index.min()} ~ {df.index.max()}")
print(f"총 {len(df)}일")

# 초기 설정
INITIAL_CAPITAL = 1
SLIPPAGE = 0.002

# 기술적 지표 계산
df['sma5'] = df['close'].rolling(window=5).mean()
df['sma10'] = df['close'].rolling(window=10).mean()
df['sma20'] = df['close'].rolling(window=20).mean()
df['sma30'] = df['close'].rolling(window=30).mean()
df['sma50'] = df['close'].rolling(window=50).mean()
df['sma100'] = df['close'].rolling(window=100).mean()
df['sma200'] = df['close'].rolling(window=200).mean()

df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
df['ema30'] = df['close'].ewm(span=30, adjust=False).mean()
df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

# Bollinger Bands
df['bb_middle'] = df['close'].rolling(window=20).mean()
df['bb_std'] = df['close'].rolling(window=20).std()
df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

# Donchian Channel
df['dc_high_20'] = df['high'].rolling(window=20).max()
df['dc_low_20'] = df['low'].rolling(window=20).min()
df['dc_high_55'] = df['high'].rolling(window=55).max()
df['dc_low_55'] = df['low'].rolling(window=55).min()

# ATR
df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    )
)
df['atr14'] = df['tr'].rolling(window=14).mean()
df['atr20'] = df['tr'].rolling(window=20).mean()

# MACD
df['macd_line'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
df['macd_hist'] = df['macd_line'] - df['macd_signal']

# ADX
def calculate_adx(df, period=14):
    df['high_diff'] = df['high'].diff()
    df['low_diff'] = -df['low'].diff()
    df['plus_dm'] = np.where((df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0), df['high_diff'], 0)
    df['minus_dm'] = np.where((df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0), df['low_diff'], 0)

    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['atr14'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['atr14'])
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(window=period).mean()

    return df

df = calculate_adx(df.copy())

# RSI
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['rsi14'] = calculate_rsi(df['close'], 14)

# 벤치마크 전략: 종가 > SMA30 (shift(1)은 백테스트 함수에서 적용)
df['benchmark_signal'] = (df['close'] > df['sma30']).astype(int)

def backtest_strategy(signals, name):
    """백테스트 실행 함수"""
    df_bt = df.copy()
    df_bt['signal'] = signals

    # 일일 수익률 계산 (슬리피지 적용)
    df_bt['position_change'] = df_bt['signal'].diff()
    df_bt['daily_return'] = df_bt['close'].pct_change()

    # 슬리피지 적용
    df_bt['slippage_cost'] = 0.0
    df_bt.loc[df_bt['position_change'] != 0, 'slippage_cost'] = SLIPPAGE

    # 전략 수익률
    df_bt['strategy_return'] = df_bt['signal'].shift(1) * df_bt['daily_return'] - abs(df_bt['position_change']) * SLIPPAGE
    df_bt['strategy_equity'] = INITIAL_CAPITAL * (1 + df_bt['strategy_return']).cumprod()

    # 벤치마크 수익률
    df_bt['benchmark_position_change'] = df_bt['benchmark_signal'].diff()
    df_bt['benchmark_return'] = df_bt['benchmark_signal'].shift(1) * df_bt['daily_return'] - abs(df_bt['benchmark_position_change']) * SLIPPAGE
    df_bt['benchmark_equity'] = INITIAL_CAPITAL * (1 + df_bt['benchmark_return']).cumprod()

    # Buy & Hold 수익률
    df_bt['bh_return'] = df_bt['daily_return']
    df_bt['bh_equity'] = INITIAL_CAPITAL * (1 + df_bt['bh_return']).cumprod()

    # 유효한 데이터만 사용
    df_bt = df_bt.dropna()

    if len(df_bt) == 0 or df_bt['strategy_equity'].iloc[-1] <= 0:
        return None

    # 성과 지표 계산
    total_return = df_bt['strategy_equity'].iloc[-1] / INITIAL_CAPITAL
    benchmark_return = df_bt['benchmark_equity'].iloc[-1] / INITIAL_CAPITAL

    total_days = (df_bt.index[-1] - df_bt.index[0]).days
    years = total_days / 365.25
    cagr = (df_bt['strategy_equity'].iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1

    # MDD 계산
    running_max = df_bt['strategy_equity'].cummax()
    drawdown = (df_bt['strategy_equity'] - running_max) / running_max * 100
    mdd = drawdown.min()

    # Sharpe Ratio 계산
    if df_bt['strategy_return'].std() != 0:
        sharpe_ratio = (df_bt['strategy_return'].mean() / df_bt['strategy_return'].std()) * np.sqrt(365)
    else:
        sharpe_ratio = 0

    return {
        'name': name,
        'total_return': total_return,
        'benchmark_return': benchmark_return,
        'cagr': cagr,
        'mdd': mdd,
        'sharpe_ratio': sharpe_ratio,
        'df': df_bt
    }

# 전략 정의
strategies = []

# 1. Dual Moving Average Crossover 전략들
for fast, slow in [(5, 20), (10, 30), (10, 50), (20, 50), (20, 100), (50, 200)]:
    signal = (df[f'sma{fast}'] > df[f'sma{slow}']).astype(int)
    strategies.append((signal, f'SMA_{fast}_{slow}_Cross'))

# 2. EMA Crossover 전략들
for fast, slow in [(10, 30), (10, 50), (20, 50)]:
    signal = (df[f'ema{fast}'] > df[f'ema{slow}']).astype(int)
    strategies.append((signal, f'EMA_{fast}_{slow}_Cross'))

# 3. Triple Moving Average 전략
signal = ((df['sma10'] > df['sma30']) & (df['sma30'] > df['sma100'])).astype(int)
strategies.append((signal, 'Triple_SMA_10_30_100'))

signal = ((df['sma20'] > df['sma50']) & (df['sma50'] > df['sma200'])).astype(int)
strategies.append((signal, 'Triple_SMA_20_50_200'))

# 4. Donchian Channel Breakout
signal = (df['close'] > df['dc_high_20'].shift(1)).astype(int)
strategies.append((signal, 'Donchian_20_Breakout'))

signal = (df['close'] > df['dc_high_55'].shift(1)).astype(int)
strategies.append((signal, 'Donchian_55_Breakout'))

# 5. Bollinger Band 전략
signal = (df['close'] > df['bb_upper']).astype(int)
strategies.append((signal, 'BB_Upper_Breakout'))

# 6. MACD 전략
signal = (df['macd_line'] > df['macd_signal']).astype(int)
strategies.append((signal, 'MACD_Cross'))

signal = ((df['macd_line'] > df['macd_signal']) & (df['macd_hist'] > 0)).astype(int)
strategies.append((signal, 'MACD_Bullish'))

# 7. ADX + Moving Average 조합
signal = ((df['sma20'] > df['sma50']) & (df['adx'] > 25)).astype(int)
strategies.append((signal, 'SMA_20_50_ADX25'))

signal = ((df['sma10'] > df['sma30']) & (df['adx'] > 30)).astype(int)
strategies.append((signal, 'SMA_10_30_ADX30'))

# 8. RSI + Trend 조합
signal = ((df['sma20'] > df['sma50']) & (df['rsi14'] > 50)).astype(int)
strategies.append((signal, 'SMA_20_50_RSI50'))

# 9. Multiple Timeframe Trend
signal = ((df['sma10'] > df['sma20']) & (df['sma20'] > df['sma50']) & (df['sma50'] > df['sma100'])).astype(int)
strategies.append((signal, 'Cascade_Trend_10_20_50_100'))

# 10. Price > Multiple MAs
signal = ((df['close'] > df['sma10']) & (df['close'] > df['sma30']) & (df['close'] > df['sma50'])).astype(int)
strategies.append((signal, 'Price_Above_All_MAs'))

# 11. Momentum Breakout
df['momentum'] = df['close'] - df['close'].shift(20)
signal = (df['momentum'] > 0).astype(int)
strategies.append((signal, 'Momentum_20'))

# 12. Rate of Change
df['roc'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
signal = (df['roc'] > 0).astype(int)
strategies.append((signal, 'ROC_10'))

# 모든 전략 테스트
results = []
print("\n백테스트 진행 중...\n")

for signal, name in strategies:
    result = backtest_strategy(signal, name)
    if result is not None:
        results.append(result)
        print(f"{name:30s} | Return: {result['total_return']:8.2f}x | Benchmark: {result['benchmark_return']:8.2f}x | CAGR: {result['cagr']:7.2%} | MDD: {result['mdd']:7.2%} | Sharpe: {result['sharpe_ratio']:6.2f}")

# 결과를 Total Return 기준으로 정렬
results.sort(key=lambda x: x['total_return'], reverse=True)

# 최고 성과 전략 찾기
print("\n" + "="*100)
print("상위 10개 전략:")
print("="*100)

for i, result in enumerate(results[:10], 1):
    print(f"{i:2d}. {result['name']:30s} | Return: {result['total_return']:8.2f}x | Benchmark: {result['benchmark_return']:8.2f}x | CAGR: {result['cagr']:7.2%} | MDD: {result['mdd']:7.2%} | Sharpe: {result['sharpe_ratio']:6.2f}")

# 벤치마크를 이기는 전략 찾기
winning_strategies = [r for r in results if r['total_return'] > r['benchmark_return']]

print("\n" + "="*100)
print(f"벤치마크를 이기는 전략: {len(winning_strategies)}개")
print("="*100)

if winning_strategies:
    best_strategy = winning_strategies[0]
    print(f"\n최고 성과 전략: {best_strategy['name']}")
    print(f"Total Return: {best_strategy['total_return']:.2f}x")
    print(f"Benchmark Return: {best_strategy['benchmark_return']:.2f}x")
    print(f"Outperformance: {(best_strategy['total_return'] / best_strategy['benchmark_return'] - 1) * 100:.2f}%")
    print(f"CAGR: {best_strategy['cagr']:.2%}")
    print(f"MDD: {best_strategy['mdd']:.2%}")
    print(f"Sharpe Ratio: {best_strategy['sharpe_ratio']:.2f}")

    # 최고 전략 시각화
    df_best = best_strategy['df']

    # 월별 수익률 계산
    df_best['month'] = df_best.index.to_period('M')
    monthly_returns = df_best.groupby('month')['strategy_return'].apply(lambda x: (1 + x).prod() - 1) * 100
    monthly_returns.index = monthly_returns.index.to_timestamp()

    # 피벗 테이블 생성
    pivot_data = []
    for date, ret in monthly_returns.items():
        pivot_data.append({'year': date.year, 'month': date.month, 'return': ret})

    pivot_df = pd.DataFrame(pivot_data)
    pivot_table = pivot_df.pivot(index='year', columns='month', values='return')
    pivot_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Drawdown 계산
    df_best['drawdown_pct'] = (df_best['strategy_equity'] - df_best['strategy_equity'].cummax()) / df_best['strategy_equity'].cummax() * 100

    # 시각화
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 2], hspace=0.3)

    # Subplot 1: 누적 자산 곡선
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df_best.index, df_best['strategy_equity'], label=f'Strategy: {best_strategy["name"]}', linewidth=2)
    ax1.plot(df_best.index, df_best['benchmark_equity'], label='Benchmark (SMA30)', linewidth=2, alpha=0.7)
    ax1.plot(df_best.index, df_best['bh_equity'], label='Buy & Hold', linewidth=1.5, alpha=0.5, linestyle='--')
    ax1.set_yscale('log')
    ax1.set_ylabel('Cumulative Returns (KRW, log scale)', fontsize=11)
    ax1.set_title('Backtest Performance Analysis - Best Trend Following Strategy', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 성과지표 텍스트 박스
    metrics_text = f'''Total Return: {best_strategy["total_return"]:.2f}x
CAGR: {best_strategy["cagr"]:.2%}
MDD: {best_strategy["mdd"]:.2%}
Sharpe: {best_strategy["sharpe_ratio"]:.2f}

vs Benchmark: {best_strategy["benchmark_return"]:.2f}x
Outperformance: {(best_strategy["total_return"] / best_strategy["benchmark_return"] - 1) * 100:.1f}%'''

    ax1.text(0.98, 0.97, metrics_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Subplot 2: Drawdown 차트
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(df_best.index, 0, df_best['drawdown_pct'], color='red', alpha=0.3)
    ax2.plot(df_best.index, df_best['drawdown_pct'], color='red', linewidth=1)
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linewidth=0.5)

    # Subplot 3: 월별 수익률 히트맵
    ax3 = fig.add_subplot(gs[2])
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                ax=ax3, cbar_kws={'label': 'Monthly Return (%)'})
    ax3.set_ylabel('Year', fontsize=11)
    ax3.set_xlabel('Month', fontsize=11)
    ax3.set_title('Monthly Returns (%)', fontsize=12)

    # 저장
    plt.savefig('output/backtest_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n시각화 완료: output/backtest_results.png")

    # 성과 요약 CSV 저장
    summary_df = pd.DataFrame([{
        'Strategy': best_strategy['name'],
        'Total_Return_x': best_strategy['total_return'],
        'Benchmark_Return_x': best_strategy['benchmark_return'],
        'CAGR_%': best_strategy['cagr'] * 100,
        'MDD_%': best_strategy['mdd'],
        'Sharpe_Ratio': best_strategy['sharpe_ratio'],
        'Outperformance_%': (best_strategy['total_return'] / best_strategy['benchmark_return'] - 1) * 100
    }])
    summary_df.to_csv('output/performance_summary.csv', index=False)
    print("성과 요약 저장: output/performance_summary.csv")

    # 월별 수익률 CSV 저장
    pivot_table.to_csv('output/monthly_returns.csv')
    print("월별 수익률 저장: output/monthly_returns.csv")

    # 전체 전략 비교표 저장
    all_strategies_df = pd.DataFrame([{
        'Rank': i + 1,
        'Strategy': r['name'],
        'Total_Return_x': r['total_return'],
        'Benchmark_Return_x': r['benchmark_return'],
        'CAGR_%': r['cagr'] * 100,
        'MDD_%': r['mdd'],
        'Sharpe_Ratio': r['sharpe_ratio'],
        'Beats_Benchmark': 'Yes' if r['total_return'] > r['benchmark_return'] else 'No'
    } for i, r in enumerate(results)])
    all_strategies_df.to_csv('output/all_strategies_comparison.csv', index=False)
    print("전체 전략 비교표 저장: output/all_strategies_comparison.csv")

else:
    print("\n경고: 벤치마크를 이기는 전략을 찾지 못했습니다.")
    print("다른 전략이나 파라미터를 시도해야 합니다.")

print("\n백테스트 완료!")
