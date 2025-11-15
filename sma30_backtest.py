"""
SMA30 전략 백테스트
===================
전일 종가가 SMA30(30일 이동평균) 위에 있으면 매수, 아니면 매도/현금 보유

벤치마크 전략: 전일종가 > SMA30
슬리피지: 0.2% (매수/매도 시 각각 적용)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# 설정
INITIAL_CAPITAL = 1.0  # 1원
SLIPPAGE = 0.002       # 0.2%
SMA_WINDOW = 30        # 30일 이동평균

# 출력 폴더 생성
os.makedirs('output', exist_ok=True)

# 데이터 로드
print("=" * 60)
print("SMA30 Strategy Backtest")
print("=" * 60)
print("\n[1] 데이터 로딩...")

df = pd.read_parquet('chart_day/BTC_KRW.parquet')
print(f"  데이터 기간: {df.index.min()} ~ {df.index.max()}")
print(f"  총 {len(df)}일")
print(f"  컬럼: {list(df.columns)}")

# SMA 계산
print("\n[2] SMA30 계산...")
df['sma30'] = df['close'].rolling(window=SMA_WINDOW).mean()

# 결측치 제거 (SMA 계산을 위해 초기 30일 제외)
df = df.dropna()
print(f"  유효 데이터: {len(df)}일 (SMA 계산 후)")

# 신호: 전일종가 > 전일SMA30
# 전일(t-1)의 종가가 전일(t-1)의 SMA30보다 높으면 오늘(t) 매수
# shift(1)을 여기서만 사용 (position에서는 shift 불필요)
df['signal'] = (df['close'].shift(1) > df['sma30'].shift(1)).astype(int)

print("\n[3] 백테스트 실행...")
print("  방법 1: 벡터화 연산 (Pandas/NumPy)")

# =============================================================================
# 방법 1: 벡터화 구현
# =============================================================================

# 일간 수익률 계산
df['daily_return'] = df['close'].pct_change()

# 포지션 = 신호 (shift는 이미 signal 계산 시 적용됨)
# signal[t]는 이미 t-1일 데이터로 계산되었으므로, t일에 바로 사용 가능
df['position'] = df['signal'].fillna(0)

# 포지션 변화 계산 (매수/매도 시점 파악)
df['position_change'] = df['position'].diff()

# 전략 수익률 계산
df['strategy_return'] = 0.0

# 포지션 유지 중일 때 수익률 적용
df.loc[df['position'] == 1, 'strategy_return'] = df['daily_return']

# 슬리피지 적용: 포지션이 바뀔 때만
# 매수 시 (0 -> 1): 슬리피지만 차감 (아직 수익/손실 없음)
df.loc[df['position_change'] == 1, 'strategy_return'] = -SLIPPAGE

# 매도 시 (1 -> 0): 당일 수익 + 슬리피지 차감
# 매도는 다음날 장 시작 시 하므로, 당일 수익은 없고 슬리피지만 발생
df.loc[df['position_change'] == -1, 'strategy_return'] = -SLIPPAGE

# 누적 자산 계산 (복리)
df['strategy_equity'] = INITIAL_CAPITAL * (1 + df['strategy_return']).cumprod()

# Buy & Hold 수익률 (참고용)
df['buyhold_equity'] = INITIAL_CAPITAL * (1 + df['daily_return']).cumprod()

# Drawdown 계산
running_max = df['strategy_equity'].cummax()
df['drawdown'] = (df['strategy_equity'] - running_max) / running_max
df['drawdown_pct'] = df['drawdown'] * 100

# 최종 결과
final_equity_vectorized = df['strategy_equity'].iloc[-1]
total_return_vectorized = final_equity_vectorized / INITIAL_CAPITAL

print(f"  벡터화 방법 최종 자산: {final_equity_vectorized:.6f}원")
print(f"  벡터화 방법 Total Return: {total_return_vectorized:.6f}x")

# =============================================================================
# 방법 2: 반복문 구현 (검증용)
# =============================================================================

print("\n  방법 2: 반복문 구현 (검증용)")

equity_loop = INITIAL_CAPITAL
position = 0  # 0: 현금, 1: 코인 보유
equity_history = []
position_list = []

for i in range(len(df)):
    # 첫날은 포지션만 설정
    if i == 0:
        equity_history.append(equity_loop)
        position_list.append(0)
        continue

    # 오늘의 신호 (이미 전일 데이터로 계산됨)
    target_position = df['signal'].iloc[i]

    # 오늘 가격 변화
    price_return = df['daily_return'].iloc[i]

    # 포지션 변경 여부 확인
    if target_position != position:
        # 포지션 변경 -> 슬리피지 발생
        equity_loop *= (1 - SLIPPAGE)
        position = target_position
    else:
        # 포지션 유지
        if position == 1:
            # 코인 보유 중이면 가격 변동 반영
            equity_loop *= (1 + price_return)

    equity_history.append(equity_loop)
    position_list.append(position)

df['strategy_equity_loop'] = equity_history
final_equity_loop = df['strategy_equity_loop'].iloc[-1]
total_return_loop = final_equity_loop / INITIAL_CAPITAL

print(f"  반복문 방법 최종 자산: {final_equity_loop:.6f}원")
print(f"  반복문 방법 Total Return: {total_return_loop:.6f}x")

# =============================================================================
# 방법 간 검증
# =============================================================================

print("\n[4] 검증: 두 방법 간 결과 비교")
difference = abs(total_return_vectorized - total_return_loop)
difference_pct = difference / total_return_vectorized * 100

print(f"  차이: {difference:.6f} ({difference_pct:.4f}%)")

if difference_pct < 0.01:
    print("  ✓ 검증 성공: 차이 < 0.01%")
else:
    print("  ✗ 검증 실패: 차이 >= 0.01%")
    print("  로직을 다시 확인해주세요.")

# 이후 분석은 벡터화 결과 사용
print("\n[5] 성과 지표 계산...")

# Total Return (배수)
total_return = final_equity_vectorized / INITIAL_CAPITAL

# CAGR (연평균 복리수익률)
total_days = (df.index[-1] - df.index[0]).days
years = total_days / 365.25
cagr = (final_equity_vectorized / INITIAL_CAPITAL) ** (1 / years) - 1

# MDD (Maximum Drawdown)
mdd = df['drawdown_pct'].min()

# 벤치마크 성과 (참고용)
benchmark_return = df['buyhold_equity'].iloc[-1] / INITIAL_CAPITAL
benchmark_cagr = (df['buyhold_equity'].iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1

# 월별 수익률 계산
monthly_returns = df['strategy_return'].resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100
monthly_returns_df = pd.DataFrame({
    'Year': monthly_returns.index.year,
    'Month': monthly_returns.index.month,
    'Return_pct': monthly_returns.values
})

print(f"  Total Return: {total_return:.2f}x")
print(f"  CAGR: {cagr:.2%}")
print(f"  MDD: {mdd:.2f}%")
print(f"  최종 자산: {final_equity_vectorized:,.6f}원")
print(f"  Buy & Hold Total Return: {benchmark_return:.2f}x")
print(f"  Buy & Hold CAGR: {benchmark_cagr:.2%}")

# =============================================================================
# 성과 요약 저장
# =============================================================================

print("\n[6] 결과 저장...")

performance_summary = pd.DataFrame({
    'Metric': ['Total Return', 'CAGR', 'MDD', 'Final Equity',
               'Buy&Hold Total Return', 'Buy&Hold CAGR',
               'Start Date', 'End Date', 'Trading Days'],
    'Value': [
        f"{total_return:.4f}x",
        f"{cagr:.4%}",
        f"{mdd:.2f}%",
        f"{final_equity_vectorized:.4f}",
        f"{benchmark_return:.4f}x",
        f"{benchmark_cagr:.4%}",
        str(df.index[0].date()),
        str(df.index[-1].date()),
        len(df)
    ]
})

performance_summary.to_csv('output/performance_summary.csv', index=False)
print("  ✓ output/performance_summary.csv 저장 완료")

# 월별 수익률 저장
monthly_returns_df.to_csv('output/monthly_returns.csv', index=False)
print("  ✓ output/monthly_returns.csv 저장 완료")

# =============================================================================
# 시각화: 하나의 그림에 3개 subplot
# =============================================================================

print("\n[7] 시각화 생성...")

# Figure 생성: 3개의 subplot (세로 배치)
fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 2], hspace=0.3)

# Subplot 1: 누적 자산 곡선 (로그 스케일)
ax1 = fig.add_subplot(gs[0])
ax1.plot(df.index, df['strategy_equity'], label='SMA30 Strategy',
         linewidth=2, color='#2E86AB')
ax1.plot(df.index, df['buyhold_equity'], label='Buy & Hold',
         linewidth=2, alpha=0.7, color='#A23B72')
ax1.set_yscale('log')
ax1.set_ylabel('Cumulative Returns (KRW, log scale)', fontsize=11)
ax1.set_title('SMA30 Strategy Backtest - BTC/KRW', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('')

# Subplot 2: Drawdown 차트 (%)
ax2 = fig.add_subplot(gs[1])
ax2.fill_between(df.index, 0, df['drawdown_pct'], color='red', alpha=0.3)
ax2.plot(df.index, df['drawdown_pct'], color='darkred', linewidth=1)
ax2.set_ylabel('Drawdown (%)', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_xlabel('')

# Subplot 3: 월별 수익률 히트맵
ax3 = fig.add_subplot(gs[2])

# 월별 수익률을 피벗 테이블로 변환
pivot_table = monthly_returns_df.pivot(index='Year', columns='Month', values='Return_pct')
pivot_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# 히트맵 생성
sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            ax=ax3, cbar_kws={'label': 'Monthly Return (%)'},
            linewidths=0.5, linecolor='gray')
ax3.set_ylabel('Year', fontsize=11)
ax3.set_xlabel('Month', fontsize=11)
ax3.set_title('Monthly Returns (%)', fontsize=12)

# 저장
plt.savefig('output/backtest_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ output/backtest_results.png 저장 완료")

# =============================================================================
# 최종 요약 출력
# =============================================================================

print("\n" + "=" * 60)
print("백테스트 완료!")
print("=" * 60)
print(f"\n📊 성과 요약:")
print(f"  • Total Return: {total_return:.2f}x")
print(f"  • CAGR: {cagr:.2%}")
print(f"  • MDD: {mdd:.2f}%")
print(f"  • 최종 자산: {final_equity_vectorized:,.6f}원")
print(f"\n📈 벤치마크 (Buy & Hold):")
print(f"  • Total Return: {benchmark_return:.2f}x")
print(f"  • CAGR: {benchmark_cagr:.2%}")
print(f"\n💾 저장된 파일:")
print(f"  • output/backtest_results.png")
print(f"  • output/performance_summary.csv")
print(f"  • output/monthly_returns.csv")
print("\n" + "=" * 60)
