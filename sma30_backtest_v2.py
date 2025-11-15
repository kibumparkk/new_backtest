"""
SMA30 전략 백테스트 - README 가이드라인 완전 준수 버전
=======================================================

전략: 전일 종가 > 전일 SMA30이면 매수, 아니면 매도/현금
슬리피지: 0.2% (포지션 변경 시마다 적용)
Look-ahead bias 방지: shift(1) 사용
이중 검증: 벡터화 + 반복문
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# ============================================================================
# 설정
# ============================================================================
INITIAL_CAPITAL = 1.0  # 1원
SLIPPAGE = 0.002       # 0.2%
SMA_WINDOW = 30        # 30일 이동평균

# 출력 폴더 생성
os.makedirs('output', exist_ok=True)

print("=" * 70)
print("SMA30 전략 백테스트 (README 가이드라인 완전 준수 버전)")
print("=" * 70)

# ============================================================================
# 1. 데이터 로드 및 준비
# ============================================================================
print("\n[1] 데이터 로딩...")
df = pd.read_parquet('chart_day/BTC_KRW.parquet')
print(f"    데이터 기간: {df.index.min().date()} ~ {df.index.max().date()}")
print(f"    총 데이터: {len(df)}일")

# SMA30 계산
print("\n[2] 기술적 지표 계산...")
df['sma30'] = df['close'].rolling(window=SMA_WINDOW).mean()

# 결측치 제거 (초기 30일은 SMA 계산 불가)
df = df.dropna().copy()
print(f"    유효 데이터: {len(df)}일 (SMA 계산 후)")

# ============================================================================
# 2. 신호 생성 (Look-ahead Bias 방지)
# ============================================================================
print("\n[3] 매매 신호 생성...")
# 전일(t-1) 종가 > 전일(t-1) SMA30 → 당일(t) 매수
# README 가이드라인: shift(1) 사용 필수
df['prev_close'] = df['close'].shift(1)
df['prev_sma30'] = df['sma30'].shift(1)
df['signal'] = (df['prev_close'] > df['prev_sma30']).astype(int)

print(f"    매수 신호 발생일: {df['signal'].sum()}일")
print(f"    현금 보유일: {(df['signal'] == 0).sum()}일")

# ============================================================================
# 3. 백테스트 실행 - 방법 1: 벡터화 (Pandas/NumPy)
# ============================================================================
print("\n[4] 백테스트 실행 - 방법 1: 벡터화")

# 일간 수익률
df['daily_return'] = df['close'].pct_change()

# 포지션 (signal이 이미 t-1 데이터로 계산되었으므로 t일에 그대로 사용)
df['position'] = df['signal']

# 포지션 변화 감지 (매수/매도 타이밍)
df['position_change'] = df['position'].diff()

# 전략 수익률 초기화
df['strategy_return'] = 0.0

# 포지션 보유 중: 시장 수익률 적용
mask_holding = (df['position'] == 1)
df.loc[mask_holding, 'strategy_return'] = df.loc[mask_holding, 'daily_return']

# 매수 시 (0→1): 슬리피지만 차감
mask_buy = (df['position_change'] == 1)
df.loc[mask_buy, 'strategy_return'] = -SLIPPAGE

# 매도 시 (1→0): 슬리피지만 차감
mask_sell = (df['position_change'] == -1)
df.loc[mask_sell, 'strategy_return'] = -SLIPPAGE

# 누적 자산 (복리)
df['equity_vectorized'] = INITIAL_CAPITAL * (1 + df['strategy_return']).cumprod()

final_equity_vec = df['equity_vectorized'].iloc[-1]
print(f"    최종 자산 (벡터화): {final_equity_vec:.8f}원")
print(f"    Total Return: {final_equity_vec / INITIAL_CAPITAL:.4f}x")

# ============================================================================
# 4. 백테스트 실행 - 방법 2: 반복문 (검증용)
# ============================================================================
print("\n[5] 백테스트 실행 - 방법 2: 반복문 (검증용)")

equity = INITIAL_CAPITAL
position = 0  # 0: 현금, 1: 코인
equity_history = []

for i in range(len(df)):
    # 첫날
    if i == 0:
        equity_history.append(equity)
        continue

    # 당일 신호 (이미 전일 데이터로 계산됨)
    target_pos = df['signal'].iloc[i]

    # 당일 가격 변동
    daily_ret = df['daily_return'].iloc[i]

    # 포지션 변경 확인
    if target_pos != position:
        # 슬리피지 발생
        equity *= (1 - SLIPPAGE)
        position = target_pos
    else:
        # 포지션 유지
        if position == 1:
            # 코인 보유 중: 가격 변동 반영
            equity *= (1 + daily_ret)
        # position == 0: 현금 보유 중, 변화 없음

    equity_history.append(equity)

df['equity_loop'] = equity_history
final_equity_loop = df['equity_loop'].iloc[-1]
print(f"    최종 자산 (반복문): {final_equity_loop:.8f}원")
print(f"    Total Return: {final_equity_loop / INITIAL_CAPITAL:.4f}x")

# ============================================================================
# 5. 이중 검증 (Cross-Check)
# ============================================================================
print("\n[6] 이중 검증 (Cross-Check)")
diff = abs(final_equity_vec - final_equity_loop)
diff_pct = (diff / final_equity_vec) * 100

print(f"    벡터화: {final_equity_vec:.8f}원")
print(f"    반복문: {final_equity_loop:.8f}원")
print(f"    차이: {diff:.8f}원 ({diff_pct:.6f}%)")

if diff_pct < 0.01:
    print(f"    ✅ 검증 성공: 차이 < 0.01%")
else:
    print(f"    ❌ 검증 실패: 차이 >= 0.01%")
    print(f"    로직을 재확인해주세요!")

# 이후 분석은 벡터화 결과 사용
final_equity = final_equity_vec

# ============================================================================
# 6. 성과 지표 계산
# ============================================================================
print("\n[7] 성과 지표 계산...")

# 6.1 Total Return (배수)
total_return = final_equity / INITIAL_CAPITAL

# 6.2 CAGR (연평균 복리수익률)
start_date = df.index[0]
end_date = df.index[-1]
total_days = (end_date - start_date).days
years = total_days / 365.25
cagr = (final_equity / INITIAL_CAPITAL) ** (1 / years) - 1

# 6.3 MDD (Maximum Drawdown)
cumulative = df['equity_vectorized']
running_max = cumulative.cummax()
drawdown = (cumulative - running_max) / running_max
mdd = drawdown.min()
mdd_pct = mdd * 100

# 6.4 Buy & Hold 벤치마크
df['buyhold_equity'] = INITIAL_CAPITAL * (1 + df['daily_return']).cumprod()
buyhold_return = df['buyhold_equity'].iloc[-1] / INITIAL_CAPITAL
buyhold_cagr = (df['buyhold_equity'].iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1

# Buy & Hold MDD
bh_cumulative = df['buyhold_equity']
bh_running_max = bh_cumulative.cummax()
bh_drawdown = (bh_cumulative - bh_running_max) / bh_running_max
bh_mdd = bh_drawdown.min()
bh_mdd_pct = bh_mdd * 100

# 6.5 거래 횟수
num_trades = (df['position_change'].abs() > 0).sum()

# 6.6 승률 계산 (옵션)
df['trade_return'] = 0.0
df.loc[mask_holding, 'trade_return'] = df.loc[mask_holding, 'daily_return']
winning_days = (df['trade_return'] > 0).sum()
losing_days = (df['trade_return'] < 0).sum()
win_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0

print(f"    Total Return: {total_return:.2f}x")
print(f"    CAGR: {cagr:.2%}")
print(f"    MDD: {mdd_pct:.2f}%")
print(f"    거래 횟수: {num_trades}회")
print(f"    승률: {win_rate:.2%}")
print(f"\n    [벤치마크: Buy & Hold]")
print(f"    Total Return: {buyhold_return:.2f}x")
print(f"    CAGR: {buyhold_cagr:.2%}")
print(f"    MDD: {bh_mdd_pct:.2f}%")
print(f"    전략 우위: {total_return / buyhold_return:.2f}배")

# ============================================================================
# 7. 월별 수익률 계산
# ============================================================================
print("\n[8] 월별 수익률 계산...")
monthly_returns = df['strategy_return'].resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100
monthly_df = pd.DataFrame({
    'Year': monthly_returns.index.year,
    'Month': monthly_returns.index.month,
    'Return_pct': monthly_returns.values
})
print(f"    총 {len(monthly_df)}개월 데이터")

# ============================================================================
# 8. 시각화 (3개 subplot을 하나의 그림에)
# ============================================================================
print("\n[9] 시각화 생성 (3개 subplot)...")

# Drawdown 시계열 데이터 준비
df['drawdown_pct'] = drawdown * 100
df['bh_drawdown_pct'] = bh_drawdown * 100

# Figure 생성
fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 1.5, 2.5], hspace=0.3)

# --- Subplot 1: 누적 수익률 (로그 스케일) ---
ax1 = fig.add_subplot(gs[0])
ax1.plot(df.index, df['equity_vectorized'],
         label='SMA30 Strategy', linewidth=2.5, color='#2E86AB', alpha=0.9)
ax1.plot(df.index, df['buyhold_equity'],
         label='Buy & Hold', linewidth=2, color='#A23B72', alpha=0.7, linestyle='--')

ax1.set_yscale('log')
ax1.set_ylabel('Cumulative Returns (KRW, log scale)', fontsize=12, fontweight='bold')
ax1.set_title(f'SMA30 Strategy Backtest - BTC/KRW\n' +
              f'Total Return: {total_return:.1f}x | CAGR: {cagr:.1%} | MDD: {mdd_pct:.1f}%',
              fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle=':')
ax1.set_xlabel('')

# --- Subplot 2: Drawdown (%) ---
ax2 = fig.add_subplot(gs[1])
ax2.fill_between(df.index, 0, df['drawdown_pct'],
                  color='#E63946', alpha=0.4, label='Strategy DD')
ax2.plot(df.index, df['drawdown_pct'],
         color='#B91C1C', linewidth=1.5, alpha=0.8)
ax2.axhline(y=0, color='black', linewidth=0.8, linestyle='-')

ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('')
ax2.grid(True, alpha=0.3, linestyle=':')
ax2.legend(loc='lower left', fontsize=10)

# --- Subplot 3: 월별 수익률 히트맵 ---
ax3 = fig.add_subplot(gs[2])

# 피벗 테이블 생성
pivot_table = monthly_df.pivot_table(
    values='Return_pct',
    index='Year',
    columns='Month',
    aggfunc='sum'
)
pivot_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# 히트맵
sns.heatmap(pivot_table,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=0,
            ax=ax3,
            cbar_kws={'label': 'Monthly Return (%)'},
            linewidths=1,
            linecolor='white',
            vmin=-30,  # 색상 범위 조정
            vmax=30)

ax3.set_ylabel('Year', fontsize=12, fontweight='bold')
ax3.set_xlabel('Month', fontsize=12, fontweight='bold')
ax3.set_title('Monthly Returns Heatmap (%)', fontsize=13, fontweight='bold', pad=10)

# 저장
plt.tight_layout()
plt.savefig('output/backtest_results_v2.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✅ output/backtest_results_v2.png 저장 완료")

# ============================================================================
# 9. 결과 저장 (CSV)
# ============================================================================
print("\n[10] 결과 파일 저장...")

# 9.1 성과 요약
performance_df = pd.DataFrame({
    'Metric': [
        'Total Return',
        'CAGR',
        'MDD',
        'Final Equity',
        'Num Trades',
        'Win Rate',
        'Buy&Hold Total Return',
        'Buy&Hold CAGR',
        'Buy&Hold MDD',
        'Strategy vs B&H',
        'Start Date',
        'End Date',
        'Trading Days'
    ],
    'Value': [
        f'{total_return:.4f}x',
        f'{cagr:.4%}',
        f'{mdd_pct:.2f}%',
        f'{final_equity:.6f}',
        f'{num_trades}',
        f'{win_rate:.2%}',
        f'{buyhold_return:.4f}x',
        f'{buyhold_cagr:.4%}',
        f'{bh_mdd_pct:.2f}%',
        f'{total_return / buyhold_return:.2f}x',
        str(start_date.date()),
        str(end_date.date()),
        f'{len(df)}'
    ]
})
performance_df.to_csv('output/performance_summary_v2.csv', index=False)
print(f"    ✅ output/performance_summary_v2.csv 저장 완료")

# 9.2 월별 수익률
monthly_df.to_csv('output/monthly_returns_v2.csv', index=False)
print(f"    ✅ output/monthly_returns_v2.csv 저장 완료")

# 9.3 상세 거래 내역 (옵션)
trade_log = df[df['position_change'].abs() > 0][['close', 'position', 'position_change']].copy()
trade_log['action'] = trade_log['position_change'].apply(
    lambda x: 'BUY' if x == 1 else ('SELL' if x == -1 else 'HOLD')
)
trade_log.to_csv('output/trade_log_v2.csv')
print(f"    ✅ output/trade_log_v2.csv 저장 완료 ({len(trade_log)}건)")

# ============================================================================
# 10. 최종 요약 출력
# ============================================================================
print("\n" + "=" * 70)
print("백테스트 완료!")
print("=" * 70)

print(f"\n📊 전략 성과:")
print(f"   • Total Return:  {total_return:.2f}x  (1원 → {final_equity:.2f}원)")
print(f"   • CAGR:          {cagr:.2%}")
print(f"   • MDD:           {mdd_pct:.2f}%")
print(f"   • 거래 횟수:     {num_trades}회")
print(f"   • 승률:          {win_rate:.2%}")

print(f"\n📈 벤치마크 (Buy & Hold):")
print(f"   • Total Return:  {buyhold_return:.2f}x")
print(f"   • CAGR:          {buyhold_cagr:.2%}")
print(f"   • MDD:           {bh_mdd_pct:.2f}%")

print(f"\n🎯 전략 우위:")
print(f"   • {total_return / buyhold_return:.2f}배 초과 수익")

print(f"\n💾 생성 파일:")
print(f"   • output/backtest_results_v2.png")
print(f"   • output/performance_summary_v2.csv")
print(f"   • output/monthly_returns_v2.csv")
print(f"   • output/trade_log_v2.csv")

print(f"\n⏱️  백테스트 기간:")
print(f"   • {start_date.date()} ~ {end_date.date()}")
print(f"   • {len(df)}일 ({years:.2f}년)")

print("\n" + "=" * 70)
