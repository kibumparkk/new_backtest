"""
Multi-Coin Strategy Finder
다양한 코인에서 벤치마크 돌파 전략 탐색
"""

import pandas as pd
import numpy as np
import warnings
import glob
warnings.filterwarnings('ignore')

INITIAL_CAPITAL = 1
SLIPPAGE = 0.002

def test_coin(coin_file):
    coin_name = coin_file.split('/')[-1].replace('.parquet', '')

    try:
        df = pd.read_parquet(coin_file)
        if len(df) < 250:  # 최소 데이터 길이
            return None

        # 기본 지표
        for p in [10, 15, 20, 25, 30, 35, 40, 50]:
            df[f'sma{p}'] = df['close'].rolling(window=p).mean()
            df[f'ema{p}'] = df['close'].ewm(span=p, adjust=False).mean()

        for p in [10, 15, 20, 25, 30]:
            df[f'mom{p}'] = df['close'] - df['close'].shift(p)
            df[f'roc{p}'] = (df['close'] - df['close'].shift(p)) / df['close'].shift(p) * 100

        # 벤치마크
        df['benchmark'] = (df['close'] > df['sma30']).astype(int)

        def backtest(signal):
            d = df.copy()
            d['sig'] = signal
            d['pos_chg'] = d['sig'].diff()
            d['ret'] = d['close'].pct_change()

            d['strat_ret'] = d['sig'].shift(1) * d['ret'] - abs(d['pos_chg']) * SLIPPAGE
            d['strat_eq'] = INITIAL_CAPITAL * (1 + d['strat_ret']).cumprod()

            d['bench_pos_chg'] = d['benchmark'].diff()
            d['bench_ret'] = d['benchmark'].shift(1) * d['ret'] - abs(d['bench_pos_chg']) * SLIPPAGE
            d['bench_eq'] = INITIAL_CAPITAL * (1 + d['bench_ret']).cumprod()

            d = d.dropna()
            if len(d) == 0 or d['strat_eq'].iloc[-1] <= 0:
                return None, None

            return d['strat_eq'].iloc[-1] / INITIAL_CAPITAL, d['bench_eq'].iloc[-1] / INITIAL_CAPITAL

        best_strategy = None
        best_return = 0
        benchmark_return = 0

        # 전략 테스트
        strategies = [
            # Price vs MA
            *[(df['close'] > df[f'sma{p}'], f'Price>SMA{p}') for p in [10, 15, 20, 25, 30, 35, 40, 50]],
            *[(df['close'] > df[f'ema{p}'], f'Price>EMA{p}') for p in [10, 15, 20, 25, 30, 35, 40, 50]],

            # MA Cross
            (df['ema5'] > df['sma30'], 'EMA5>SMA30'),
            (df['ema10'] > df['sma30'], 'EMA10>SMA30'),
            (df['sma10'] > df['sma30'], 'SMA10>30'),
            (df['sma15'] > df['sma40'], 'SMA15>40'),
            (df['sma20'] > df['sma50'], 'SMA20>50'),

            # Momentum
            *[(df[f'mom{p}'] > 0, f'Mom{p}') for p in [10, 15, 20, 25, 30]],
            *[(df[f'roc{p}'] > 0, f'ROC{p}') for p in [10, 15, 20, 25, 30]],

            # Trend + Momentum
            ((df['close'] > df['sma25']) & (df['mom20'] > 0), 'Trend25+Mom20'),
            ((df['close'] > df['sma30']) & (df['mom20'] > 0), 'Trend30+Mom20'),
            ((df['close'] > df['sma30']) & (df['mom15'] > 0), 'Trend30+Mom15'),
            ((df['close'] > df['sma35']) & (df['mom20'] > 0), 'Trend35+Mom20'),

            # Strong momentum
            ((df['close'] > df['sma30']) & (df['roc20'] > 3), 'Trend30+ROC20>3%'),
            ((df['close'] > df['sma25']) & (df['roc20'] > 3), 'Trend25+ROC20>3%'),
            (df['roc20'] > 3, 'ROC20>3%'),
            (df['roc15'] > 3, 'ROC15>3%'),
            (df['roc20'] > 5, 'ROC20>5%'),
        ]

        for sig, name in strategies:
            tr, br = backtest(sig.astype(int))
            if tr is not None and tr > best_return:
                best_return = tr
                benchmark_return = br
                best_strategy = name

        if best_strategy and best_return > benchmark_return:
            return {
                'coin': coin_name,
                'strategy': best_strategy,
                'return': best_return,
                'benchmark': benchmark_return,
                'outperformance': (best_return / benchmark_return - 1) * 100,
                'days': len(df)
            }

        return None

    except Exception as e:
        return None

# 모든 코인 테스트
coin_files = glob.glob('chart_day/*.parquet')
print(f"총 {len(coin_files)}개 코인 테스트 시작...\n")

winners = []
for i, coin_file in enumerate(coin_files, 1):
    result = test_coin(coin_file)
    if result:
        winners.append(result)
        print(f"🏆 {i:3d}. {result['coin']:15s} | {result['strategy']:25s} | Return: {result['return']:8.2f}x | Benchmark: {result['benchmark']:8.2f}x | Out: +{result['outperformance']:6.2f}%")
    else:
        coin_name = coin_file.split('/')[-1].replace('.parquet', '')
        if i % 10 == 0:
            print(f"   {i:3d}. {coin_name:15s} - 벤치마크 유지")

print("\n" + "="*110)
print(f"벤치마크를 이기는 코인/전략 조합: {len(winners)}개")
print("="*110)

if winners:
    winners.sort(key=lambda x: x['outperformance'], reverse=True)

    print("\n상위 전략:")
    for i, w in enumerate(winners[:20], 1):
        print(f"{i:2d}. {w['coin']:15s} | {w['strategy']:25s} | {w['return']:8.2f}x vs {w['benchmark']:8.2f}x | +{w['outperformance']:6.2f}% | {w['days']}일")

    best = winners[0]
    print(f"\n🎉 최고 성과:")
    print(f"   코인: {best['coin']}")
    print(f"   전략: {best['strategy']}")
    print(f"   수익률: {best['return']:.2f}x")
    print(f"   벤치마크: {best['benchmark']:.2f}x")
    print(f"   초과성과: +{best['outperformance']:.2f}%")

    # 저장
    pd.DataFrame(winners).to_csv('output/multi_coin_winners.csv', index=False)
    print(f"\n결과 저장: output/multi_coin_winners.csv")
else:
    print("\n⚠️  어떤 코인에서도 벤치마크를 이기는 전략을 찾지 못했습니다.")

print("\n테스트 완료!")
