# 벤치마크 계산 오류 분석

## 문제점 발견

### 현재 코드 (잘못됨)

```python
# Line 92: 벤치마크 시그널 생성
df['benchmark_signal'] = (df['close'].shift(1) > df['sma30'].shift(1)).astype(int)

# Line 113: 벤치마크 수익률 계산
df_bt['benchmark_return'] = df_bt['benchmark_signal'].shift(1) * df_bt['daily_return'] - ...
```

### 문제 분석

**벤치마크가 총 2일 지연**되어 계산되고 있습니다:

1. 시그널 생성 시점에서 `shift(1)` → 전일 종가와 전일 SMA30 비교
2. 수익률 계산 시점에서 또 `shift(1)` → 전일 시그널 사용

**결과:** t-2일의 정보로 t일의 포지션을 결정
- 예: 11월 13일 종가와 SMA30을 비교 → 11월 14일 시그널 생성 → 11월 15일에 포지션 진입

### 전략 코드 비교

```python
# Momentum 20 전략 (올바름)
df['momentum'] = df['close'] - df['close'].shift(20)  # 당일 close 사용
signal = (df['momentum'] > 0).astype(int)              # 당일 시그널 생성
df_bt['strategy_return'] = df_bt['signal'].shift(1) * df_bt['daily_return']  # 1일 지연만 적용
```

**전략은 1일 지연**만 있음:
- 예: 11월 14일 momentum 계산 → 11월 14일 시그널 생성 → 11월 15일에 포지션 진입

### 영향

**벤치마크 성과가 과소평가**되었을 가능성이 높습니다.
- 벤치마크는 2일 늦은 정보로 거래
- 전략은 1일 늦은 정보로 거래
- **불공정한 비교**

## 올바른 수정 방법

### 옵션 1: 벤치마크 시그널을 당일 정보로 생성

```python
# 시그널 생성: shift 제거
df['benchmark_signal'] = (df['close'] > df['sma30']).astype(int)

# 수익률 계산: shift(1) 유지
df_bt['benchmark_return'] = df_bt['benchmark_signal'].shift(1) * df_bt['daily_return'] - ...
```

### 옵션 2: 수익률 계산에서 shift 제거

```python
# 시그널 생성: shift 유지
df['benchmark_signal'] = (df['close'].shift(1) > df['sma30'].shift(1)).astype(int)

# 수익률 계산: shift 제거
df_bt['benchmark_return'] = df_bt['benchmark_signal'] * df_bt['daily_return'] - ...
```

**두 방법 모두 동일한 결과**를 제공합니다. 옵션 1이 더 직관적입니다.

## 재테스트 필요성

이 오류로 인해:
1. 벤치마크 성과가 실제보다 **낮게** 계산됨
2. Momentum 20 전략의 상대적 우위가 **과대평가**되었을 가능성
3. 일부 전략들도 동일한 이슈가 있을 수 있음

**모든 전략을 재테스트해야 합니다.**
