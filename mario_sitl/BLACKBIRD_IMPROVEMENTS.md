# Blackbird MARIO 모델 개선 보고서

폐루프 보고서(`REPORT.md`)와 별개 트랙. MIT Blackbird 데이터셋에서
`configs/trial8.yaml` 기준선을 개선하고, AirIO와 동일 조건에서 비교한 결과.

전 구간 실험은 `mario/` 아래 코드를 수정하지 않고
`scripts/train_blackbird_v2.py`가 감싸는 방식으로 수행했다(작업 규칙 3).

---

## 1. 최종 수치 — 시드 4개(42/1/2/3) 평균

| 설정 | seen ATE | seen TDE | unseen ATE | unseen TDE |
|---|---|---|---|---|
| MARIO baseline (trial8) | 0.4251 | 1.3114 | 1.4992 | 7.6992 |
| **M — causal, 모터 제거** | 0.4370 | 1.3295 | **0.9311** | **5.4029** |
| **MB — 양방향, 모터 제거** | **0.3837** | **1.1494** | **0.9525** | **5.7044** |
| AirIO 양방향 (387k) | 0.4644 | — | 1.2938 | — |
| AirIO 단방향 (206k) | 1.0318 | — | 2.2853 | — |

- **MB는 baseline 대비 네 지표 전부 개선** (seen −9.7 % / −12.4 %, unseen −36.5 % / −25.9 %)
- **M(실시간 배포 가능한 causal)이 AirIO 양방향을 seen −5.9 %, unseen −28.0 %로 이긴다**
- 73개 런 중 unseen ATE 상위 8위가 전부 모터 제거 런. 모터를 쓴 65개 런은 하나도 못 넘었다

AirIO 수치는 AirIO 자신의 `evaluate_motion.py:133`이 저장한 `sqrt(mean(pos_dist²))`이고,
`mario/evaluate.py:72`의 ATE와 같은 정의다. AirIO가 화면에 찍는 `pos_err`는
오차 노름의 산술 평균(`velocity_integrator.py:133`)이라 다른 값이므로 혼용하면 안 된다.

## 2. 무엇을 바꿨나

한 가지뿐이다 — **추력(모터) 입력 채널을 0으로 채웠다.**

```python
def drop_motor(seq: dict) -> None:
    seq["motor"] = torch.zeros_like(seq["motor"])
```

아키텍처, lr, weight_decay, epochs, huber_delta, uncertainty_weight 모두 동일.
`motor_encoder`도 그대로 남아 있어 파라미터 수가 95,974로 변하지 않는다 —
입력이 상수가 되어 그 경로가 비활성화될 뿐이다.

## 3. 왜 좋아지나 — 추력 채널은 비행 식별자였다

Blackbird `thrust_data.csv`는 20개 파일 전수 확인 결과 ch0/ch1이 정확히 0이고
ch2만 활성인, 질량 정규화된 z축 집합추력이다. `/9.80665` 후 비행별 평균:

| 궤적 | split | 평균 | 표준편차 |
|---|---|---|---|
| oval | unseen | −1.225 | 0.050 |
| halfMoon | seen | −1.209 | 0.094 |
| star | seen | −1.192 | 0.157 |
| clover | seen | −1.123 | 0.087 |
| sphinx | unseen | −1.099 | 0.172 |
| winter | seen | −1.084 | 0.057 |
| sid | unseen | −1.035 | 0.042 |
| bentDice | unseen | −1.031 | 0.096 |
| egg | seen | −1.014 | 0.237 |
| ampersand | unseen | −1.006 | 0.048 |

**비행 내 변동(std 0.04~0.24)이 비행 간 폭(0.22)보다 작다.** 호버 추력 근처의
거의 일정한 값이라 동역학 정보는 빈약한데 "어느 비행인지" 식별력은 강하다 —
예측에 쓸모없고 암기에 완벽한 조합이다.

네트워크는 DC 값으로 학습 비행을 판별하고 비행별 보정을 적용한다. unseen에서는
그 값이 학습 비행 중 하나 근처에 떨어져 **엉뚱한 보정이 걸린다.** `oval`(−1.225)이
학습 비행 `halfMoon`(−1.209)과 사실상 같은 DC를 갖는 것이 그 예다.

이 오적용은 매 스텝 같은 방향으로 작용하는 계통 편향이라 적분에서 누적된다:

| | 창 단위 RMSE (4시드) | 적분 후 unseen ATE |
|---|---|---|
| 모터 사용 | 0.01747 (0.01718~0.01777) | 1.5843 |
| 모터 제거 | 0.02020 (0.01990~0.02044) **나빠짐** | 0.9311 (**−41 %**) |

두 분포는 시드 4개에서 전혀 겹치지 않는다. 국소 예측은 4/4 시드에서 나빠졌는데
궤적 오차는 4/4 시드에서 좋아졌다. 지표 아티팩트라면 나올 수 없는 방향이고,
계통 편향 제거의 전형적 서명이다.

이는 `mario/train.py:108`이 창 RMSE로 `best.pt`를 고르는 것이 왜 무력했는지도 설명한다.
선택 지표가 이 편향을 거의 잡아내지 못한다.

**범위 제한**: 이는 Blackbird의 이 채널에 대한 결론이지 "모터 입력이 무용하다"가 아니다.
PX4에서 4개 모터 개별 출력을 받으면 기동 중 차동 성분이 실려 성질이 다르다.
SITL 쪽에 적용할 때는 이 구분이 필요하다.

## 4. 기각된 가설 — 전부 4시드 또는 4런

| 가설 | 근거 | 결과 |
|---|---|---|
| 검증 궤적 홀드아웃으로 과적합 차단 | `data.py:134`가 선택셋을 cfg.seen에서 만든다 | seen +21 %, unseen +37 % |
| 바디 z축 회전 증강 | 전 비행 yawForward라 기수가 궤적에 묶임 | seen +35 %, unseen +5 % |
| 용량 부족 (170k / 363k) | AirIO가 2~4배 크다 | unseen +18~31 % |
| huber δ 0.05로 RMS 정렬 | MARIO δ=0.002는 사실상 L1 | unseen +24~33 % |
| 양방향성이 AirIO 우위의 원인 | AirIO는 GRU 2단 모두 양방향 | MARIO 양방향은 unseen −3.6 %에 그침 |

홀드아웃 실험은 유용한 부산물을 남겼다: 홀드아웃 궤적 RMSE가 epoch 20에 정체한 뒤
100까지 **오르지 않는다.** 과적합이 없다는 뜻이고, 이 모델이 데이터/용량 과잉이 아니라
부족 쪽임을 보여준다. 이후 용량 증대도 실패했으므로 병목은 둘 다 아니었다.

**양방향성은 AirIO에서만 결정적이다.** AirIO uni(2.2853) vs bi(1.2938)는 unseen 43 % 차이인데
MARIO는 causal(0.9311) vs bi(0.9525)로 사실상 동률이다. Mamba의 상태공간 재귀가 GRU와 달리
단방향에서도 충분한 문맥을 유지하며, 그래서 실시간 배포 가능한 causal 모델이 살아남는다.

## 5. 방법론 주의사항

이 모델은 학습이 혼돈적이다 — lr을 0.011 % 바꾸면 seen ATE가 14 % 움직인다.
같은 시드면 결과가 0.08 % 내로 재현되지만, 시드를 바꾸면 baseline unseen TDE가
7.36~7.99로 ±8 % 흩어진다.

이 때문에 단일 시드 결론은 이 프로젝트에서 세 번 뒤집혔다(B1 −8.0 %, B3 −32.0 %,
"B2가 seen·unseen 동시 개선"). **모든 주장은 시드 4개 평균으로 판정하고,
개별 시드 최고값은 배포 체크포인트의 실측값으로만 쓴다.**

모터 제거는 이 기준을 통과한다 — 4/4 시드에서 개선되고 최소 개선폭 −29 %로
시드 잡음의 3배 이상이다.

## 6. 산출물

```
runs/m_s{42,1,2,3}/          M  — causal, 모터 제거 (배포 후보)
runs/mb_s{42,1,2,3}/         MB — 양방향, 모터 제거 (오프라인 최고)
runs/b8_val_s*/              기각: 검증 궤적 홀드아웃
runs/b8_rot_s*/              기각: 회전 증강
runs/c{1,2}_s*/              기각: 용량 증대
runs/h_s*/                   기각: huber δ 0.05
mario_sitl/results/
├── final_comparison.json    최종 4시드 평균 + AirIO
├── all_runs.csv / .json     73개 런 전수
├── arch_comparison.json     causal vs 양방향
├── airio_bi_result.json     AirIO 양방향 궤적별 ATE
└── airio_uni_result.json    AirIO 단방향 궤적별 ATE
mario_sitl/scripts/
├── train_blackbird_v2.py    실험 하네스 (--no-motor, --arch, --rot-aug, --val-traj,
│                            --d-model/--expand/--num-layers, --huber-delta)
└── bimamba_model.py         BiMambaDispNet (131,878 params)
```

AirIO 단방향 변형은 `Air-IO/model/code.py`에 `AIRIO_UNIDIRECTIONAL=1` 환경변수
스위치로 구현했다(MARIO 저장소 밖).
