# Blackbird MARIO 모델 개선 보고서

폐루프 보고서(`REPORT.md`)와 별개 트랙. MIT Blackbird 데이터셋에서
`configs/trial8.yaml` 기준선을 개선하고, AirIO와 동일 조건에서 비교한 결과.

전 구간 실험은 `mario/` 아래 코드를 수정하지 않고
`scripts/train_blackbird_v2.py`가 감싸는 방식으로 수행했다(작업 규칙 3).

---

## 0. 정정 — §1의 AirIO 비교는 같은 구간이 아니었다

§1 이하의 MARIO 수치는 전부 `mario/evaluate.py`의 `rollout_trajectory`로 잰 것인데,
이 함수는 궤적의 첫 윈도우만 평가한다. 윈도우를 `window_size`(1000)씩 건너뛰지만
한 윈도우가 만드는 라벨은 `s+14`에서 `s+1013`까지라, 다음 윈도우의 첫 라벨 `s+1014`가
`if t_start in positions`를 통과하지 못하고 체인이 끊긴다. Blackbird eval split에서는
113 pose ≈ 10.2 초만 남고, 이는 궤적 거리의 46~71 %다.

AirIO는 `evaluate_motion.py`가 `integrate(..., gtinit=False, save_full_traj=True)`로
GT 초기 상태 하나에서 시퀀스 전체를 적분한다. 앵커는 같지만 구간이 다르고,
적분 오차는 구간이 길수록 쌓이므로 짧은 쪽이 유리하다. 따라서 §1의
MARIO-vs-AirIO 비교는 성립하지 않는다. MARIO 설정끼리의 비교는 모두 같은 조건이므로
§1의 "무엇을 바꿨나"와 "왜 좋아지나"는 그대로 유효하다.

윈도우 보폭을 999로 두면 `s+1013 = (s+999)+14`로 라벨이 맞물려 궤적 전체를 덮는다.
추론 방식은 동일하다(윈도우마다 상태 초기화, GT 첫 위치에 한 번만 앵커).
`scripts/eval_full_rollout.py`가 이 방식으로 다시 잰 결과가 아래다.

### 전체 궤적 기준 — 시드 4개 평균, AirIO와 동일 구간

| 설정 | 방향성 | params | seen ATE | seen TDE | unseen ATE | unseen TDE |
|---|---|---:|---:|---:|---:|---:|
| **M — causal, 모터 제거** | 단방향 | 95,974 | 0.7230 | 1.245 | **1.1215** | **4.036** |
| **MB — 양방향, 모터 제거** | 양방향 | 131,878 | **0.5513** | **0.967** | 1.1264 | 4.238 |
| AirIO 단방향 | 단방향 | 205,638 | 1.0318 | 1.427 | 2.2853 | 6.957 |
| AirIO 양방향 | 양방향 | 387,014 | 0.4644 | 0.647 | 1.2938 | 4.077 |

AirIO의 TDE는 그 자신이 저장한 ATE를 같은 GT 거리로 나눈 값이다(`airio_tde.json`).

**같은 방향성끼리:**

- 단방향 M vs AirIO 단방향 — seen −29.9 %, **unseen ATE −50.9 %, unseen TDE −42.0 %**
- 양방향 MB vs AirIO 양방향 — seen **+18.7 %(진다)**, unseen ATE −12.9 %, unseen TDE +3.9 %

**무엇이 달라지나.** 단방향 주장은 유지된다 — 공정한 구간에서도 unseen을 절반으로 줄인다.
양방향 주장은 철회해야 한다. §1에서 MB가 AirIO 양방향을 seen에서 이긴 것(0.3837 vs 0.4644)은
짧은 구간의 산물이고, 전체 궤적에서는 진다(0.5513 vs 0.4644).
M의 seen 시드 편차가 ±0.30으로 크므로 seen 단일값은 인용하면 안 된다.

산출물: `results/full_rollout.json`, `results/airio_tde.json`,
`scripts/eval_full_rollout.py`.

---

## 0-B. 브랜치 `no-motor` — 모터 분기를 코드에서 제거

§0/§1의 M은 모터 **입력**을 0으로 넣었을 뿐 `motor_encoder`는 그래프에 남아 있었다.
그것은 없는 것과 다르다 — bias를 가진 conv 뒤에 BatchNorm이 붙으면 0 입력이 고정된
0이 아닌 벡터로 나가므로, 인코더는 192차원 concat에 학습된 상수를 계속 기여했고
데이터를 한 번도 보지 않는 파라미터 12,672개를 들고 있었다.

이 브랜치는 분기를 `mario/` 에서 들어냈다(작업 규칙 3에 대한 예외, 지시로 승인).
`model.py`(인코더 2개, concat 128), `data.py`(thrust_data.csv를 읽지 않음),
`dataset.py`·`train.py`·`evaluate.py`(motor 키/인자 제거), 그리고 런타임 경로
`mario_frames.py`(acc_z로 모터를 지어내던 코드 삭제)·`mario_infer_server.py`
(기본 체크포인트 `nm_s42`).

| | params | seen ATE | unseen ATE |
|---|---:|---:|---:|
| 모터 ON (같은 설정) | 95,974 | 7.243 | 5.361 |
| M (모터 입력 0, 인코더 유지) | 95,974 | 0.723 | 1.122 |
| **NM (분기 삭제)** | **76,582** | **0.589** | 1.122 → **1.503** ※ |

전체 궤적, 4시드 평균. 모터 ON 대비 seen −91.9 %, unseen −79.1 %.

**※ 구간이 바뀌었다 — 성능 저하가 아니다.** 기존 `data.py`는 시퀀스를 thrust 파일의
시간 범위로 클리핑했다. unseen 비행들의 thrust는 IMU보다 5~6초 늦게 시작하므로
**unseen 평가에서 앞부분이 잘려 있었다**(seen은 범위가 같아 영향 없음). thrust를 읽지
않게 되면서 클리핑이 사라져 unseen 시퀀스가 평균 29.2 % 길어졌고, 적분 구간이 늘어
ATE가 1.122 → 1.503이 됐다. 같은 체크포인트를 더 긴 궤적에 적용한 결과다.
seen은 0.589로 변하지 않는다. **이 브랜치의 unseen 수치는 이전 수치와 직접 비교하면 안 된다.**

**호환성.** 모터 시대의 체크포인트(trial8, m_*, mb_*, b*)는 새 모델 클래스로 읽히지 않는다.
`runs/nm_s{42,1,2,3}` 만 사용 가능하다. our2 로더와 `probe_motor_input.py`는
함께 제거했다.

---

## 1. 최종 수치 (10.2 초 구간 기준 — §0 참조) — 시드 4개(42/1/2/3) 평균

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

§1 표의 MARIO 수치는 10.2 초 구간 기준이므로 AirIO 열과 직접 비교하지 말 것(§0).
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
