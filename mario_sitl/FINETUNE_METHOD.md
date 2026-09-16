# 파인튜닝 방법 — 재현용 상세 기록

무엇을, 얼마나, 어떻게 학습시켰는지를 재현 가능한 수준으로 적는다. 결과 해석은
`APPLICATION_VALIDATION.md`와 `PAPER_NOTES.md`에 있고, 이 문서는 절차만 다룬다.

전이 대상이 둘이고 목적이 달라 방법도 다르다.

| | 트랙 A — PX4 x500 (SITL) | 트랙 B — EuRoC MAV |
|---|---|---|
| 목적 | 새 기체에서 **작동하게 만들기** | **무엇이 전이되는지 측정하기** |
| 스크립트 | `scripts/finetune_sitl.py` | `scripts/finetune_transfer.py` |
| 방법 | 전체 파라미터 파인튜닝 1종 | 동결 범위 4종 절제 비교 |
| 데이터 | 자체 수집 20.0분 | 공개 split 12.25분 중 9.80분 |

---

## 0. 공통 설정

### 0.1 윈도우 구성

모든 학습은 `mario/dataset.py`의 `BlackbirdDispDataset`이 만드는 슬라이딩 윈도우 단위다.

| 항목 | 값 | 의미 |
|---|---:|---|
| `window_size` | 1000 | 입력 10 초 (100 Hz) |
| `train_step_size` | 3 | 학습 윈도우 보폭 — 인접 윈도우가 997 샘플 공유 |
| `test_step_size` | 10 | 평가 윈도우 보폭 |
| `label_start_index` | 14 | 윈도우 내 첫 라벨 위치 |
| `label_stride` | 9 | 라벨 간격 = CNN 인코더 다운샘플 배율 (3×3) |
| `dt` | 0.01 | 100 Hz |

윈도우 하나가 90 ms 단위 바디 프레임 변위 **111개**를 만든다.

### 0.2 손실과 최적화 (`configs/trial8.yaml`)

```
loss = 1000.0 · huber(pred − gt, δ=0.002) + 1e-4 · NLL(residual.detach(), cov)
```

| 항목 | 값 |
|---|---|
| optimizer | Adam, `weight_decay` 1e-3 |
| batch_size | 128 |
| grad clip | 1.0 |
| scheduler | `ReduceLROnPlateau`, factor 0.2, patience 5, min_lr 1e-5 |
| 기본 lr (처음부터 학습) | 2.5757e-4 |

`huber_delta=0.002`는 변위 스케일(90 ms 동안 수 cm) 대비 매우 작아 사실상 L1이다.
`uncertainty_weight=1e-4`가 `loss_weight=1000.0`에 맞서 있어 공분산 항의 기울기 기여는
거의 0이며, 이 때문에 cov 헤드가 학습되지 않는다(`REPORT.md` §5.1). 파인튜닝에서도
이 값을 바꾸지 않았다 — 변수를 하나로 유지하기 위해서다.

### 0.3 하드웨어

NVIDIA RTX A6000. 트랙 A는 단일 GPU, 트랙 B는 GPU 3–7에 작업 분산.

---

## 1. 트랙 A — PX4 x500 (SITL)

### 1.1 데이터 수집 — 두 번 모았고, 그 차이가 결과를 갈랐다

Gazebo + PX4 SITL에서 offboard 미션 노드로 비행시키며 IMU·자세·위치를 기록했다
(`mario_ros/varied_flight_node.py`, `record_flight_node.py`). 100 Hz.

**동일한 분량을 두 가지 방식으로 모았다.**

| | 1차 (`results/dataset/`) | 2차 (`results/dataset_agg/`) |
|---|---|---|
| 셋포인트 주는 법 | 일정 속도로 **연속 이동**(걷기) | **계단식 점프**(스텝) |
| 비행 수 | 6 (학습 5 / 테스트 1) | 6 (학습 5 / 테스트 1) |
| 비행당 길이 | 284.9 s | 284.8–284.9 s |
| 지상 구간 제거 후 학습량 | **20.0 분** | **20.0 분** |
| 평균 속도 | 0.29–0.62 m/s | **2.75–3.29 m/s** |
| 0.2 m/s 미만 비율 | 30–66 % | 22–33 % |
| \|accel\| > 1 m/s² 비율 | 13–26 % | **88–97 %** |
| 항력 신호 | 0.13–0.18 | **0.46–0.53** |

**분량이 같습니다 — 20.0분 대 20.0분.** 다른 것은 비행 체제뿐이고, 결과는 다음과 같이
갈렸다(held-out `flight_6`):

| 학습 데이터 | cos | TDE |
|---|---:|---:|
| 1차 (걷기) | −0.002 | 56.5 % |
| **2차 (스텝)** | **+0.964** | **9.70 %** |

즉 **파인튜닝 방법이 아니라 수집 설계가 지배 변수였다.** 등속 비행에는 속도 정보가
물리적으로 없기 때문이다(비추력 a − g는 등속에서 중력만 읽는다).

참고로 다른 미션 형태로 더 길게 모은 세트도 있다 —
`dataset_pat/` 12비행 55.0분, `dataset_yf/` 13비행 57.5분. 분량이 2.5배지만 체제가
1차와 같아 개선되지 않았다(`sitl_pat` best RMSE 0.129, `sitl_yf` 0.106).

### 1.2 전처리 세 가지

**(a) 지상 구간 제거** — `airborne_slice()`

이륙 전후 주차 구간을 잘라낸다. 기준은 이륙 지점 대비 **고도 0.5 m**. 모터가 꺼져
진동도 움직임도 정확히 0인 구간이라 손실을 지배하면서 호버에 대해 아무것도 가르치지
않는다. 비행 샘플이 2000개 미만이면 그 기록을 **통째로 버린다** — 무장에 실패한 비행
하나가 이 경로로 들어와 "항상 0 근처로 답하기" 편향을 만든 적이 있다.

284.9 s → 235–249 s (약 83 %)가 남는다.

**(b) 비행 단위 홀드아웃**

윈도우를 섞어 나누면 인접 윈도우가 1000개 중 **997개를 공유**하므로 사실상 같은
데이터가 양쪽에 들어간다. 그래서 **비행 통째로** 뺀다. 목록에 고르게 분산시켜
여러 궤적 모양을 걸치게 했고(`step = len//(k+1)`), 파일명은 `flight_(\d+)`를 정수로
파싱해 정렬한다 — 사전순이면 `flight_10`이 `flight_2`보다 앞서 의도와 다른 비행이
빠진다.

**(c) 속도 히스토그램 평탄화** — `balance_by_speed()` (선택 옵션)

수집 데이터가 저속에 쏠려 있어 그대로 학습하면 편향이 **교체**될 뿐이다:

```
주차 시 유령 속도   1.06 → 0.21 m/s   개선
순항 속도 과소추정   40 % → 51 %      악화
```

그래서 윈도우별 평균 속도(`gt_disp.norm().mean() / 0.09`)로 **등폭 8구간**을 만들고
**중앙값 구간 크기**에 맞춰 레벨링한다 — 붐비는 저속 구간은 서브샘플링, 드문 고속
구간은 복원추출로 오버샘플링.

> 분위수 경계는 쓰면 안 된다. 정의상 등개수라 평탄화가 무의미하다 — 첫 시도가
> 38,357 → 38,352로 아무것도 바꾸지 않아서 알았다.

### 1.3 학습 설정 — 실제 실행한 런 전부

출발 체크포인트는 전부 `runs/trial8_100ep/best.pt` (Blackbird 100 epoch 학습본).

| run | 데이터 | 비행 수 | Blackbird 혼합 | lr | epochs | 학습 윈도우 | best RMSE | TDE |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `sitl_finetune` | 1차 | 5 | **혼합** | 5e-5 | 30 | 49,717 | 0.02397 | 59.3 % |
| `sitl_only` | 1차 | 5 | 없음 | 1e-4 | 60 | 38,357 | 0.02617 | 56.5 % |
| `sitl_balanced` | 1차 | 5 | 없음 | 1e-4 | 25 | 7,016 ※ | 0.05214 | 76.8 % |
| `sitl_pat` | pat | 10 | 없음 | 1e-4 | 30 | 70,367 | 0.12909 | — |
| `sitl_pat2` | pat | 10 | 없음 | 5e-5 | 40 | 70,367 | 0.13624 | — |
| `sitl_yf` | yf | 11 | 없음 | 1e-4 | 30 | 82,825 | 0.10567 | — |
| **`sitl_agg`** | **2차** | **5** | 없음 | **1e-4** | **30** | **38,280** | 0.08620 | **9.70 %** |

※ 속도 평탄화 적용 후 윈도우 수.

**채택은 `sitl_agg`다.** 학습 시간 206 초(30 epoch).

lr 5e-5는 처음부터 학습할 때의 2.58e-4보다 낮게 잡은 값으로, "재학습이 아니라 조정"
이라는 의도다. 1e-4는 그 절충이다.

**Blackbird 혼합 여부**: `sitl_finetune`만 Blackbird `seen` 5궤적을 섞었다. 목적은
저속 편향을 잡으면서 고속 비행이 퇴행하지 않게 하는 것이었는데, 2차 수집 데이터가
이미 고속을 포함하므로 `sitl_agg`에서는 불필요해졌다.

**자세 입력**: 전부 `--attitude gt`로 학습했다. 평가 시 EKF2 자세로 바꿔도
ATE 2.332 → 2.314로 차이가 없다(§`APPLICATION_VALIDATION.md` 2.2).

### 1.4 재현 명령

```bash
PY=/src/gs25122/miniconda3/envs/mamba/bin/python
$PY mario_sitl/scripts/finetune_sitl.py \
    --dataset mario_sitl/results/dataset_agg \
    --ckpt   runs/trial8_100ep/best.pt \
    --out    runs/sitl_agg \
    --epochs 30 --lr 1e-4 --holdout 1 --no-blackbird --attitude gt
```

윈도우 구축은 윈도우당 pypose 연산 111회라 CPU로 수십 분이 걸린다. 입력 목록·자세
소스·윈도우 파라미터를 md5로 묶어 `dataset/.windows_{tag}_{key}.pt`에 캐시한다.

---

## 2. 트랙 B — EuRoC MAV

### 2.1 데이터 — 공식 split을 그대로 사용

`train_list.txt` / `test_list.txt`가 데이터셋에 동봉되어 있어 split은 우리가 고른 것이
아니다. `val_list.txt`는 `train_list.txt`와 **동일**하므로 검증에 쓰지 않았다(§2.3).

**학습 split (6 시퀀스)** — 200 Hz 원본을 dt=0.01로 리샘플한 뒤

| 순서 | 시퀀스 | 샘플 | 길이 | 거리 | 평균 속도 |
|---:|---|---:|---:|---:|---:|
| 1 | V1_02_medium | 8,350 | 83.5 s | 75.9 m | 0.91 m/s |
| 2 | V2_03_difficult | 11,484 | 114.8 s | 86.1 m | 0.75 m/s |
| 3 | MH_05_difficult | 11,105 | 111.0 s | 97.6 m | 0.88 m/s |
| 4 | MH_01_easy | 18,190 | 181.9 s | 80.6 m | 0.44 m/s |
| 5 | V2_01_easy | 11,200 | 112.0 s | 36.5 m | 0.33 m/s |
| 6 | MH_03_medium | 13,150 | 131.5 s | 130.9 m | 1.00 m/s |
| | **합계** | **73,479** | **734.8 s = 12.25 분** | **507.5 m** | |

**테스트 split (5 시퀀스)** — 학습에도 체크포인트 선택에도 쓰지 않았다

| 시퀀스 | 샘플 | 길이 | 거리 | 평균 속도 |
|---|---:|---:|---:|---:|
| MH_02_easy | 14,996 | 150.0 s | 73.5 m | 0.49 m/s |
| MH_04_difficult | 9,875 | 98.8 s | 91.7 m | 0.93 m/s |
| V1_03_difficult | 10,465 | 104.7 s | 79.0 m | 0.75 m/s |
| V2_02_medium | 11,544 | 115.4 s | 83.2 m | 0.72 m/s |
| V1_01_easy | 14,470 | 144.7 s | 58.4 m | 0.40 m/s |
| **합계** | **61,350** | **613.5 s = 10.22 분** | **385.7 m** | |

`V1_01_easy`만 GT가 20 Hz다(나머지 200 Hz). 같은 보간 경로를 타지만 분리해 기록한다.

**다운로드 경로 주의.** 원본 호스트 `robotics.ethz.ch`는 응답하지 않는다. ASL이
2025-12-18에 ETH Research Collection(DOI `10.3929/ethz-b-000690084`)으로 옮기고 파일
서버를 껐는데, 거기에는 스테레오 영상이 포함된 6–12 GB 아카이브뿐이고 중첩 zip이
deflate라 CSV만 골라 받을 수 없다. IMU+GT만 추린 20 MB 판을 사용했다:
`github.com/Air-IO/Air-IO/releases/download/datasets/EuRoC-Dataset.zip`

### 2.2 전처리 — 바디 프레임 정렬 (필수)

두 데이터셋의 IMU 장착 규약이 다르다. 정지 시 바디 프레임 가속도:

| | 바디 가속도 | 중력 방향 |
|---|---|---|
| Blackbird | `[−0.24, +0.95, −11.19]` | −z |
| EuRoC | `[+9.0, 0, −3.6]` | **+x**, Blackbird 대비 70.8° |

EuRoC의 이 방향은 11개 시퀀스에서 산포 **0.53°**로 일정하다 — 비행 특성이 아니라 고정
장착이다. 변위 추정은 바디 상수 회전에 등변이므로(acc·gyro·라벨을 함께 돌리면 물리가
불변) 이 정렬은 **좌표 변환이지 적합이 아니다.**

중력 방향만으로는 중력축 회전(요)이 정해지지 않아 **30° 격자로 훑고, 최적값을
학습 split에서 골라 테스트에 적용**했다(`--yaw-deg 185`).

```python
EUROC_GRAVITY_DIR     = [ 0.94200,  0.00018, -0.33538]   # 학습 split에서 측정
BLACKBIRD_GRAVITY_DIR = [-0.00970,  0.10890, -0.99400]
```

| zero-shot | ATE |
|---|---:|
| 정렬 없음 | 21.24 m |
| 중력 정렬만 (요 0°) | 15.19 m |
| **중력 정렬 + 요 185°** | **2.61 m** |

**샘플레이트도 맞춰야 한다.** 200 Hz 원본(dt=0.005)을 그대로 넣으면 3.37 m로 학습
분포에 맞춘 100 Hz(2.46 m)보다 나쁘다. 전부 dt=0.01을 쓴다.

### 2.3 검증 분할 — 공식 테스트를 건드리지 않기 위해

`val_list.txt`가 `train_list.txt`와 같으므로 쓸 수 없고, 공식 테스트는 아껴야 한다.
그래서 **각 학습 시퀀스의 꼬리 20 %**를 검증으로 떼어냈다(`split_tail`).

```python
VAL_TAIL = 0.2      # 꼬리 20 %가 검증
VAL_GAP  = 1200     # 학습부와 검증부 사이에 버리는 샘플
```

`VAL_GAP`이 필요한 이유: 윈도우가 1000 샘플이므로 경계를 걸친 윈도우는 학습과 검증
양쪽에 나타난다. 1200 샘플(12 초)을 버려 그것을 차단한다.

| 시퀀스 | 전체 | 학습부 | 검증 꼬리 |
|---|---:|---:|---:|
| V1_02_medium | 8,350 | 5,480 (54.8 s) | 1,670 (16.7 s) |
| V2_03_difficult | 11,484 | 7,987 (79.9 s) | 2,297 (23.0 s) |
| MH_05_difficult | 11,105 | 7,684 (76.8 s) | 2,221 (22.2 s) |
| MH_01_easy | 18,190 | 13,352 (133.5 s) | 3,638 (36.4 s) |
| V2_01_easy | 11,200 | 7,760 (77.6 s) | 2,240 (22.4 s) |
| MH_03_medium | 13,150 | 9,320 (93.2 s) | 2,630 (26.3 s) |
| **합계** | 73,479 | **51,583 = 8.60 분** | 14,696 = 2.45 분 |

윈도우로는 **학습 15,191개 / 검증 865개**(보폭 3 / 10).

> **표기 주의**: `TRANSFER_STUDY.md`가 쓰는 "9.8분"은 전체의 명목 80 %
> (734.8 s × 0.8 = 587.8 s)이고, `VAL_GAP`으로 버리는 6 × 12 s = 72 s를 빼지 않은
> 값이다. 실제로 윈도우가 만들어진 구간은 **8.60분**이다. 아래 데이터 효율 표는
> 두 값을 모두 적는다.

### 2.4 4개 arm — 동결 범위만 다르다

| arm | 학습 파라미터 | 동결 | 무엇을 묻는가 |
|---|---:|---|---|
| `head` | 17,414 (18.1 %) | 인코더 + Mamba 전부 | 인코더 특징이 전이됐고 출력 매핑만 틀렸나? |
| `trunk` | 49,414 (51.5 %) | CNN 인코더 3개 | 원시 IMU 프론트엔드는 재사용 가능한가? |
| `full` | 95,974 (100 %) | 없음 | |
| `scratch` | 95,974 (100 %) | 없음, **무작위 초기화** | 사전학습이 뭐라도 벌어줬나? |

**동결이 진짜 동결이게** — `requires_grad_(False)`만으로는 BatchNorm running statistics가
계속 갱신된다. 그러면 "동결"이 거짓말이 되고 `head` arm이 크레딧 없는 특징 적응을
하게 된다. 그래서 `module.train`을 no-op 람다로 치환해 eval 모드에 못 박는다:

```python
def pin_eval(modules):
    for module in modules:
        module.eval()
        module.train = lambda self_mode=True, _m=module: _m
```

### 2.5 체크포인트 선택 — 창 RMSE가 아니라 롤아웃 ATE

`mario/train.py:108`은 창 단위 RMSE로 `best.pt`를 고른다. 그런데 그것은 0.09 초짜리
변위 하나의 오차라, **작은 계통 편향은 창 단위로 거의 보이지 않고 적분하면 지배적이
된다.** Blackbird에서 이미 확인된 현상이고(창 RMSE는 나빠지는데 궤적 ATE는 −41 %),
전이에서는 더 심하다 — **100 epoch까지 창 RMSE는 계속 떨어지는데 테스트 ATE는
zero-shot 값 위로 올라갔다.**

그래서 `train_select_ate()`는 매 epoch 떼어둔 꼬리를 **롤아웃해서 ATE로** 고른다.
`ReduceLROnPlateau`도 ATE를 본다. 최적화 자체(손실·옵티마이저·클리핑)는
`mario.train.train`과 동일하다.

절제를 위해 `--select rmse`도 돌렸고, 결과는 결론이 나지 않았다(차이가 대부분 시드
편차 안). 채택하되 개선 수단으로 주장하지 않는다.

### 2.6 학습률

```
scratch : 2.5757e-4   (원본 lr 그대로)
그 외    : 2.5757e-5   (원본의 1/10)
```

lr이 순위를 만든 것이 아님을 확인하기 위해 **초기화 × lr 2×2**를 돌렸다. 낮은 lr에서도
순위가 유지된다(scratch 6.071 < trunk 7.722 ≈ full 7.749 < head 7.958).

epochs 60, 시드 42/1/2/3.

### 2.7 데이터 효율 축

`--n-train N`으로 학습 시퀀스를 앞에서부터 N개만 쓴다. 순서는 `train_list.txt` 그대로.

| N | 사용 시퀀스 | 명목 80 % | **실제 학습 구간** |
|---:|---|---:|---:|
| 1 | V1_02_medium | 1.11 분 | **0.91 분** |
| 2 | + V2_03_difficult | 2.64 분 | **2.24 분** |
| 4 | + MH_05_difficult, MH_01_easy | 6.55 분 | **5.75 분** |
| 6 | + V2_01_easy, MH_03_medium | 9.80 분 | **8.60 분** |

`scratch` / `head` / `full` 세 arm × 4개 데이터량 × 4시드 = 48런.

### 2.8 매트릭스 실행

`scripts/run_transfer_matrix.sh`가 (arm × 시드) 조합을 GPU 3–7에 패킹한다.

```bash
# arm 비교 (4 arm × 4 seed)
./run_transfer_matrix.sh euroc 60 "42 1 2 3" "head trunk full scratch"

# 데이터 효율 (n_train 별로 반복)
for n in 1 2 4; do
  ./run_transfer_matrix.sh euroc 60 "42 1 2 3" "head full scratch" "" "" ate $n
done
```

작업 시작을 20초씩 어긋내 윈도우 캐시 쓰기 경합을 피한다. 각 작업이 자기
`results.json`을 쓰므로 실패해도 그 작업만 손실된다.

전 런 재채점은 `reeval_transfer_full.py`가 전체 궤적 롤아웃으로 다시 수행하고
(`eval_full_rollout.py`의 보폭 999 방식), `summarise_transfer.py`가 시드 평균을 낸다.

### 2.9 재현 명령 (단일 런)

```bash
PY=/src/gs25122/miniconda3/envs/mamba/bin/python
CUDA_VISIBLE_DEVICES=3 $PY mario_sitl/scripts/finetune_transfer.py \
    --mode scratch --dataset euroc --seed 42 --epochs 60 \
    --align gravity --yaw-deg 185 --select ate \
    --init runs/m_s42/best.pt \
    --out  runs/transfer/euroc_scratch_s42
```

> **호환성 주의**: `--init`의 기본값 `runs/m_s42/best.pt`는 모터 인코더를 가진
> 95,974-파라미터 모델이다. 브랜치 `no-motor`는 그 분기를 제거했으므로 이 체크포인트가
> 로드되지 않는다. 그 브랜치에서 재현하려면 `runs/nm_s42/best.pt`로 바꿔야 하고,
> 그러면 zero-shot 기준선을 포함해 모든 수치를 다시 내야 한다.

---

## 3. 두 트랙의 방법론적 차이 정리

| | 트랙 A (x500) | 트랙 B (EuRoC) |
|---|---|---|
| 학습 파라미터 | 전체 100 % | 4종 (18.1 / 51.5 / 100 / 100 %) |
| 출발점 | Blackbird 체크포인트 | Blackbird 체크포인트 + 무작위 대조 |
| 원 도메인 혼합 | 일부 런에서 혼합 | 없음 |
| 검증 분할 | 비행 통째 홀드아웃 | 시퀀스 꼬리 20 % + 12 s 간격 |
| 체크포인트 선택 | 창 RMSE (`mario.train.train`) | **롤아웃 ATE** |
| 좌표 정렬 | 불필요 (같은 규약) | **필수** (70.8° 차이) |
| 데이터 전처리 | 지상 제거 + 속도 평탄화 | 리샘플 + 바디 회전 |
| 시드 | 1 | 4 (42/1/2/3) |

트랙 A가 시드 1개인 것은 한계다. 이 프로젝트에서 단일 시드 결론은 세 번 뒤집혔으므로
(`PAPER_NOTES.md` §7), x500 수치를 인용할 때는 시드 반복이 없다는 점을 밝혀야 한다.

---

## 4. 산출물

```
scripts/
├── finetune_sitl.py         트랙 A
├── finetune_transfer.py     트랙 B
├── run_transfer_matrix.sh   트랙 B 매트릭스
├── reeval_transfer_full.py  전 런 전체 궤적 재채점
├── summarise_transfer.py    시드 평균
└── euroc_data.py            EuRoC 로더 + 바디 정렬

results/
├── dataset/         트랙 A 1차 수집 (걷기)   6비행 × 284.9 s
├── dataset_agg/     트랙 A 2차 수집 (스텝)   6비행 × 284.8 s  ← 채택
├── dataset_pat/     12비행 55.0분 (1차 체제)
├── dataset_yf/      13비행 57.5분 (1차 체제)
└── transfer/
    ├── cache/                     윈도우 캐시
    ├── logs/                      런별 로그
    └── full_rollout_rescore.json  전 런 재채점 결과

runs/
├── sitl_agg/            채택된 x500 모델 + finetune_meta.json
├── sitl_{finetune,only,balanced,pat,pat2,yf}/   대조군
└── transfer/euroc_*/    트랙 B 전 런
```

각 런 디렉터리의 `finetune_meta.json` / `run_meta.json`에 사용한 비행 목록·
하이퍼파라미터·윈도우 수·소요 시간이 기록되어 있다.
