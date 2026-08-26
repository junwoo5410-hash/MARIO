# MARIO — Causal Mamba Displacement Estimation

Blackbird 드론 데이터셋에서 IMU · 자세 · 모터 추력을 입력받아 **바디 프레임 변위(displacement)** 를 예측하는 Causal Mamba 네트워크입니다.
예측된 변위를 적분해 궤적을 복원하고 ATE / TDE로 평가합니다.

`notebooks/Causal_Mamba_Disp_Trial8.ipynb` 노트북을 재현 가능한 패키지 형태로 정리한 것입니다.

---

## 구조

```
.
├── configs/
│   └── trial8.yaml           # Optuna Trial 8 하이퍼파라미터
├── mario/
│   ├── config.py             # 데이터클래스 기반 설정 (+ YAML 로딩)
│   ├── data.py               # Blackbird 로딩 / 시간 정렬 / 리샘플링
│   ├── dataset.py            # 슬라이딩 윈도우 + 바디 프레임 변위 라벨
│   ├── model.py              # CNNEncoder, CausalMambaBlock, CausalMambaDispNet
│   ├── losses.py             # Huber + 불확실성(NLL) 손실
│   ├── train.py              # 학습 루프, 체크포인팅
│   ├── evaluate.py           # 궤적 롤아웃, ATE / TDE
│   ├── visualize.py          # 3D 궤적 PNG / GIF
│   └── utils.py              # 시드, 디바이스 헬퍼
├── notebooks/
│   └── Causal_Mamba_Disp_Trial8.ipynb   # 원본 실험 노트북
├── scripts/
│   ├── train.py              # 학습 진입점
│   ├── evaluate.py           # 평가 진입점
│   └── animate.py            # 궤적 애니메이션 진입점
├── requirements.txt
└── pyproject.toml
```

---

## 설치

`mamba-ssm`은 CUDA 툴체인이 필요하므로 **PyTorch를 먼저 설치**해야 합니다.

```bash
python -m venv .venv && source .venv/bin/activate

# 1) 환경에 맞는 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2) 나머지
pip install -r requirements.txt

# 3) (선택) 패키지로 설치
pip install -e .
```

## 데이터

[Blackbird 데이터셋](https://github.com/mit-fast/Blackbird-Dataset)을 아래 구조로 배치합니다.
기본 경로는 `~/blackbird_data`이며 `configs/trial8.yaml`의 `data.data_dir` 또는 `--data-dir`로 변경할 수 있습니다.

```
blackbird_data/
├── train/<trajectory>/yawForward/<maxSpeed>/{imu_data.csv, groundTruthPoses.csv, thrust_data.csv}
├── test/ ...
└── eval/ ...
```

`thrust_data.csv`는 선택 사항입니다. 없으면 모터 채널이 0으로 채워져 IMU만으로 동작합니다.

---

## 실행

```bash
# 학습 (100 epoch)
python scripts/train.py --config configs/trial8.yaml

# 평가 — SEEN / UNSEEN ATE, TDE
python scripts/evaluate.py --config configs/trial8.yaml

# 궤적 GIF (UNSEEN 1번 궤적)
python scripts/animate.py --split unseen --index 1
```

자주 쓰는 옵션:

| 옵션 | 설명 |
|---|---|
| `--data-dir PATH` | 데이터 루트 재정의 |
| `--output-dir PATH` | 결과 저장 위치 재정의 |
| `--epochs N` | 에폭 수 재정의 |
| `--device cuda\|cpu\|auto` | 디바이스 지정 |
| `--checkpoint PATH` | 평가/시각화에 쓸 가중치 지정 |
| `--all` (animate) | 스플릿 전체 궤적 렌더링 |

출력물은 `output_dir` 아래에 생성됩니다.

```
best.pt                 # 테스트 RMSE 기준 최고 가중치
config.yaml             # 실제로 사용된 설정 스냅샷
training_history.json   # 에폭별 loss / RMSE / lr
results.json            # 궤적별 + 평균 ATE, TDE
figures/                # PNG, GIF
```

---

## 방법

**입력** — 100 Hz로 리샘플링한 1000 샘플(10초) 윈도우: 가속도(3) · 자이로(3) · `SO3.Log()` 자세(3) · 정규화된 모터 추력(3).

**모델** — 세 개의 1D CNN 인코더가 각 입력을 9배 다운샘플링(stride 3 × 2)한 뒤 concat → 64차원 투영 → residual Causal Mamba 블록 → 변위 헤드와 공분산 헤드. 공분산은 `exp(x − 5)`로 양수를 보장합니다.

**라벨** — 9 샘플(0.09초)마다의 월드 프레임 변위를 시점 `t`의 GT 자세로 바디 프레임에 회전시킨 값. 윈도우당 111개.

**손실** — `1000 × Huber(residual, δ=0.002) + 1e-4 × NLL(residual.detach(), cov)`.
공분산 헤드는 detach된 잔차로만 학습되어 변위 헤드에 그래디언트를 흘리지 않습니다.

**평가** — 겹치지 않는 윈도우로 변위를 예측하고, GT 자세로 월드 프레임에 되돌린 뒤 첫 GT 위치부터 누적 적분합니다. ATE는 위치 오차의 RMSE, TDE는 이동 거리 대비 백분율입니다.

---

## 참고 성능 (평균 ATE [m])

| 설정 | SEEN | UNSEEN |
|---|---:|---:|
| 속도 예측 (Optuna) | 0.523 | 2.263 |
| 변위 예측 baseline | 0.429 | 1.809 |
| 변위 Trial 8 (30 epoch) | 0.429 | **1.703** |
| AirIO + Motor (양방향, 참고) | 0.460 | 0.767 |

100 epoch 결과는 `scripts/evaluate.py`가 위 표와 함께 출력합니다.
`AirIO + Motor`는 양방향 모델이라 인과(causal) 조건이 아니며 상한 참고용입니다.

---

## Trial 8 하이퍼파라미터

| | |
|---|---|
| `d_state` / `d_conv` / `expand` / `num_layers` | 32 / 4 / 1 / 1 |
| learning rate | 2.576e-4 |
| batch size | 128 |
| weight decay | 1e-3 |
| loss weight / huber delta | 1000 / 0.002 |
| scheduler | `ReduceLROnPlateau` (factor 0.2, patience 5, min 1e-5) |
| grad clip | 1.0 |
