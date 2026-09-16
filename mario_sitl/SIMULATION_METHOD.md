# 시뮬레이션·폐루프 방법 — 재현용 상세 기록

MARIO를 PX4 SITL의 실제 비행 제어 루프에 물리기까지의 절차. `FINETUNE_METHOD.md`의
짝이며, 결과 해석은 `APPLICATION_VALIDATION.md`와 `REPORT.md`에 있다.

여기서 다루는 것은 넷이다 — 시뮬레이션 스택 구성, 좌표계 변환, 실시간 추론 파이프라인,
그리고 폐루프 미션 설계.

---

## 1. 시뮬레이션 스택

| 구성 | 버전 / 설정 |
|---|---|
| PX4-Autopilot | **v1.15.4** (`99c4040`), `px4_sitl_default` |
| 시뮬레이터 | Gazebo (gz sim), 기체 **x500** 쿼드로터 |
| 미들웨어 | ROS 2 + `uxrce_dds_client` (PX4 ↔ ROS 브리지) |
| GPU | NVIDIA RTX A6000 |

### 1.1 인터프리터가 둘로 나뉜 이유

ROS 2 스택은 py3.11 RoboStack 환경인데 `mamba-ssm`은 **cp310 휠**이고, 이 호스트의
GLIBC는 2.31로 사전 빌드된 cp311 휠이 요구하는 2.32보다 낮다. `rclpy`와 `mamba_ssm`을
한 인터프리터에 넣을 수 없다.

그래서 **추론을 별도 프로세스로 분리**하고 Unix 도메인 소켓으로 연결했다:

```
[ROS 2 노드 · py3.11]  ──UDS(SOCK_SEQPACKET)──>  [추론 서버 · py3.10 mamba env]
   numpy만 사용                                       torch + pypose + mamba_ssm
```

**이 소켓 왕복은 종단 지연에 포함해 측정한다** — 숨긴 비용이 아니다(§3.3).

### 1.2 환경 게이트

`scripts/check_env.py`가 학습 환경(torch, mamba_ssm, 체크포인트 로드, 출력 형상)과
SITL 스택(ROS 2, PX4, Gazebo)을 각각 확인한다. Stage 0의 통과 조건이다.

---

## 2. 좌표계 — 네 가지 규약을 잇는다

`mario_ros/mario_frames.py`. **Stage 3에서 왕복 오차 2.4e-07 m로 검증**했다
(`scripts/verify_frames.py`, 4개 변환 전부 통과).

| 시스템 | world | body |
|---|---|---|
| PX4 | **NED** | **FRD** (Front-Right-Down), 쿼터니언은 Hamilton (w,x,y,z), FRD→NED |
| MARIO | **NWU** | x=Right, y=Back, z=Down |

변환 행렬 두 개로 전부 처리된다:

```python
C_W = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]   # NED -> NWU
C_B = [[0, 1, 0], [-1, 0, 0], [0, 0,  1]]   # FRD -> B_MARIO
```

적용:

```python
acc_m  = C_B @ acc_frd
gyro_m = C_B @ gyro_frd
R_m    = C_W @ R_ned_frd @ C_B.T            # B_MARIO -> NWU
```

자세는 SO(3) Log로 넣는다(`pypose`). 출력 변위는 역방향으로 되돌린다:

```python
disp_ned = R_ned_frd[idx] @ C_B.T @ disp_body
```

### 2.1 타임스탬프 정리 — 두 가지 실제 문제

`make_monotonic()`이 처리한다.

**(a) 중복·역순 타임스탬프.** best-effort QoS이고 FMU가 리셋 시 재발행한다. `Slerp`와
`interp1d`는 이를 거부한다 — 1차 Stage 2 실행에서 약 2000회 중 **297회**가
`Times must be in strictly increasing order`로 죽었다.

**(b) 시계가 둘 섞인다.** `uxrce_dds_client`가 에이전트 시각 오프셋을 적용하기 전까지
PX4는 원시 hrt 시각으로 찍는다. 그래서 기록 앞부분 4~10개 샘플이 부팅 상대 시각(~1e3 s)을,
나머지가 epoch 오프셋 시각(~1.8e9 s)을 갖는다. 한 비행은 이 때문에 **1.79e9 초 격자**를
요구해 1.3 TiB 할당을 시도했다.

→ 중앙값에서 **3600 초** 넘게 떨어진 샘플을 버리고, 안정 정렬 후 중복을 제거한다.

### 2.2 리샘플

학습 파이프라인(`mario.data.load_blackbird`)과 동일하게 맞춘다.

- IMU 채널: `interp1d` 선형 보간
- 자세: **`Slerp`** — 쿼터니언 성분의 선형 보간은 등가가 아니며 사양에서 금지
- 격자: 100 Hz, 두 센서가 **실제로 덮는 가장 최근 시각**에서 뒤로 1000 샘플

```python
t_end = min(imu_t[-1], att_t[-1])          # 절대 앞으로 외삽하지 않는다
grid  = t_end - (1000 - 1 - arange(1000)) * 0.01
```

`Slerp`는 외삽이 불가능하므로 격자를 자세의 시간 범위로 클램프한다.

---

## 3. 실시간 추론 파이프라인

### 3.1 어느 출력을 쓰는가 — `OUTPUT_INDEX = 109`

네트워크는 1000 샘플 윈도우에 대해 112개 스텝을 내지만, `out[i]`의 수용 영역은 입력
`9i+12`에서 끝난다. 따라서 `out[110]`과 `out[111]`은 **윈도우 가장자리의 제로 패딩에
잘린다** — 실측하면 둘 다 수용 영역이 자연값 1002/1011이 아니라 정확히 999에서 끝난다.

`out[109]`가 패딩 없는 가장 최신 출력이며, 입력 샘플 995–1004 구간의 변위를 예측한다.

### 3.2 변위가 아니라 속도를 낸다

한 스텝이 `LABEL_STRIDE = 9` 샘플을 덮으므로 `dt = 9 × 0.01 = 0.09 s`다.

```python
vel_ned = disp_to_ned(disp_body, R_ned_frd[idx]) / 0.09
t_ref   = grid[idx]                      # idx = 14 + 109*9
```

여기서 적분하지 않는다. 적분은 EKF2가 한다.

**`net.eval()`은 필수다.** BatchNorm이 train 모드면 `out[0]`이 윈도우 전체에 의존하게
되어 인과성이 깨진다. eval 모드라야 causal이다.

### 3.3 지연 측정 프로토콜

**웜업**: 첫 CUDA 실행은 커널 컴파일과 할당자 초기화 비용을 치르므로, 서버 기동 시
더미 텐서로 5회 예열한다. 그러지 않으면 그 비용이 노드가 보고하는 p95에 들어간다.

**측정 구간**: ROS 노드가 요청을 만들기 시작한 시점부터 응답을 받은 시점까지 —
직렬화·소켓 왕복·리샘플·GPU 추론·역직렬화 전부 포함.

**실측 4회 (전부 목표 25 Hz)**

| 실행 | 추론 수 | 달성 rate | mean | p50 | p95 | 실패 |
|---|---:|---:|---:|---:|---:|---:|
| `ev_validity_probe` | 870 | 24.97 Hz | 34.10 | 34.07 | 37.84 | 0 |
| `stage2_shadow` | 1,996 | 25.00 Hz | 34.34 | 34.63 | 38.49 | 0 |
| `stage5_shadow` | 4,746 | 25.00 Hz | 33.95 | 34.03 | **37.90** | 0 |
| `stage5b_shadow` | 4,751 | 25.00 Hz | 33.88 | 34.01 | 38.03 | 0 |

단위 ms. 네 실행이 37.8–38.5 ms 안에서 일치한다. `REPORT.md`가 인용하는 37.9 ms는
`stage5_shadow`(폐루프 실행) 값이다.

**내역**: 서버 측 22.08 ms, 소켓 왕복 22.24 ms, 종단 34.34 ms. 순수 GPU 추론은
별도 벤치에서 **1.776 ms**(p99 1.853)이므로, 종단 지연의 **95 % 이상이 모델 밖**이다
(ROS·DDS·직렬화·리샘플).

### 3.4 EKF2로 발행

`mario_ros/mario_odometry_node.py`가 25 Hz 타이머로 추론하고
`/fmu/in/vehicle_visual_odometry`에 `VehicleOdometry`를 발행한다.

```python
m.pose_frame     = POSE_FRAME_NED
m.velocity_frame = VELOCITY_FRAME_NED
m.velocity       = vel_ned
m.velocity_variance = [max(c, 1e-6) / (0.09 ** 2) for c in cov]
```

QoS는 PX4 규약대로 `BEST_EFFORT` + `TRANSIENT_LOCAL` + `KEEP_LAST`.

> **주의**: `cov` 헤드는 입력과 무관한 상수를 낸다(변동계수 0.000–0.001, ~0.90 m/s
> 고정). 원인은 `uncertainty_weight=1e-4`가 `loss_weight=1000.0`에 맞서 NLL 항의 기울기
> 기여가 사실상 0인 것이다. 따라서 **EKF2는 학습된 불확실성의 외양을 쓴 고정 측정
> 잡음을 받아 왔다.** 상수도 유효한 잡음 모델이므로 치명적이지는 않으나, "원리적
> 가중"은 작동하지 않았다. `uncertainty_weight`를 0.01로 올리면 헤드가 살아난다
> (변동폭 55.9 %, E[r²/cov] 0.024 → 0.317).

**섀도 모드**가 기본이다(`--no-publish`). 발행 없이 추론만 돌려 지연·정확도를 먼저
검증하고, 통과한 뒤에 EKF2에 물린다.

---

## 4. EKF2 설정

`scripts/set_stage5_params.sh`. **모든 값은 이 v1.15 트리의
`src/modules/ekf2/module.yaml`에서 확인했고 문서나 기억에서 쓰지 않았다**(작업 규칙 4).

| 파라미터 | 값 | 의미 |
|---|---:|---|
| `EKF2_GPS_CTRL` | **0** | 비트마스크 0–15 (bit0 위경도, bit1 고도, bit2 3D 속도, bit3 듀얼안테나 헤딩). 0 = GNSS 보조 완전 차단 |
| `EKF2_EV_CTRL` | **4** | 비트마스크 (bit0 수평위치, bit1 수직위치, bit2 3D 속도, bit3 요). 4 = bit2만 → **속도만 융합** |
| `EKF2_HGT_REF` | **0** | enum (0 기압, 1 GPS, 2 레인지, 3 비전). 기압계로 두어 수직 추정 오차가 상승·하강을 유발하지 못하게 함 |
| `EKF2_BARO_CTRL` | 1 | 기압 고도 보조 유지 (기본값) |
| `EKF2_EV_DELAY` | **35** ms | Stage 2 실측 p50 34.6 ms. p95가 아니라 p50 — EKF2는 전형적 지연을 원하고, p95 꼬리는 1996개 중 4개뿐 |

**저장이 필요하다.** `px4-param`은 방금 설정한 값을 미저장(`*`)으로 표시하고, SITL은
정상 종료 시에만 `rootfs/parameters.bson`에 flush한다. 재시작 절차가 `pkill`을 쓰므로
`px4-param save`를 하지 않으면 `EKF2_HGT_REF`·`EKF2_EV_DELAY`(둘 다 reboot_required)가
조용히 사라진다.

`bash scripts/set_stage5_params.sh restore`로 기본값 복원 — 이후 GPS 기준선 실행이
이 설정에 오염되지 않게 한다.

---

## 5. 미션 설계

### 5.1 공통 offboard 규약

`mario_ros/offboard_mission_node.py`

| 항목 | 값 | 근거 |
|---|---:|---|
| 셋포인트 발행 | **50 Hz** | 사양: 최소 50 Hz, 2 Hz 미만이면 페일세이프 |
| 무장 전 선발행 | **20 tick** (0.4 s) | 사양: 10회 이상 |
| 원점 래치 조건 | `xy_valid` + `z_valid` **3 초 유지** | 추정기가 안정된 뒤에 좌표를 고정 |
| 도착 판정 | 반경 **0.3 m** 안에 **2 초** 체류 | 사양 Stage 1 성공 기준 |

### 5.2 Stage 1 — GPS 기준선

이륙 3 m → 북쪽 5 m → 30 초 호버 → 착륙. **모든 폐루프 수치의 잣대**이며 이것 없이는
MARIO 수치가 무의미하다.

결과: 도착 오차 **0.126 m**, 정착 중 이탈 0.59 m.

### 5.3 Stage 5 — MARIO 단독 폐루프

동일 미션을 `EKF2_GPS_CTRL=0`, `EKF2_EV_CTRL=4`로 실행. GPS는 완전히 꺼져 있고
EKF2는 MARIO 속도만 받는다.

**Gazebo GT는 검증용으로만 기록하고 제어에는 절대 쓰지 않는다**(사양 §1).

### 5.4 Stage 6 — 한계 스윕

`scripts/run_sweep.py`. 4×3×4 완전 격자(48회 폐루프 비행)는 수 시간이 걸리고 축이
서로 간섭할 이유가 없으므로 **한 번에 한 축씩(one factor at a time)** 10회로 줄였다.

```python
GRID = ( [{"north": d,  "hover": 10,  "speed": 0.0}  for d in (2, 5, 10, 20)]     # 거리
       + [{"north": 5,  "hover": h,   "speed": 0.0}  for h in (30, 60, 120)]      # 호버
       + [{"north": 10, "hover": 10,  "speed": v}    for v in (0.5, 1.0, 2.0)] )  # 순항
```

**매 실행마다 PX4와 Gazebo를 처음부터 재시작한다.** 이탈한 채(또는 옆으로 누운 채)
끝난 실행이 다음 실행을 오염시키기 때문이다. 종료는 pgrep으로 PID를 찾아 SIGKILL —
`pkill` 패턴은 이 스크립트 자신의 셸까지 잡을 수 있다.

### 5.5 연속 이동 미션

`mario_ros/continuous_mission_node.py`. Stage 5가 제자리 유지에서 실패한 뒤,
**드리프트가 시간 비례에서 거리 비례로 바뀌는지**를 직접 확인하려고 만들었다.

| 인자 | 기본값 |
|---|---:|
| 고도 | 6.0 m |
| 선회 반경 | 8.0 m |
| 선회 속도 | 3.5 m/s |
| GPS 구간 | 1 랩 (기준 랩) |
| MARIO 구간 | **3 랩** |

기준 랩을 GPS로 돈 뒤 GPS를 끊고 MARIO만으로 3랩을 더 돈다. 같은 궤적을 같은 속도로
도므로 두 구간이 직접 비교된다.

---

## 6. 데이터 기록

`mario_ros/record_flight_node.py`가 원시 스트림을 그대로 받아 `.npz`로 저장한다.

| 키 | 내용 |
|---|---|
| `imu` | `SensorCombined` — 자이로·가속도 + 타임스탬프 |
| `att_gt` / `att_ekf` | 자세 (Gazebo GT / EKF2) |
| `pos_gt` / `pos_ekf` | 위치 (Gazebo GT / EKF2) |

GT를 함께 기록하되 **검증에만** 쓴다.

`varied_flight_node.py`가 학습 데이터 수집용 비행을 만든다(`--aggressive`로 계단식
셋포인트 — `FINETUNE_METHOD.md` §1.1의 2차 수집이 이것이다).

### 6.1 실기 이전 시 주의 (SITL에서 확인된 것)

- **GT는 GPS 도플러 속도만 쓸 것.** 위치를 미분하면 90 ms 라벨에 11–33 m/s 잡음이
  실려 학습이 불가능하다. `SensorGps`의 `vel_n/e/d_m_s`와 `s_variance_m_s`를 쓰면
  0.05–0.1 m/s로 라벨의 3 % 수준이다.
- **`vehicle_local_position`을 GT로 쓰지 말 것.** EKF2 출력이라 IMU가 이미 섞여 있어
  순환 참조가 된다.
- **자세 GT는 EKF2로 충분하다.** ATE 차이 0.08 m (5.193 vs 5.273).
- **MARIO는 FC에서 못 돈다.** Pixhawk는 마이크로컨트롤러다. 보조 컴퓨터(Jetson 권장)가
  필요하고, 지연을 재측정해 `EKF2_EV_DELAY`에 반영해야 한다 — 현재 35 ms는 데스크톱
  GPU 기준이다.

---

## 7. 실행 절차

```bash
# 0. 환경 게이트
$PY mario_sitl/scripts/check_env.py

# 1. 좌표계 검증 (4개 변환)
$PY mario_sitl/scripts/verify_frames.py

# 2. 추론 서버 기동 (mamba env, 별도 터미널)
$PY mario_sitl/mario_ros/mario_infer_server.py \
      --ckpt runs/nm_s42/best.pt --device cuda --socket /tmp/mario_infer.sock

# 3. PX4 + Gazebo 기동, 그 다음 EKF2 전환
bash mario_sitl/scripts/set_stage5_params.sh      # 이후 PX4 재시작 필요

# 4. 섀도 모드로 지연·정확도 먼저 검증 (발행 없음)
ros2 run ... mario_odometry_node --rate 25 --duration 180 --no-publish \
      --out mario_sitl/results/stage2_shadow.json

# 5. 통과하면 EKF2에 발행하며 폐루프 미션
ros2 run ... mario_odometry_node --rate 25 --publish &
ros2 run ... offboard_mission_node --north 5 --hover 30 --alt 3

# 6. 연속 이동
ros2 run ... continuous_mission_node --radius 8 --speed 3.5 --alt 6 \
      --laps-gps 1 --laps-mario 3

# 7. 한계 스윕 (10회, 매번 PX4/Gazebo 재시작)
$PY mario_sitl/scripts/run_sweep.py --ckpt runs/nm_s42/best.pt
```

> **체크포인트 주의**: 추론 서버의 기본값은 `runs/nm_s42/best.pt`(브랜치 `no-motor`)로
> 바뀌었으나 **이 체크포인트로 폐루프를 돌린 적이 없다.** §3–5의 모든 실측은 모터
> 시대의 `trial8_100ep`·`sitl_agg` 계열로 얻은 것이다. 배포 전에 NM으로 최소한 Stage 2
> 섀도와 Stage 5를 다시 돌려야 한다.

---

## 8. 알려진 함정 (전부 실제로 겪은 것)

| 증상 | 원인 | 대응 |
|---|---|---|
| `Times must be in strictly increasing order`, 2000회 중 297회 실패 | best-effort QoS 중복 + FMU 재발행 | `make_monotonic` |
| 1.3 TiB 할당 시도 | `uxrce_dds_client` 동기화 전 원시 hrt 타임스탬프 혼입 | 중앙값 ±3600 s 밖 제거 |
| 재시작 후 EKF2 파라미터 소실 | SITL이 정상 종료 시에만 flush, 절차는 `pkill` 사용 | `px4-param save` |
| 스윕 1번 케이스에서 정지 | `capture_output=True`가 백그라운드 PX4에 파이프 상속 → `communicate()` 블록 | 출력을 `/dev/null`로 |
| 스윕 스크립트 자신이 종료됨 | `pkill` 패턴이 자기 셸까지 매치 | `pgrep`으로 PID 찾아 SIGKILL |
| Gazebo ↔ PX4 y축 부호 반전 | Gazebo 모델 프레임은 x전방/y**좌**/z상, PX4 할당기는 FRD로 y**우** | SDF의 y가 `CA_ROTORn_PY`와 부호 반대 |
| BatchNorm이 인과성을 깸 | train 모드에서 `out[0]`이 윈도우 전체에 의존 | `net.eval()` |
| 첫 추론이 p95를 오염 | CUDA 커널 컴파일 + 할당자 초기화 | 더미 5회 웜업 |

---

## 9. 산출물

```
mario_ros/
├── mario_frames.py             §2 좌표 변환 + 윈도우 구성
├── mario_infer_server.py       §3 UDS 추론 서버
├── mario_odometry_node.py      §3.4 섀도/발행 노드
├── offboard_mission_node.py    §5.1–5.3 미션
├── continuous_mission_node.py  §5.5 연속 이동
├── varied_flight_node.py       §6 학습 데이터 수집
└── record_flight_node.py       §6 원시 스트림 기록

scripts/
├── check_env.py                §1.2 환경 게이트
├── verify_frames.py            §2 좌표계 검증
├── set_stage5_params.sh        §4 EKF2 전환/복원
├── run_sweep.py                §5.4 한계 스윕
├── analyze_sweep.py            §5.4 사후 상관 분석
├── analyze_closedloop.py       chase_ratio 판별
├── analyze_uncertainty.py      cov 헤드 검증
├── eval_openloop.py            세그먼트 ATE/TDE
├── diagnose_observability.py   관측성 + 방향 정확도
├── plot_latency.py             §3.3 지연 히스토그램
└── make_airframe.py            커스텀 기체 생성기

results/
├── stage1_baseline{,_analysis}.json   §5.2 GPS 기준선
├── stage2_shadow.json                 §3.3 섀도 지연
├── stage5_shadow.json / stage5_flight.npz     §5.3 폐루프
├── stage5b_shadow.json / stage5b_flight.npz   §5.3 재실행
├── stage5_divergence.json             §5.3 발산 판별
├── stage6_sweep{,_analysis}.json      §5.4 스윕 + 상관
├── stage6_uncertainty.json            §3.4 cov 헤드
├── continuous_run{,_yf}.json          §5.5 연속 이동
├── ev_validity_probe.json             EKF2 유효성 게이트
└── figures/                           지연·궤적·발산 그림
```
