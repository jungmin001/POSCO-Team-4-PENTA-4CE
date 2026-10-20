# 언리얼 연동 출력 스키마

`python Main.py all`(또는 `python Bundle.py`)을 실행하면 이 폴더에 파트별 결과가 모이고 `manifest.json`에 각 파일의 상태(`ok` / `missing`)가 기록된다. 언리얼은 `manifest.json`을 먼저 읽어 어떤 파일을 쓸 수 있는지 확인한다.

모든 파일은 UTF-8, 한글 포함. 설비·부품·비용 파일의 상세 항목은 `jungmin/outputs/unreal/UNREAL_PROMPT.md`에 있다. 이 문서는 파트 사이의 공통 약속과 파일 목록을 정리한다.

## 공통 약속

| 항목 | 값 |
|---|---|
| 설비 이름 `Equip` | `A` 소둔로, `B` 용접기, `C` 연주, `D` 조질압연, `E1`·`E2` 배풍기, `F` 교반기 |
| 설비 태그 `EqpTag` | 예: `P1-CC1-INS01` (C). `ontology.json`의 `Equipments`가 이름과 태그의 대응표 |
| 공정 순서 | `ontology.json`의 `ProcessOrder` (F → E1 → E2 → C → B → A → D) |
| 시나리오 `Scenario` | `S0001`~`S0360`. 파일 사이를 잇는 키 |
| 시각 | 모든 파일이 UTC ISO 형식. 위험·결함 파일은 `2027-05-15T00:00:00Z`, 진단 CSV는 샘플 간격이 짧아 밀리초까지 `2028-02-27T00:00:01.800Z` |
| 데이터 구분 | 모든 예측 파일이 **validation 시나리오 72개(S0217~S0288)** 로 같다. 시연에서 한 시나리오의 위험·진단·결함 위험을 함께 볼 수 있다. validation은 임계값·보정에 쓰인 데이터라 "처음 보는 데이터"로 소개하지 않는다 |

## 파일 목록

| 파일 | 만드는 곳 | 내용 | 키 |
|---|---|---|---|
| `manifest.json` | `Bundle.py` | 파일별 상태·크기·원본 수정 시각 | `File` |
| `ontology.json` | `jungmin/unreal.py` | 설비 7개, 부품, 공정 관계, 액션, 권한 | `Equipments[].Id` |
| `costs.json` | `jungmin/unreal.py` | 부품 가격·재고·납기, 정지·복구 비용 (모두 임의 가정값) | `Equipments[].Equip` |
| `risk_{A,B,C,D,E1,E2,F}.json` | `jungmin/unreal.py` | 설비별 6시간·24시간 고장 확률, 경보 단계, 이유, 정기 정비 일정 (설비마다 파일 하나, 형식 동일) | `Scenarios[].Scenario` + `Points[].Time` |
| `decisions_{설비}_check.csv` | `jungmin/unreal.py` | 기본 가정값 기준 선택지별 기대 비용 (언리얼 계산 검증용) | `Scenario` + `Time` |
| `anomaly_model.json` | `AnomalyModel.py` | C 이상탐지 모델 (정상 범위, 임계값, 경보 설정, 성능) | 해당 없음 (모델 정의) |
| `diagnosis_{A,B,C,D,E1,E2,F}.csv` | `Diagnosis.py --validation-dashboard` | 설비별 상태·원인 진단 예측 (validation, 정답 열 포함) | `SCENARIO_ID` + `MEAS_DT` |
| `flow_quality.csv` | `chanhee/Analysis.py` | 흐름(주편) 단위 최종(D) 결함 위험, C 검사 직후 예측 | `SCENARIO_ID` + `FLOW_ID` |
| `quality_summary.json` | `chanhee/Analysis.py` | 위 모델의 검증 성능·기준값 | 해당 없음 |
| `diagnosis_metrics.csv` | `Diagnosis.py` | 설비·과제별 검증 성능 | `equipment` + `task` |

`diagnosis_*.csv`는 모델 결과가 git에 없어서 `Diagnosis.py`로 모델을 만든 뒤 `Diagnosis.py --validation-dashboard`를 실행해야 생긴다(`Main.py all`은 둘 다 실행). 없으면 `manifest.json`에 `missing`으로 나온다.

## risk_{설비}.json (고장 위험)

```json
{
  "Version": 1, "Equip": "C", "EqpTag": "P1-CC1-INS01", "Split": "validation", "IntervalHours": 6,
  "Model": {
    "ModelVersion": "C-ENS-...", "Algorithm": "ExtraTrees + HistGradientBoosting (soft voting)",
    "Validation": { "6H": {"PR_AUC": 0, "Baseline_PR_AUC": 0, "Precision": 0, "Recall": 0, "Alarms": 0, "Points": 0},
                    "24H": { "...": "같은 항목" } }
  },
  "Thresholds": { "P6H": 0.50, "P24H": 0.56 },
  "WatchRatio": 0.5,
  "Scenarios": [ {
    "Scenario": "S0217", "StartAt": "2027-05-15T00:00:00Z",
    "PlannedMaintenance": [ {"StartTSec": 0, "EndTSec": 0, "Scope": "..."} ],
    "Points": [ {
      "Time": "2027-05-15T18:00:00Z", "TSec": 64800,
      "P6H": 0.01, "P24H": 0.18,            // 고장 확률
      "Alarm6H": false, "Alarm24H": false,  // 확률 >= Thresholds
      "Level": "normal",                    // normal | watch | alarm
      "Reasons": [ {"Feature": "w6_MLD-LVL_std", "Sensor": "MLD-LVL", "Stat": "std", "WindowHours": 6,
                    "Value": 0, "NormalLow": 0, "NormalHigh": 0, "Score": 0.3} ]
    } ]
  } ]
}
```

- `Stat`이 `std_change`처럼 `_change`로 끝나면 6시간 창과 12시간 창의 차이(최근 변화량)이고 `WindowHours`는 12다.
- `Level`: 6시간 또는 24시간 경보면 `alarm`, 24시간 임계값의 `WatchRatio`배 이상이면 `watch`, 아니면 `normal`.
- 신뢰도 주의: 6시간 예측은 설비마다 고장 사례가 매우 적어(학습 15~52건, validation 5~16건) 성능을 믿기 어렵다. 화면에서는 "점검 권고" 수준으로 보여 주고 확정 고장으로 표시하지 않는다. 24시간 예측이 더 안정적이다.
- 설비별 validation 성능은 각 파일의 `Model.Validation`에 들어 있다. 아래는 요약이다.

| 설비 | 24시간 PR-AUC (기준선) | 24시간 정밀도 / 재현율 | 6시간 PR-AUC (기준선) | 6시간 validation 고장 사례 |
|---|---|---|---|---|
| A | 0.69 (0.03) | 0.72 / 0.59 | 0.35 (0.01) | 5 |
| B | 0.84 (0.04) | 0.56 / 0.83 | 0.30 (0.01) | 5 |
| C | 0.77 (0.11) | 0.78 / 0.77 | 0.58 (0.02) | 16 |
| D | 0.79 (0.04) | 0.77 / 0.80 | 0.69 (0.01) | 5 |
| E1 | 0.90 (0.04) | 0.75 / 0.78 | 0.66 (0.01) | 5 |
| E2 | 0.85 (0.03) | 0.92 / 0.57 | 0.64 (0.01) | 5 |
| F | 0.87 (0.04) | 0.88 / 0.75 | 0.35 (0.01) | 6 |

## diagnosis_{설비}.csv (상태·원인 진단)

| 열 | 뜻 |
|---|---|
| `SCENARIO_ID`, `EQP_CD`, `MEAS_DT`, `equipment` | 시나리오, 설비 태그, 관측 시각, 설비 이름(A~F) |
| `state_prediction` | `normal` / `degraded` / `failed` / `maintenance` |
| `state_probability` | 상태 예측 확률 (0~1) |
| `needs_review`, `uncertainty_reasons` | 참이면 확정 대신 "추가 점검 필요"로 표시, 이유는 글자 |
| `cause_candidate`, `cause_candidate_2` | 원인 후보 1순위·2순위. 상태가 `degraded`/`failed`일 때만 채워짐 |
| `cause_probability_conditional` | 이상 상태라는 조건에서의 후보 점수. 실제 발생 확률이 아님 |
| `cause_needs_review`, `cause_uncertainty_reasons` | 참이면 원인은 확정하지 않고 후보로만 표시 |
| `actual_state`, `actual_cause` | validation 정답 (`부품::고장모드`). 시연에서 예측과 나란히 보여 주는 용도이며 운영 예측에는 없다 |

- A·C·F 설비의 원인은 검증 성능이 낮아 모두 `cause_needs_review`가 참이다. 후보로만 보여 준다.
- 상태 예측에서 `failed` 클래스는 학습·검증 사례가 거의 없어 신뢰할 수 없다.

## anomaly_model.json (C 이상탐지)

모델 정의 파일이다. 주요 항목은 `feature_columns`(입력 센서 특징), `normal_range`(센서별 `median`·`scale`), `operating_point`(선택된 임계값 백분위, 지속 시간, 검증 재현율·오경보), `alarm_config`, `quality`, `training_summary`. 새 데이터 예측은 `AnomalyModel.py`의 `RunMode = "predict"`로 한다.

## 언리얼 → Python (결정 기록)

언리얼이 `decision_log.json`을 저장하면 `python jungmin/unreal.py feedback=경로/decision_log.json`으로 실제 결과와 비교한다.

```json
{ "Decisions": [ { "Scenario": "S0217", "Equip": "C", "Time": "2027-05-15T18:00:00Z",
                   "Chosen": "replace_6h", "Recommended": "wait" } ] }
```

`Chosen`이 `replace`로 시작하면 교체, 아니면 대기로 본다. 결과 판정은 `예방` / `불필요 정비 가능성` / `놓침` / `정상 대기` / `판정 불가`이고 `jungmin/outputs/unreal/feedback.csv`에 저장된다. 결정 기록에 여러 설비가 섞여 있어도 한 파일(`feedback.csv`)에 모인다.

## flow_quality.csv (최종 결함 위험, 흐름 단위)

설비 고장이 아니라 **생산 흐름(주편) 하나가 마지막 D 공정에서 결함이 될 위험**이다. 설비 파일과는 키가 다르다(`SCENARIO_ID` + `FLOW_ID`).

| 열 | 뜻 |
|---|---|
| `SCENARIO_ID`, `FLOW_ID` | 시나리오, 생산 흐름 (예: `S0217-FLOW01`) |
| `PREDICTED_AT` | 예측 시점 = 그 흐름의 C 검사 직후 (`2027-05-15T06:00:00Z` 형식) |
| `defect_score` | 최종 결함 위험 점수 (0~1) |
| `inspection_requested` | `defect_score >= threshold`이면 참. 화면에는 "결함 검사 요망"으로 표시 |
| `threshold` | 경보 기준 (0.1311) |
| `origin_eqp` | 정답 기준 결함 원인 설비 (validation 평가용) |
| `actual_defect` | 실제 최종 결함 여부 0/1 (validation 정답, 평가·피드백용) |

- 출력은 "불량 확정"이 아니라 "결함 검사 요망"으로 표시한다. validation에서 결함 308개 중 299개를 찾았고(재현율 97.1%) 오경보가 96개(정밀도 75.7%)다.
- `origin_eqp`, `actual_defect`는 validation 정답이라 새 데이터 예측에는 없다. 화면에는 쓰지 않고 성능 확인에만 쓴다.
- 이 파일은 validation 시나리오 72개만 담는다. 모델은 `Results/Quality/quality_model.joblib`에 저장된다.
