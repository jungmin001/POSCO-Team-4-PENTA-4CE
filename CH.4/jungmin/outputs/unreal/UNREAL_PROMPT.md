# 언리얼 MCP 작업 프롬프트 — 철강 설비 정비 의사결정 솔루션 (디지털 트윈)

아래 내용을 언리얼 MCP에 연결된 Claude에게 그대로 전달하세요.

---

## 목표

철강 공장 디지털 트윈에 **정비 의사결정 솔루션**을 만든다. 핵심은 예측 결과를 화면에 띄우는 데서 끝내지 않고,
**예측 → 비교(what-if) → 결정 → 기록 → 실제 결과와 비교**가 한 화면에서 이어지게 하는 것이다.

```
Python: 데이터 → 모델 예측 (risk_{설비}.json)
언리얼: 위험 확인 → 가정값을 바꿔 선택지 비교 → 정비 결정 실행 → 결정 기록 (decision_log.json)
Python: 결정 기록 ↔ 실제 결과 비교 (feedback) → 모델·가정값 개선 → 다음 예측
```

구조는 **온톨로지** 방식으로 만든다. 화면의 모든 정보는 "어느 설비의 어느 부품"에 연결되어 있어야 한다.

| 구성 | 뜻 | 이 프로젝트 |
|---|---|---|
| 객체 | 업무 대상 | 설비 7개(A, B, C, D, E1, E2, F), 부품 16개, 시점별 예측, 정비 결정 |
| 속성 | 대상의 정보 | 설비 이름·센서·공정 순서, 부품 가격·재고·납기, 고장 확률 |
| 관계 | 대상 사이 연결 | 부품 → 설비 장착(InstalledOn), 설비 → 다음 공정 설비(FeedsInto) |
| 액션 | 상태를 바꾸는 작업 | 가정값 수정, 교체 예약, 대기, 현장 메모, 기본값 복원 (권한별) |

## 입력 파일 (Python이 만듦, 읽기 전용)

원본 위치: `POSCO-Team-4-PENTA-4CE/CH.4/jungmin/outputs/unreal/` → 언리얼 프로젝트 `Content/DSSData/`로 복사.
패키징 시 Project Settings → Packaging → "Additional Non-Asset Directories to Copy"에 `DSSData` 추가.
모든 파일은 UTF-8(BOM 없음), 한글 포함. 시각은 `"2027-05-15T00:00:00Z"` 형식 문자열, `TSec`는 시나리오 시작부터 초.

| 파일 | 내용 |
|---|---|
| `ontology.json` | 객체·관계·액션·권한·데이터 출처 |
| `costs.json` | 가정값 **기본값** (부품 가격·재고·납기, 설비별 정지·복구 비용). 모두 임의값 |
| `risk_A.json` ~ `risk_F.json` | 설비 7개 각각의 모델 정보(버전·성능·출처) + validation 시나리오 72개 × 6시간 간격 13시점의 예측·이유·정기 정비 일정 (파일마다 형식 동일) |
| `decisions_{설비}_check.csv` | 기본 가정값으로 Python이 계산한 정답 비용표 (언리얼 계산 검증용, 불러오지 않아도 됨) |

### ontology.json

```json
{
  "Version": 1, "GeneratedAt": "...",
  "ObjectTypes": { "Equipment": "설명", "Part": "설명", "RiskPoint": "설명", "Decision": "설명" },
  "Equipments": [
    { "Id": "C", "EqpTag": "P1-CC1-INS01", "EqpName": "연주 주편", "Plant": "CC1",
      "Sensors": ["LOAD","WD-SPD","WD-LOD","OSC-FRQ","OSC-STK","MLD-LVL","CONTROL_DELAY"],
      "RiskFile": "risk_C.json",   // 설비마다 risk_{Id}.json,          // 빈 문자열이면 아직 예측 없음
      "Stage": 3,                         // 공정 순서 (0부터)
      "Downstream": ["B","A","D"] }       // 이 설비가 멈추면 영향을 받는 다음 공정 설비
  ],
  "Parts": [ { "Id": "C.valve", "Equip": "C", "Part": "valve", "PartName": "valve" } ],
  "Relations": [
    { "Type": "InstalledOn", "From": "C.valve", "To": "C" },
    { "Type": "FeedsInto", "From": "C", "To": "B", "AvgWaitHours": 0.21 }
  ],
  "ProcessOrder": ["F","E1","E2","C","B","A","D"],
  "Actions": [ { "Action": "ScheduleReplacement", "Label": "부품 교체 예약", "Roles": ["engineer","admin"], "Target": "Part", "Params": [...] } ],
  "Roles": { "viewer": "보기만", "engineer": "...", "admin": "..." },
  "Lineage": { "Dataset": "pdm-v2.2.0", "Synthetic": true, "Preprocessed": "...", "Sources": [...] }
}
```

### costs.json (가정값, 언리얼에서 수정)

```json
{
  "Version": 1, "Currency": "만원", "PlanHours": 6, "UpdatedAt": "...", "UpdatedBy": "default", "Note": "...",
  "Equipments": [
    { "Equip": "C", "EqpTag": "P1-CC1-INS01", "EqpName": "연주 주편",
      "DowntimeCostPerHour": 200,   // 정지 1시간당 생산 차질 비용
      "ReplaceHours": 4,            // 부품 교체 정지 시간
      "RepairCost": 3000,           // 고장 후 복구 비용
      "RepairHours": 8,             // 고장 후 복구 정지 시간
      "FailureLoss": 2000,          // 고장 자체의 추가 손실 (품질·안전 등)
      "PlannedDowntimeRatio": 0.5,  // 계획 교체는 정지 비용이 이 비율만큼만 듦 (0~1)
      "DefaultPart": "valve",
      "Parts": [ { "Part": "valve", "PartName": "valve", "Price": 1200, "Stock": 1, "LeadTimeHours": 24 } ] }
  ]
}
```

### risk_{설비}.json (아래는 C 예시, 다른 설비도 같은 형식)

```json
{
  "Version": 1, "Equip": "C", "EqpTag": "P1-CC1-INS01", "Split": "validation", "IntervalHours": 6,
  "Model": {
    "ModelVersion": "C-ENS-20261019-0031", "Algorithm": "ExtraTrees + HistGradientBoosting (soft voting)", "GeneratedAt": "...",
    "TrainedOn": "train 216개 시나리오", "Features": 94, "FeatureWindowsHours": [6, 12], "ThresholdRule": "...",
    "Validation": { "24H": { "PR_AUC": 0.7616, "Baseline_PR_AUC": 0.1095, "Precision": 0.8254, "Recall": 0.7429, "Alarms": 63, "Points": 639 },
                    "6H": { ... } },
    "Lineage": { "Dataset": "pdm-v2.2.0", "Preprocessed": "...", "FeatureTable": "...", "Code": "jungmin/unreal.py" }
  },
  "Thresholds": { "P6H": 0.3231, "P24H": 0.5662 },
  "WatchRatio": 0.5,
  "Scenarios": [
    { "Scenario": "S0217", "StartAt": "2027-05-15T00:00:00Z",
      "PlannedMaintenance": [ { "StartTSec": 172800, "EndTSec": 187200, "Scope": "정기 부품 교체" } ],   // 없으면 빈 배열
      "Points": [
        { "Time": "2027-05-15T00:00:00Z", "TSec": 0.0,
          "P6H": 0.006, "P24H": 0.1176,                 // 6·24시간 안에 고장 날 확률
          "Alarm6H": false, "Alarm24H": false,
          "Level": "normal",                             // normal | watch | alarm
          "Reasons": [                                   // 정상 범위를 벗어난 특징, 최대 3개 (없으면 빈 배열)
            { "Feature": "w12_CONTROL_DELAY_std", "Sensor": "CONTROL_DELAY", "Stat": "std", "WindowHours": 12,
              "Value": 0.0188, "NormalLow": 0.0, "NormalHigh": 0.0101, "Score": 0.86 } ] } ] }
  ]
}
```

- `Level`: alarm = 6H 또는 24H 확률이 임계값 이상, watch = 24H 확률이 임계값 × WatchRatio 이상, 나머지 normal.
- `Reasons[].Score` = 정상 범위 밖으로 벗어난 거리 ÷ 정상 범위 폭. `Stat`은 mean, std, min, max, last, slope 중 하나(직전 `WindowHours`시간 요약).
- 다른 설비의 `risk_A.json` 등이 같은 형식으로 추가될 수 있다. 설비 코드를 하드코딩하지 말고 `ontology.json`의 `RiskFile`로 찾는다.

## 언리얼이 만드는 파일 (쓰기 반영)

언리얼 프로젝트의 `Saved/DSS/` 폴더에 저장한다. `Content/DSSData` 원본에는 절대 쓰지 않는다.

### Saved/DSS/costs.json — 수정한 가정값
- costs.json과 같은 구조. 저장할 때 `UpdatedAt`(현재 UTC), `UpdatedBy`(사용자 이름) 갱신.
- 시작할 때 이 파일이 있으면 이것을, 없으면 `Content/DSSData/costs.json`을 읽는다.
- 기본 파일에 새 설비·부품이 생기면 기본값으로 추가해 병합한다(기존 수정값은 유지).
- 언리얼 종료 시 저장. 값을 바꿀 때마다 저장해도 된다.

### Saved/DSS/decision_log.json — 정비 결정 기록 (감사 기록)
액션(교체 예약, 대기, 현장 메모)을 실행할 때마다 한 건씩 **추가만** 한다(수정·삭제 없음).

```json
{
  "Decisions": [
    { "DecisionId": "D-20261016-0001", "Timestamp": "2026-10-16T09:00:00Z",
      "User": "jungmin", "Role": "engineer",
      "Scenario": "S0222", "Time": "2027-06-05T06:00:00Z", "Equip": "C", "Part": "valve",
      "ModelVersion": "C-ENS-20261019-0031",
      "P6H": 0.663, "P24H": 0.967, "Level": "alarm",
      "Recommended": "replace_now",
      "Chosen": "replace_now",               // wait | replace_now | replace_6h | note
      "Costs": { "wait": 6384.18, "replace_now": 2000, "replace_6h": 4913.0 },
      "Assumptions": { ...그 순간 설비·부품 가정값 사본... },
      "Note": "현장 확인 결과 밸브 고착 흔적" }
  ]
}
```

Python에서 실제 결과와 비교: `python jungmin/unreal.py feedback=<언리얼프로젝트>/Saved/DSS/decision_log.json`
→ 결정마다 예방 / 불필요 정비 가능성 / 놓침 / 정상 대기를 판정하고, 추천을 따른 결정과 따르지 않은 결정의 결과를 나눠 보여 준다.

## 불러오기·저장 구현 (권장)

1. C++ `USTRUCT(BlueprintType)`로 위 JSON 구조를 그대로 정의한다(속성 이름 = JSON 키).
   예: `FDSSEquipment`, `FDSSPartObject`, `FDSSRelation`, `FDSSAction`, `FDSSOntology`, `FDSSPartCost`, `FDSSEquipmentCost`,
   `FDSSCosts`, `FDSSReason`, `FDSSPoint`, `FDSSScenario`, `FDSSModelInfo`, `FDSSRisk`, `FDSSDecision`, `FDSSDecisionLog`.
2. `UGameInstanceSubsystem`(예: `UDSSSubsystem`)에 데이터를 보관하고 블루프린트용 함수를 둔다.
   - 읽기: `FFileHelper::LoadFileToString` → `FJsonObjectConverter::JsonObjectStringToUStruct`
   - 쓰기: `FJsonObjectConverter::UStructToJsonObjectString` → `FFileHelper::SaveStringToFile(..., EEncodingOptions::ForceUTF8WithoutBOM)`
   - 함수: `LoadAll()`, `GetRisk(Equip)`, `Decide(Point, Equip, Part)`, `SetAssumption(...)`, `SaveCosts()`, `ResetCosts()`,
     `ExecuteAction(ActionName, Scenario, Time, Equip, Part, Note)`, `HasPermission(ActionName)`.
3. `Initialize`에서 `LoadAll`, `Deinitialize`에서 `SaveCosts`.
4. 현재 사용자 이름과 역할(viewer, engineer, admin)을 화면 상단에서 고른다. 각 버튼은 `ontology.json`의 `Actions[].Roles`로 활성·비활성.

## 정비 선택지 계산식 (Python `jungmin/unreal.py`의 `Decide`와 같아야 함)

설비 가정값 E, 부품 가정값 Part, 시점 Point, `PlanHours`(기본 6)에 대해:

```
F  = E.RepairCost + E.RepairHours × E.DowntimeCostPerHour + E.FailureLoss        // 고장 시 손실
R  = Part.Price + E.ReplaceHours × E.DowntimeCostPerHour                           // 즉시 교체
Rp = Part.Price + E.ReplaceHours × E.DowntimeCostPerHour × E.PlannedDowntimeRatio  // 계획 교체

Cost_wait        = Point.P24H × F
Cost_replace_now = R
Cost_replace_6h  = Point.P6H × F + (1 − Point.P6H) × Rp

가능 여부: replace_now  = Part.Stock ≥ 1
           replace_6h   = Part.Stock ≥ 1 또는 Part.LeadTimeHours ≤ PlanHours
           wait         = 항상 가능
Best        = 가능한 선택지 중 비용이 가장 작은 것
Fastest     = 가능한 것 중 replace_now → replace_6h 순서로 첫 번째, 둘 다 불가면 wait
Recommended = Point.Alarm6H 이면 Fastest (안전 우선), 아니면 Best
SafetyOverride = Recommended ≠ Best
PartShortage   = replace_now 불가
UrgentOrder    = Point.Alarm6H 이고 Fastest = wait (경보인데 교체할 부품이 없음 → 긴급 발주)
Saving_vs_wait = Cost_wait − Cost_(Recommended)
```

값을 바꾸면 비용, 추천, 순위가 **즉시 다시 계산**되어야 한다.

## 화면 구성

1. **설비 지도 (공정 흐름)** — `ProcessOrder` 순서로 설비 7개를 배치하고 `FeedsInto` 화살표로 연결.
   설비 색 = 현재 시점 Level(normal 초록, watch 노랑, alarm 빨강, 예측 없음 회색). 설비를 클릭하면 아래 상세로.
2. **위험 시계열** — x축 0~72시간, P6H·P24H 두 선, 두 임계값 가로 점선, 점 색은 Level,
   `PlannedMaintenance` 구간은 회색 띠. 타임라인 슬라이더로 현재 시점 선택(재생 가능).
3. **현재 상태 카드** — P24H 게이지, Level, 이유 최대 3개
   (예: "CONTROL_DELAY 12시간 표준편차 0.019, 정상 범위 0~0.010 초과"), 다음 정기 정비까지 남은 시간,
   이 설비가 멈추면 영향받는 설비(`Downstream`).
4. **what-if 비교 (가정 시나리오)** — 대기 / 즉시 교체 / 6시간 후 계획 교체의 기대 비용 막대, 불가능한 선택지는 회색과 사유.
   가정값 입력란을 바꾸면 **현재 저장값(기준)과 바꾼 값(가정)을 나란히** 보여 주고, [적용]을 눌러야 저장값이 바뀐다. [취소]로 되돌림.
   추천 강조, SafetyOverride면 "안전 우선: 즉시 교체", UrgentOrder면 "긴급 부품 발주 필요" 경고.
5. **결정 실행** — [추천대로 실행] [다른 선택 실행] [현장 메모] 버튼. 실행하면 decision_log.json에 기록하고
   "교체 예약됨" 상태를 해당 부품에 표시한다.
6. **위험 순위표** — 선택 시점에서 시나리오(나중에는 설비)별 `Cost_wait`(기다릴 때 기대 손실) 높은 순.
   열: 대상, Level, P24H, 추천, 기대 비용, 정지 시간(교체 ReplaceHours / 복구 RepairHours), 영향받는 설비 수.
7. **모델 정보 패널** — ModelVersion, 학습 데이터, 검증 성능(PR-AUC, 정밀도, 재현율), 데이터 출처(Lineage).
   "모든 금액은 임의 가정값으로 계산한 시뮬레이션 예상값" 문구.
8. **결정 기록 패널** — decision_log를 최근 순으로 표시(누가, 언제, 어떤 모델 버전, 추천, 선택, 메모).

발표·화면 문구에서는 "서비스" 대신 **"솔루션"** 이라고 쓴다.

## 검증

기본 가정값으로 계산한 결과가 `decisions_{설비}_check.csv`와 같은지 확인한다. 예:

| Scenario | Time | P6H | P24H | Cost_wait | Cost_replace_now | Cost_replace_6h | Recommended |
|---|---|---|---|---|---|---|---|
| S0222 | 2027-06-05T06:00:00Z | 0.663 | 0.967 | 6384.18 | 2000 | 4913.0 | replace_now |

가정값을 바꾼 뒤 Python으로 같은 계산을 확인: `python jungmin/unreal.py costs=<언리얼프로젝트>/Saved/DSS/costs.json`

## 하지 말 것

- 확률·이유를 언리얼에서 다시 계산하거나 수정하지 않는다(모델 결과는 Python이 만든다).
- `Content/DSSData` 원본에 저장하지 않는다.
- decision_log의 기존 기록을 고치거나 지우지 않는다(추가만).
