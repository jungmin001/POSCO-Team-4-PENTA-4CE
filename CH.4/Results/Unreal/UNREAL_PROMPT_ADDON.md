# 언리얼 MCP 추가 작업 프롬프트 — 진단·최종 결함 위험·이상탐지 표시

`UNREAL_PROMPT.md`(정비 의사결정 솔루션)로 만든 화면에 **상태·원인 진단, 최종 결함 위험, 이상탐지 모델 정보**를 추가한다.
`UNREAL_PROMPT.md`의 규칙(온톨로지 구조, 읽기 전용 원본, `Saved/DSS` 쓰기, 권한, "솔루션" 표현)은 그대로 따른다.
아래 내용을 언리얼 MCP에 연결된 Claude에게 그대로 전달하세요.

---

## 목표

이미 있는 설비 지도·위험 시계열·what-if 화면에서 **같은 시나리오, 같은 시각**의 다음 정보를 함께 보여 준다.

| 새 정보 | 파일 | 단위 |
|---|---|---|
| 설비 상태·원인 후보 | `diagnosis_{설비}.csv` | 설비 × 시각 |
| 최종 결함 위험 ("결함 검사 요망") | `flow_quality.csv` | 생산 흐름(주편) |
| 이상탐지 모델 정보 | `anomaly_model.json` | 모델 설명 (시계열 아님) |
| 진단·결함 위험 성능 | `diagnosis_metrics.csv`, `quality_summary.json` | 요약 |

이것은 **미리 계산한 예측을 시간순으로 재생하는 시연**이다. 새 데이터를 받아 예측하지 않는다.
시연 범위는 validation 시나리오 72개(`S0217`~`S0288`)이고, 위험 예측(`risk_*.json`)·진단·결함 위험이 모두 같은 72개다.

## 입력 파일 (Python이 만듦, 읽기 전용)

원본 위치: `POSCO-Team-4-PENTA-4CE/CH.4/Results/Unreal/` → 언리얼 프로젝트 `Content/DSSData/`로 복사
(`UNREAL_PROMPT.md`의 파일들도 같은 폴더에 들어 있다).

- 시작할 때 `manifest.json`을 먼저 읽는다. `Files[]`의 `Status`가 `ok`인 파일만 쓰고, `missing`인 파일의 화면은 "데이터 없음"으로 표시한다(오류로 멈추지 않는다).
- 모든 파일 UTF-8, 한글 포함. 시각은 UTC ISO 형식(`2027-05-15T00:00:00.000Z`). 밀리초가 있거나 없을 수 있으므로 둘 다 파싱한다.
- CSV는 따옴표로 감싼 필드를 처리하는 정식 CSV 파서를 쓴다(필드 안에 쉼표가 있을 수 있다). 불리언은 `True`/`False` 글자, 빈 칸은 "값 없음".

### diagnosis_{설비}.csv — 상태·원인 진단 (설비 7개, A B C D E1 E2 F)

| 열 | 뜻 |
|---|---|
| `SCENARIO_ID`, `EQP_CD`, `MEAS_DT`, `equipment` | 시나리오, 설비 태그, 관측 시각, 설비 이름(A~F) |
| `state_prediction` | `normal` / `degraded` / `failed` / `maintenance` |
| `state_probability` | 상태 예측 확률 (0~1) |
| `needs_review`, `uncertainty_reasons` | 참이면 확정하지 않고 "추가 점검 필요". 이유는 글자 |
| `cause_candidate`, `cause_candidate_2` | 원인 후보 1순위·2순위 (`부품::고장모드`). 상태가 `degraded`/`failed`일 때만 값이 있음 |
| `cause_probability_conditional` | 이상 상태라는 조건에서의 후보 점수 (실제 발생 확률 아님) |
| `cause_needs_review`, `cause_uncertainty_reasons` | 참이면 원인은 후보로만 표시 |
| `actual_state`, `actual_cause` | **시연용 정답**(validation). 운영 예측에는 없는 열 |

- 행은 `SCENARIO_ID`, `MEAS_DT` 순으로 정렬돼 있다. 시나리오마다 샘플 간격이 설비별로 다르다:
  A·C·D 5분(864행), B 10분(432행), F 30분(144행), E1·E2 2시간(36행).
- 행 수(검증용): A 62,208 / B 31,104 / C 62,208 / D 62,208 / E1 2,592 / E2 2,592 / F 10,368.
- 시작할 때 전부 메모리에 올리지 않는다. 설비를 선택할 때 그 설비 파일만 읽고, 시나리오별로 나눠 보관한다
  (A·C·D는 한 파일이 약 6MB, 합쳐도 25MB 정도지만 화면 전환이 빠르도록 지연 로딩).

### flow_quality.csv — 최종 결함 위험 (흐름 단위)

| 열 | 뜻 |
|---|---|
| `SCENARIO_ID`, `FLOW_ID` | 시나리오, 생산 흐름 (예: `S0217-FLOW01`) |
| `PREDICTED_AT` | 예측 시점 = 그 흐름의 C 검사 직후 |
| `defect_score` | 최종(D 공정) 결함 위험 점수 (0~1) |
| `inspection_requested` | `defect_score >= threshold`이면 `True` → "결함 검사 요망" |
| `threshold` | 경보 기준 (0.1311, 모든 행 같음) |
| `origin_eqp`, `actual_defect` | **시연용 정답**: 결함의 원인 설비 태그(없으면 `원인 정보 없음`), 실제 결함 0/1 |

- 설비 고장이 아니라 **주편 하나가 마지막 공정에서 결함이 될 위험**이다. 시나리오당 24~31개 흐름, 전체 1,978개.
- 키가 설비가 아니므로 설비 카드에 섞지 않고 "생산 흐름" 패널에 따로 둔다.

### anomaly_model.json — 이상탐지 모델 (C 연주)

- **시점별 경보 결과가 없는 모델 정의 파일**이다. 시계열에 경보를 그리지 않는다.
- 모델 정보 패널에만 쓴다: `model_type`, `version`, `created_at_utc`,
  `operating_point.selected_metrics`(24시간 내 탐지율 `recall_24h_pct` 95.1%, 6시간 내 `recall_6h_pct` 87.7%,
  하루 오경보 `false_alarms_day` 6.0건, 중간 선행시간 `median_lead_h` 6.45시간, 놓친 건수 `missed_24h` 4),
  `operating_point.selection_reason`.

### diagnosis_metrics.csv, quality_summary.json — 성능 요약

- `diagnosis_metrics.csv`: `equipment`, `task`(state/cause), `macro_f1`, `accuracy`, `accepted_coverage`(자동 채택률) 등. 설비·과제당 한 줄(14줄).
- `quality_summary.json`: `Metrics`에 `AP` 0.9635, `Precision` 0.757, `Recall` 0.9708, `ActualDefects` 308, `Found` 299, `Missed` 9, `FalseAlarms` 96.

## 재생 방식 (시간 맞추기)

기존 화면의 **시나리오 선택과 타임라인 슬라이더 하나**로 모든 정보를 움직인다. 현재 시각 `T`(시나리오 시작 + 슬라이더 시간)에서:

- 위험 예측: 6시간 간격 점 중 `Time ≤ T`인 마지막 점 (기존 방식).
- 진단: 선택한 시나리오·설비에서 `MEAS_DT ≤ T`인 **마지막 행**(as-of). 시간순 정렬이므로 이진 탐색으로 찾는다.
- 결함 위험: `PREDICTED_AT ≤ T`인 흐름들을 모두 "지금까지 예측된 흐름"으로 표시한다. 아직 시각이 안 된 흐름은 숨긴다.
- 데이터가 없는 구간(슬라이더가 첫 행보다 앞)은 "관측 전"으로 표시.
- 시나리오 시작 시각은 `risk_*.json`의 `Scenarios[].StartAt`을 쓴다.

## 화면 추가·변경

기존 화면 번호 기준.

1. **설비 지도 (1번)** — 색은 지금처럼 위험 `Level` 기준으로 유지한다. 설비 노드 옆에 현재 진단 상태를 **작은 아이콘**으로 추가
   (normal 없음, degraded 주황 삼각형, failed 빨강 느낌표, maintenance 파랑 렌치). 두 정보를 한 색으로 합치지 않는다.
2. **현재 상태 카드 (3번)** — 위험 카드 아래에 **진단 섹션**을 추가:
   - 상태(`state_prediction`)와 확률, `needs_review`가 참이면 "추가 점검 필요" 배지와 `uncertainty_reasons`.
   - 상태가 degraded/failed면 **원인 후보** 1·2순위를 `부품::고장모드` 그대로 표시.
     `cause_needs_review`가 참이면 후보 옆에 "후보 (확정 아님)"을 붙인다.
     `cause_probability_conditional`은 "이상 상태일 때의 후보 점수"로 쓰고 확률이라고 쓰지 않는다.
3. **진단 상태 띠 (2번 아래)** — 위험 시계열 아래에 같은 x축으로 상태를 색 띠로 그린다(normal 초록, degraded 주황, failed 빨강, maintenance 파랑).
   `needs_review` 구간은 빗금. 이상(degraded/failed) 구간에 마우스를 올리면 원인 후보를 툴팁으로.
4. **생산 흐름 패널 (새 패널)** — 선택한 시나리오의 흐름 목록: `FLOW_ID`, 예측 시각, `defect_score` 막대, `inspection_requested`가 참이면 "결함 검사 요망".
   임계값(0.1311) 세로선을 표시. 문구는 반드시 **"결함 검사 요망"** 이고 "불량", "불량 확정"이라고 쓰지 않는다.
5. **정답 비교 모드 (토글, 기본 꺼짐)** — 켜면 `actual_state`·`actual_cause`·`actual_defect`를 예측 옆에 나란히 보여 주고
   일치·불일치를 색으로 구분한다. 라벨: "시연용 정답 (운영 예측에는 없음)". 끄면 정답 열은 어디에도 쓰지 않는다.
6. **모델 정보 패널 (7번)** — 모델 3종을 구분해서 표시:
   - 고장 위험(설비별): 기존대로.
   - 상태·원인 진단: 선택 설비의 `macro_f1`·`accepted_coverage`(state, cause 각각).
   - 최종 결함 위험: `quality_summary.json`의 AP·정밀도·재현율·발견/놓침/오경보.
   - 이상탐지(C): 위에 적은 탐지율·오경보·선행시간.
   - 문구: "validation 시나리오로 시연하는 결과이며, 학습 후 임계값·보정에 쓰인 데이터입니다."
7. **위험 순위표 (6번)** — 열에 "현재 진단 상태"를 추가(없으면 `-`). 정렬 기준은 기존 `Cost_wait` 그대로.

## 표시 규칙 (신뢰도)

모델 신뢰도가 낮은 부분을 확정처럼 보여 주지 않는다.

- **원인 진단은 항상 "후보"** 로 표시한다. A·C·F 설비는 원인 자동 채택률이 0%라서 `cause_needs_review`가 거의 항상 참이다. 이것이 정상이다.
  원인 후보 정확도(`diagnosis_metrics.csv`의 cause `macro_f1`)가 낮은 설비(A 0.32, C 0.16, F 0.30)는 후보 옆에 "참고용"을 붙인다.
- `failed` 상태는 전체에서 합쳐 몇 건뿐이라(예: B 1건, C 4건) 학습·검증 사례가 거의 없다. 표시하되 "고장 의심 (사례 적음)"이라고 쓴다.
- 6시간 위험(`Alarm6H`)은 설비마다 검증 고장이 5~16건뿐이라 성능을 믿기 어렵다. "점검 권고"로 표시하고 "고장 확정"이라고 쓰지 않는다. 24시간 위험이 더 안정적이다.
- 진단 상태와 위험 `Level`이 서로 다를 수 있다(예: 위험은 normal인데 진단은 degraded). 어느 쪽도 숨기거나 하나로 합치지 말고 **둘 다 보여 준다**.
- 정상인데 이상으로 예측한 행(오경보)도 그대로 보여 준다. 정답 비교 모드를 켜면 오경보가 드러나는 것이 정상 동작이다.

## 하지 말 것

- 진단 상태, 원인, 결함 점수, 이상탐지 결과를 언리얼에서 계산·수정하지 않는다(모델 결과는 Python이 만든다).
- 새 데이터를 입력받아 예측하는 기능을 만들지 않는다. 이번 범위는 재생 시연이다.
- `anomaly_model.json`으로 시점별 경보를 만들어 그리지 않는다(그 파일에는 경보 결과가 없다).
- `actual_*`, `origin_eqp`, `actual_defect`를 정답 비교 모드 밖에서 사용하지 않는다.
- `Content/DSSData` 원본에 저장하지 않는다. 정비 결정 기록(`decision_log.json`)은 기존 위험 예측 기반 결정만 담는다(진단·결함 위험은 기록 대상이 아님).

## 검증

파일을 읽은 뒤 아래와 맞는지 확인한다.

| 확인 | 기대 값 |
|---|---|
| 시나리오 수 (모든 파일) | 72 (`S0217`~`S0288`) |
| `diagnosis_C.csv` 첫 행 | `S0217`, `P1-CC1-INS01`, `2027-05-15T00:00:00.000Z`, `normal`, 확률 약 0.960 |
| `diagnosis_C.csv`의 `S0217` 행 수 | 864 (5분 간격) |
| `diagnosis_E1.csv`의 `S0217` 행 수 | 36 (2시간 간격) |
| `flow_quality.csv` 행 수 / `inspection_requested`가 True인 수 | 1,978 / 395 |
| `flow_quality.csv`에서 `actual_defect`가 1인 수 | 308 |
| 설비 C 상태 정확도 (`state_prediction == actual_state`) | 약 0.920 |

기대 값과 다르면 파일을 잘못 읽은 것이므로(인코딩, 따옴표, 시각 형식) 화면을 만들기 전에 먼저 고친다.
