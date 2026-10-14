# 데이터셋마다 달라지는 값 모음, 다른 설비·데이터셋에 쓸 때는 이 파일만 바꿈

# ---- 설비마다 달라지는 값 (기본값은 연주 C, 아래 Profiles에서 설비별로 덮어씀) ----
# 원자료 종류(manifest.json의 files[].id)와 설비 ID (None이면 설비로 거르지 않음)
Source = "casting"
EqpCode = "P1-CC1-INS01"

# 시각이 같을 때 순서를 정하는 컬럼, 시나리오 안에서 구간을 나누는 컬럼 (직전 값 비교는 같은 구간 안에서만)
TieCols = ["SEQ_NO"]
GroupCols = ["SLAB_NO"]

# 센서가 아닌 컬럼 (부하·설정값·식별 코드·가동 여부), 나머지 숫자 컬럼은 모두 센서로 봄
ExcludeCols = ["LOAD", "RUNNING"]

# 정지·품질 판정 컬럼 (데이터에 없으면 건너뜀)
RunCol = "RUNNING"
QualityCol = "QUALITY_FLAG"

# ---- 모든 설비에 공통인 값 ----
EqpCol = "EQP_CD"
TimeCol = "MEAS_DT"
StopValue = 0
QualityOk = "ok"

# 정비 이력 (정비 구간: 시작 <= 시각 < 종료, 종료가 공란이면 관측 끝까지)
MaintSource = "maintenance"
MaintEqpCol = "EQP_TAG"
MaintStartCol = "STRT_DT"
MaintEndCol = "END_DT"

# 결측: 연속 FillMaxGap행 이하는 직전 값으로 채움, 더 길면 채우지 않음
FillMaxGap = 3

# 이상치: 직전 Window개 값의 중앙값에서 NSigma * 척도 이상 벗어나면 이상치
# 척도 하한은 중앙값의 RelFloor 비율 (거의 일정한 신호의 작은 변화가 잡히지 않게)
Window = 11
DetectNSigma = 10.0
RelFloor = 0.01

# 훑어보기: 시나리오별 요약(정지·결측·정비 행 제외 min/max/mean)이 train 분포의 중앙값에서 NSigma * 1.4826 * MAD 이상 벗어나면 의심
ScreenStats = ("min", "max", "mean")
ScreenNSigma = 6.0

# 주변 탐지: 의심 지점 앞뒤로 볼 행 수
ZoomRows = 60

# 정답 파일 종류별 시각 컬럼 (분석 확인에만 사용)
TruthTimeCols = {
    "diagnostics": ["START_AT", "END_AT"],
    "labels": ["OBSERVED_AT"],
    "targets": ["OBSERVED_AT"],
}
DegradedStates = ("degraded", "failed")
HealthCol = "HEALTH_STATE"

# 결과 저장 폴더 (CH.4 기준)
OutDirParts = ("Data", "Processed")

# 시나리오별 전처리 완료 파일: Scenarios/{시나리오}/{설비}.csv
ScenarioDirName = "Scenarios"
TimeFormat = "%Y-%m-%dT%H:%M:%SZ"
FlagNames = {"missing": "IS_MISSING", "stopped": "IS_STOPPED", "maintenance": "IS_MAINTENANCE", "outlier": "IS_OUTLIER"}

# 시나리오 폴더에 함께 넣는 표 (종류별 시각 열 -> 경과 초 열 이름)
# 흐름 정보: Raw/flow/{종류}/{시나리오}.csv -> Scenarios/{시나리오}/flow/{종류}.csv
FlowTimeCols = {
    "operations": {"START_AT": "T_START_SEC", "END_AT": "T_END_SEC"},
    "inspections": {"OBSERVED_AT": "T_SEC"},
    "quality": {},
}
FlowTruthKinds = ["quality"]
FlowKeyCols = ["SCENARIO_ID", "FLOW_ID", "STAGE", "EQP_CD"]

# 정답: Raw/{종류}/{시나리오}.csv -> Scenarios/{시나리오}/answers/{종류}.csv (train·validation만)
# 키 열을 뺀 모든 열에 TruthPrefix를 붙임, 입력 특징으로 쓰지 않음
AnswerTimeCols = {
    "labels": {"OBSERVED_AT": "T_SEC"},
    "targets": {"OBSERVED_AT": "T_SEC"},
    "diagnostics": {"START_AT": "T_START_SEC", "END_AT": "T_END_SEC"},
}
AnswerKeyCols = {
    "labels": ["SCENARIO_ID", "EQP_CD", "OBSERVED_AT"],
    "targets": ["SCENARIO_ID", "EQP_CD", "OBSERVED_AT"],
    "diagnostics": ["SCENARIO_ID", "EQP_CD", "START_AT", "END_AT"],
}

# 정비 이력(원본 G 6종): 원자료 종류(manifest의 files[].id) -> Scenarios/{시나리오}/history/{종류}.csv
# 정비 이력 maintenance, 교체 내역 replacements, 부품 parts, 정비 계획 calendar, 운전 조건 변경 events, 설비 목록 equipment
# 시각이 있는 행은 예측 시각 이전의 것만 사용해야 함
HistoryTimeCols = {
    "maintenance": {"DETC_DT": "T_DETECT_SEC", "FAILURE_DT": "T_FAILURE_SEC", "STRT_DT": "T_START_SEC", "END_DT": "T_END_SEC"},
    "replacements": {},
    "parts": {},
    "calendar": {"STRT_DT": "T_START_SEC", "END_DT": "T_END_SEC"},
    "events": {"EVT_DT": "T_SEC"},
    "equipment": {},
}

# 정답지에서 붙이는 상태 열 (연출·분석 확인용이며 모델 입력으로 쓰지 않음, test 시나리오는 비어 있음)
TruthPrefix = "GT_"
TruthCols = ["HEALTH_STATE", "CAUSE_PART", "FAULT_MODE"]
TruthStartCol = "START_AT"
TruthEndCol = "END_AT"

# ---- 설비별 설정: 기본값과 다른 항목만 적음 ----
Profiles = {
    "A": {"Source": "annealing", "EqpCode": "P1-CR1-CAL01", "GroupCols": ["LOT_NO"],
          "ExcludeCols": ["LOAD", "RUNNING", "GRADE", "AMBIENT", "SET_TEMP"]},
    # 코일 안의 13개 행은 용접선 위치마다 값이 달라서, 같은 위치(SEQ_NO)끼리 코일 순서대로 비교
    "B": {"Source": "welding", "EqpCode": "P1-CR1-WLD01", "GroupCols": ["SEQ_NO"],
          "ExcludeCols": ["RUNNING", "SEQ_NO", "GRADE", "THICKNESS", "SET-PWR"]},
    "C": {},
    "D": {"Source": "temper-features", "EqpCode": "P1-CR1-SPM01", "GroupCols": [],
          "ExcludeCols": ["LOAD", "N", "FS_HZ"]},
    # E는 시나리오당 36행뿐이라 기본 Window(11)면 앞 21행이 판정되지 않아 Window를 줄임
    "E1": {"Source": "fan-features", "EqpCode": "P1-SP1-EXF01", "GroupCols": [], "Window": 5,
           "ExcludeCols": ["LOAD", "N", "FS_HZ"]},
    "E2": {"Source": "fan-features", "EqpCode": "P1-SP2-EXF01", "GroupCols": [], "Window": 5,
           "ExcludeCols": ["LOAD", "N", "FS_HZ"]},
    "F": {"Source": "agitator-features", "EqpCode": "P1-SP1-AGT01", "GroupCols": [],
          "ExcludeCols": ["LOAD", "N", "FS_HZ"]},
}

_ProfileKeys = ["Source", "EqpCode", "TieCols", "GroupCols", "ExcludeCols", "RunCol", "QualityCol", "Window"]
_Defaults = {Key: globals()[Key] for Key in _ProfileKeys}
Name = "C"


# 설비 설정 적용 (기본값으로 되돌린 뒤 해당 설비의 항목만 덮어씀), 결과 파일 이름에 Name이 붙음
def Use(ProfileName):
    global Name

    globals().update(_Defaults)
    globals().update(Profiles[ProfileName])
    Name = ProfileName
