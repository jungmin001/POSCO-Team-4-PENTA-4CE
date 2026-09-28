import pandas as pd
# 1. 데이터 불러오기
Df = pd.read_csv(
    "CH.1/Data/Row/sensor.csv"
)

# 2. timestamp를 날짜/시간 자료형으로 변환
Df["timestamp"] = pd.to_datetime(
    Df["timestamp"]
)

# 3. 시간순으로 정렬
# 시계열 데이터이므로
# 과거 → 미래 순서로 정렬한다.
Df = (
    Df
    .sort_values("timestamp")
    .reset_index(drop=True)
)

# 4. machine_status 정리
# 혹시 NORMAL, BROKEN 등에
# 공백이나 소문자가 섞여 있을 가능성을 막기 위해
# 문자열을 정리한다.
Df["machine_status"] = (
    Df["machine_status"]
    .astype("string")
    .str.strip()
    .str.upper()
)

# 5. 실제 고장 발생시간 찾기
# machine_status가 BROKEN인 행에서
# timestamp만 가져온다.
FailureTimes = (
    Df.loc[
        Df["machine_status"] == "BROKEN",
        "timestamp"
    ]
    .sort_values()
    .tolist()
)

print(
    "실제 고장 횟수:",
    len(FailureTimes)
)

# 6. 몇 시간 전부터 고장위험으로 볼지 설정
# 현재는 고장 발생 6시간 전부터
# 고장 직전까지를 '고장위험'으로 정의한다.
FailureWindowHours = 6

# 7. 고장위험 라벨 생성
# failure_soon
#
# 0 = 정상
# 1 = 고장위험
#
# 우선 모든 행을 정상(0)으로 설정한다.
Df["failure_soon"] = 0

# 실제 고장 발생시간을 하나씩 확인한다.
for FailureTime in FailureTimes:
    # 현재 고장 발생시간을 기준으로
    #
    # 고장 6시간 전 이상
    # AND
    # 실제 고장시간 미만
    #
    # 인 행을 찾는다.
    Mask = (
        (
            Df["timestamp"]
            >= FailureTime - pd.Timedelta(
                hours=FailureWindowHours
            )
        )
        &
        (
            Df["timestamp"]
            < FailureTime
        )
    )
    # 위 조건을 만족하는 구간을
    # 고장위험(1)으로 변경한다.
    Df.loc[
        Mask,
        "failure_soon"
    ] = 1


# 8. 모델링 대상 데이터 생성
# 이미 고장이 난 BROKEN 상태나
# 고장 후 복구 중인 RECOVERING 상태를 제외한다.
#
# 정상 운전 중에 앞으로 고장이 날지를 예측하는 것이 목적이므로
# NORMAL 상태만 사용한다.
Data = Df[
    Df["machine_status"] == "NORMAL"
].copy()


# 9. 정상 / 고장위험 개수 확인
print(
    "\n===== 정상 / 고장위험 개수 ====="
)

print(
    Data["failure_soon"]
    .value_counts()
)


# 10. 정상 / 고장위험 비율 확인
print(
    "\n===== 정상 / 고장위험 비율 ====="
)

print(
    Data["failure_soon"]
    .value_counts(
        normalize=True
    )
    * 100
)

# 11. 고장위험 비율만 따로 계산
# failure_soon은 0과 1로만 이루어져 있다.
# 따라서 평균을 구하면
# 1이 차지하는 비율을 바로 계산할 수 있다.
FailureRatio = (
    Data["failure_soon"]
    .mean()
    * 100
)

print(
    "\n고장위험 비율:",
    round(FailureRatio, 2),
    "%"
)
# 12. 고장 비율 분석 결과 및 모델링 방향
# 실제 고장 발생 횟수는 총 7회이다.
#
# 고장 발생 6시간 전부터 고장 직전까지를
# 고장위험 구간(failure_soon = 1)으로 정의하였다.
#
# 원본 데이터는 1분 단위로 측정되므로
# 고장 1회당 6시간의 위험구간은 최대 360개의 행이 생성된다.
#
# 6시간 × 60분 = 360행
#
# 실제 고장 횟수가 7회이므로
#
# 360행 × 7회 = 2,520행
#
# 의 고장위험 데이터가 생성되었다.
#
#
# 정상 데이터:
# 203,316행
# 약 98.78%
#
# 고장위험 데이터:
# 2,520행
# 약 1.22%
#
#
# 따라서 전체 데이터에서 고장위험 데이터의 비율은
# 약 1.22%로 매우 낮으며,
# 정상 데이터와 고장위험 데이터 사이에
# 큰 클래스 불균형이 존재한다.
#
#
# 또한 고장위험 데이터가 2,520행 존재하지만,
# 이는 2,520개의 서로 독립적인 고장 사례가 아니다.
#
# 실제로는 7회의 고장 이벤트에서
# 각각 고장 전 6시간 구간을 잘라서 만든 데이터이므로,
# 독립적인 고장 사례 자체는 총 7회뿐이다.
#
#
# 즉,
#
# - 정상 데이터 비율이 약 98.78%로 매우 높음
# - 고장위험 데이터 비율은 약 1.22%로 매우 낮음
# - 실제 독립적인 고장 이벤트는 7회뿐임
#
# 이러한 이유로 고장 데이터가 충분히 많다고 보기 어렵다. (!!!!! 핵심 !!!!!!)
#
#
# 따라서 모델링 방식은
# 정상 상태의 패턴을 학습한 뒤
# 정상 패턴에서 벗어나는 데이터를 탐지하는
# 반지도학습 또는 이상탐지 방식의 적용을 우선 검토한다.
#
#
# 다만 failure_soon이라는 0 / 1 라벨을 생성할 수 있으므로
# Random Forest와 같은 지도학습 분류모델도 적용할 수 있다.
#
# 따라서 이후에는
#
# 1. 이상탐지 모델
# 2. 지도학습 모델
# 을 비교하여 성능을 확인하는 방향도 고려할 수 있다.
#
#
# 특히 지도학습 모델의 성능을 평가할 때는
# Accuracy만 보는 것이 아니라
# Precision, Recall, F1-score, Confusion Matrix를 함께 확인해야 한다.
#
# 정상 데이터가 약 98.78%이기 때문에
# 모든 데이터를 정상이라고 예측해도
# Accuracy가 높게 나올 수 있기 때문이다.

#======================================
# 13. 시계열 담당이 선정한 센서 설정
# 시계열 분석 결과를 바탕으로
# 모델링에 사용할 후보 센서를 설정한다.
#
# 아래 센서들은 시계열 담당이
# 고장 전후 패턴 분석에 사용한 센서들이다.

CandidateSensors = [
    "sensor_04",
    "sensor_02",
    "sensor_13",
    "sensor_33",
    "sensor_32",
    "sensor_28",
]

# 실제 데이터에 존재하는 센서만 사용한다.
#
# 예를 들어 CandidateSensors에 sensor_04가 있어도
# 실제 CSV에 sensor_04가 없다면 자동으로 제외된다.

Sensors = [
    Sensor
    for Sensor in CandidateSensors
    if Sensor in Df.columns
]


print(
    "\n사용 센서:",
    Sensors
)

# 각 센서마다 아래 5개의 시계열 특성을 생성한다.
for Sensor in Sensors:

    # 10분 전 센서값
    Df[f"{Sensor}_lag10"] = (
        Df[Sensor]
        .shift(10)
    )

    # 30분 전 센서값
    Df[f"{Sensor}_lag30"] = (
        Df[Sensor]
        .shift(30)
    )

    # 최근 30분 센서 평균
    Df[f"{Sensor}_mean30"] = (
        Df[Sensor]
        .rolling(30)
        .mean()
    )

    # 최근 30분 센서 표준편차
    # 센서값이 얼마나 불안정하게 흔들리는지 확인
    Df[f"{Sensor}_std30"] = (
        Df[Sensor]
        .rolling(30)
        .std()
    )

    # 현재값 - 10분 전 값
    # 최근 센서값 변화량
    Df[f"{Sensor}_diff10"] = (
        Df[Sensor]
        -
        Df[Sensor].shift(10)
    )

# ============================================================
# 15. 모델 입력 특성 목록 만들기
# ============================================================

FeatureCols = []


for Sensor in Sensors:

    FeatureCols += [
        f"{Sensor}_lag10",
        f"{Sensor}_lag30",
        f"{Sensor}_mean30",
        f"{Sensor}_std30",
        f"{Sensor}_diff10",
    ]

# 여기서 for문 끝


print(
    "\n생성된 시계열 특성 개수:",
    len(FeatureCols)
)

print(
    "\n사용할 시계열 특성:"
)

print(
    FeatureCols
)

# 16. 시계열 특성의 결측행 제거

# lag / rolling 특성을 만들면
# 데이터 처음 부분에는 과거 정보가 부족하여
# NaN이 발생한다.
#
# 모델은 NaN이 포함된 데이터를 바로 사용할 수 없으므로
# 시계열 특성이 모두 존재하는 행만 남긴다.

ModelData = (
    Df
    .dropna(
        subset=FeatureCols
    )
    .copy()
)


print(
    "\n시계열 특성 결측 제거 후 데이터 크기:",
    ModelData.shape
)

# ============================================================
# 17. 모델링에 사용할 데이터 정리
# ============================================================

# 먼저 시계열 특성에 결측값이 있는 행을 제거한다.
ModelData = (
    Df
    .dropna(subset=FeatureCols)
    .copy()
)

# 그중 NORMAL 상태만 남긴다.
ModelData = ModelData[
    ModelData["machine_status"] == "NORMAL"
].copy()


print(
    "\n모델링 데이터 크기:",
    ModelData.shape
)


# 18. 시간 기준 Train / Test 분할
# ============================================================

# 실제 고장 이벤트는 총 7회이다.
#
# 마지막 2개의 고장 이벤트를
# 모델이 학습하지 않은 미래 구간으로 남겨두기 위해
# 두 번째 마지막 고장 발생 12시간 전을
# Train / Test 분할 기준으로 사용한다.

SplitTime = (
    FailureTimes[-2]
    - pd.Timedelta(hours=12)
)


print(
    "\nTrain / Test 분할 시간:",
    SplitTime
)


# ------------------------------------------------------------
# 과거 데이터 = Train
# 미래 데이터 = Test
# ------------------------------------------------------------

TrainData = ModelData[
    ModelData["timestamp"] < SplitTime
].copy()

TestData = ModelData[
    ModelData["timestamp"] >= SplitTime
].copy()


print(
    "\nTrain 데이터 크기:",
    TrainData.shape
)

print(
    "Test 데이터 크기:",
    TestData.shape
)

# 19. Isolation Forest 학습용 정상 데이터 생성
# ============================================================

# Train 데이터 중에서도
# failure_soon = 0인 완전 정상구간만 학습에 사용한다.
#
# failure_soon = 1은 고장 전 위험구간이므로
# 정상 패턴 학습에서는 제외한다.

NormalTrainData = TrainData[
    TrainData["failure_soon"] == 0
].copy()


# 모델이 공부할 입력값 X
XTrain = NormalTrainData[
    FeatureCols
]


# 테스트에서는 정상과 고장위험을 모두 사용한다.
XTest = TestData[
    FeatureCols
]

# 실제 정답
YTest = TestData[
    "failure_soon"
]


print(
    "\n정상 학습 데이터:",
    XTrain.shape
)

print(
    "테스트 데이터:",
    XTest.shape
)

print(
    "\n테스트 데이터 정상 / 고장위험 개수:"
)

print(
    YTest.value_counts()
)


from sklearn.ensemble import IsolationForest
# 20. Isolation Forest 모델 생성
# ============================================================

# Isolation Forest:
# 정상 데이터와 다른 패턴을 가진 데이터를
# 이상치로 탐지하는 모델
#
# random_state=42
# → 실행할 때마다 결과가 크게 달라지지 않도록 고정
#
# n_estimators=150
# → 150개의 트리를 사용하여 이상 여부를 판단
#
# contamination="auto"
# → 이상치 비율을 미리 강제로 지정하지 않고
#   모델 기본 기준을 사용

Model = IsolationForest(
    n_estimators=150,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

# 21. 정상 데이터 학습
# ============================================================

Model.fit(
    XTrain
)

print(
    "\nIsolation Forest 학습 완료"
)

# 22. Test 데이터 이상 탐지
# ============================================================

IsolationPred = Model.predict(
    XTest
)


# Isolation Forest의 출력:
#
#  1 = 정상
# -1 = 이상
#
# 우리 failure_soon의 기준:
#
# 0 = 정상
# 1 = 고장위험
#
# 따라서 비교하기 위해
# -1을 1로,
#  1을 0으로 변환한다.

YPred = (
    IsolationPred == -1
).astype(int)


print(
    "\n예측 결과 처음 20개:"
)

print(
    YPred[:20]
)

# 23. Isolation Forest 예측 결과 개수 확인
# ============================================================

# YPred의 의미:
#
# 0 = Isolation Forest가 정상이라고 판단
# 1 = Isolation Forest가 이상이라고 판단
#
# 전체 Test 데이터에서
# 모델이 각각 몇 개를 정상 / 이상으로 판단했는지 확인한다.

PredictionCount = pd.Series(
    YPred
).value_counts().sort_index()

print(
    "\n===== Isolation Forest 예측 개수 ====="
)

print(
    PredictionCount
)

# 24. 평가 지표 불러오기
# ============================================================

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# 25. Confusion Matrix 계산
# ============================================================

# YTest:
# 실제 정답
#
# YPred:
# Isolation Forest의 예측
#
# labels=[0, 1]:
# 0을 정상,
# 1을 고장위험으로 고정하여 계산한다.

ConfusionMatrix = confusion_matrix(
    YTest,
    YPred,
    labels=[0, 1]
)


print(
    "\n===== Confusion Matrix ====="
)

print(
    ConfusionMatrix
)


# Confusion Matrix 구조:
#
# [[TN, FP],
#  [FN, TP]]
#
# 각각의 값을 변수에 저장한다.

Tn, Fp, Fn, Tp = (
    ConfusionMatrix.ravel()
)


print(
    "\nTN (정상을 정상으로 판단):",
    Tn
)

print(
    "FP (정상인데 이상으로 판단 / 오탐):",
    Fp
)

print(
    "FN (고장위험인데 정상으로 판단 / 미탐):",
    Fn
)

print(
    "TP (고장위험을 고장위험으로 판단):",
    Tp
)

# 26. Precision / Recall / F1-score 계산
# ============================================================

# ------------------------------------------------------------
# Precision (정밀도)
# ------------------------------------------------------------
#
# 모델이 "이상이다"라고 경보한 것들 중
# 실제로 고장위험이었던 비율
#
# Precision = TP / (TP + FP)
#
# 높을수록 헛경보가 적다는 의미

Precision = precision_score(
    YTest,
    YPred,
    zero_division=0
)


# ------------------------------------------------------------
# Recall (재현율)
# ------------------------------------------------------------
#
# 실제 고장위험 데이터 중에서
# 모델이 얼마나 많이 찾아냈는지
#
# Recall = TP / (TP + FN)
#
# 높을수록 실제 고장위험을
# 덜 놓쳤다는 의미

Recall = recall_score(
    YTest,
    YPred,
    zero_division=0
)


# ------------------------------------------------------------
# F1-score
# ------------------------------------------------------------
#
# Precision과 Recall을
# 동시에 고려하는 지표
#
# 클래스 불균형이 큰 현재 데이터에서는
# Accuracy보다 중요한 평가 지표 중 하나

F1 = f1_score(
    YTest,
    YPred,
    zero_division=0
)


print(
    "\n===== Isolation Forest 성능 ====="
)

print(
    "Precision:",
    round(Precision, 4)
)

print(
    "Recall:",
    round(Recall, 4)
)

print(
    "F1-score:",
    round(F1, 4)
)

# 27. Isolation Forest 결과 해석
# ============================================================

# Test 데이터의 실제 고장위험 데이터는 총 720행이었다.
#
# Isolation Forest 평가 결과:
#
# TN = 75,577
# FP = 3,366
# FN = 715
# TP = 5
#
#
# Precision = 0.0015
# Recall    = 0.0069
# F1-score  = 0.0024
#
#
# 실제 고장위험 720행 중
# 모델이 정상적으로 탐지한 것은 5행뿐이며,
# 715행의 고장위험을 정상으로 판단하였다.
#
# 따라서 실제 고장위험을 탐지하는 Recall이
# 약 0.69%로 매우 낮게 나타났다.
#
#
# 또한 모델이 이상이라고 판단한 데이터는
# 총 3,371행이었지만,
# 이 중 실제 고장위험은 5행뿐이었다.
#
# 즉 대부분의 이상 탐지가 실제 고장위험과
# 연결되지 않는 오탐(FP)이었다.
#
#
# 결과적으로 현재 시계열 특성과
# Isolation Forest 조합은
# 고장 전 위험 패턴을 탐지하는 데 적합하지 않은 것으로 판단된다.
#
#
# 고장위험 데이터의 비율이 약 1.22%로 낮아
# 이상탐지 방식을 우선 검토하였으나,
# 단순히 정상 패턴에서 벗어나는 이상을 찾는 것과
# 실제 고장 전 징후를 탐지하는 것은
# 서로 다른 문제일 수 있음을 확인하였다.

# 28. Random Forest 지도학습 모델
# ============================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# Train 데이터
XTrainRf = TrainData[FeatureCols]
YTrainRf = TrainData["failure_soon"]

# Test 데이터
XTestRf = TestData[FeatureCols]
YTestRf = TestData["failure_soon"]


# ============================================================
# 29. Random Forest 모델 생성 및 학습
# ============================================================

# class_weight="balanced"
# → 정상 데이터가 훨씬 많은 클래스 불균형을 보정
RandomForestModel = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

RandomForestModel.fit(
    XTrainRf,
    YTrainRf
)


# ============================================================
# 30. Test 데이터 예측
# ============================================================

RandomForestPred = RandomForestModel.predict(
    XTestRf
)


# ============================================================
# 31. Confusion Matrix
# ============================================================

RandomForestCm = confusion_matrix(
    YTestRf,
    RandomForestPred,
    labels=[0, 1]
)

TnRf, FpRf, FnRf, TpRf = RandomForestCm.ravel()


# ============================================================
# 32. Precision / Recall / F1-score
# ============================================================

PrecisionRf = precision_score(
    YTestRf,
    RandomForestPred,
    zero_division=0
)

RecallRf = recall_score(
    YTestRf,
    RandomForestPred,
    zero_division=0
)

F1Rf = f1_score(
    YTestRf,
    RandomForestPred,
    zero_division=0
)


# ============================================================
# 33. 결과 출력
# ============================================================

print("\n===== Random Forest 결과 =====")

print("\nConfusion Matrix")
print(RandomForestCm)

print("\nTN:", TnRf)
print("FP:", FpRf)
print("FN:", FnRf)
print("TP:", TpRf)

print("\nPrecision:", round(PrecisionRf, 4))
print("Recall:", round(RecallRf, 4))
print("F1-score:", round(F1Rf, 4))


# ============================================================
# 34. 모델링 방향 정리
# ============================================================

# 고장위험 데이터의 비율은 약 1.22%이며
# 실제 고장 이벤트도 7회로 적어
# 우선 Isolation Forest 이상탐지 모델을 적용하였다.
#
# 그러나 Isolation Forest 결과:
#
# Precision = 0.0015
# Recall    = 0.0069
# F1-score  = 0.0024
#
# 로 실제 고장위험을 충분히 탐지하지 못하였다.
#
# 따라서 failure_soon이라는 명확한 0 / 1 라벨을 활용하여
# Random Forest 지도학습 모델을 추가로 적용하였다.
#
# Random Forest에는 class_weight="balanced"를 적용하여
# 정상 98.78%, 고장위험 1.22%의 클래스 불균형을 보정하였다.
#
# 두 모델은 동일한 시계열 특성과 동일한 시간 기준
# Train / Test 데이터를 사용하여 성능을 비교한다.
#
# 최종 모델은 Accuracy만으로 판단하지 않고
# Precision, Recall, F1-score와
# FP, FN을 함께 비교하여 결정한다.

# 35. Random Forest 결과 해석 및 최종 판단
# ============================================================

# Random Forest 평가 결과:
#
# TN = 78,943
# FP = 0
# FN = 720
# TP = 0
#
# Precision = 0.0
# Recall    = 0.0
# F1-score  = 0.0
#
#
# Random Forest는 Test 데이터의 모든 행을
# 정상(0)으로 예측하였다.
#
# 따라서 실제 고장위험 데이터 720행을
# 하나도 탐지하지 못하였으며,
# 모든 고장위험 데이터를 미탐(FN)하였다.
#
#
# class_weight="balanced"를 적용하였지만
# 실제 독립적인 고장 이벤트가 총 7회로 매우 적고,
# 고장위험 데이터 비율도 약 1.22%에 불과하여
# 학습 가능한 고장 패턴이 충분하지 않은 것으로 판단된다.
#
#
# Isolation Forest 결과:
#
# Precision = 0.0015
# Recall    = 0.0069
# F1-score  = 0.0024
#
# Random Forest 결과:
#
# Precision = 0.0
# Recall    = 0.0
# F1-score  = 0.0
#
#
# 두 모델 모두 현재 시계열 특성으로는
# 고장위험을 충분히 탐지하지 못하였다.
#
#
# 데이터 구성 측면에서는
# 정상 데이터가 약 98.78%,
# 고장위험 데이터가 약 1.22%이며,
# 실제 고장 이벤트도 7회뿐이다.
#
# 따라서 현재 데이터 특성상
# 지도학습보다는 정상 패턴을 중심으로 학습하는
# 반지도학습 / 이상탐지 방식을 우선 검토하는 것이
# 데이터 구조상 더 적절하다고 판단하였다.
#
#
# 다만 현재 Isolation Forest 성능 역시 매우 낮으므로,
# 이후에는 센서 및 시계열 특성 재선정,
# 고장위험 시간구간 조정,
# 이상탐지 임계값 조정 등의 추가 개선이 필요하다.
