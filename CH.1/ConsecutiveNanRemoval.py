import pandas as pd
# ----------------     Test1_machine_data.csv    ----------------------------
Df = pd.read_csv("data/sensor.csv") # senser 데이터 불러오기

DfRaw = Df.copy() # 원본 보관 변수
DfClean = Df.copy() # 데이터 전처리 작업용 변수


# =========================================================
# 1. 불필요한 열 제거
# =========================================================

if "Unnamed: 0" in DfClean.columns:
    DfClean = DfClean.drop(columns=["Unnamed: 0"]) # sensor의 단순 행열 삭제

# =========================================================
# 2. timestamp 자료형 변환
# =========================================================

DfClean["timestamp"] = pd.to_datetime(DfClean["timestamp"], errors="coerce") # 시간순 정렬과 시간 기반 결측치 보간


# timestamp 자체가 없는 행 제거
DfClean = DfClean.dropna(subset=["timestamp"])


# =========================================================
# 3. 시간순 정렬
# =========================================================

DfClean = DfClean.sort_values("timestamp").reset_index(drop=True)


# =========================================================
# 4. 완전히 동일한 중복 행 제거
# =========================================================

DfClean = DfClean.drop_duplicates().reset_index(drop=True)


# =========================================================
# 5. 센서 열 찾기
# =========================================================

SensorCols = [Col for Col in DfClean.columns if Col.startswith("sensor_")]


# =========================================================
# 6. 센서값을 숫자형으로 변환
# =========================================================

for Col in SensorCols:
    DfClean[Col] = pd.to_numeric(DfClean[Col], errors="coerce")


# =========================================================
# 7. 결측률 30% 이상 센서 제거
# =========================================================

MissingRate = DfClean[SensorCols].isna().mean() * 100

HighMissingCols = MissingRate[MissingRate >= 30].index.tolist()


print("결측률 30% 이상 센서")
print(HighMissingCols)


DfClean = DfClean.drop(columns=HighMissingCols)


# 센서 목록 다시 설정
SensorCols = [Col for Col in DfClean.columns if Col.startswith("sensor_")]


# =========================================================
# 8. 결측치가 50개 이상 연속되는 센서 찾기
# =========================================================

LONG_GAP = 50


def MaxConsecutiveNan(series):
    """
    한 열에서 NaN이 최대 몇 번 연속되는지 계산
    """

    is_nan = series.isna()

    # True / False가 바뀔 때마다 그룹 생성
    groups = (is_nan != is_nan.shift()).cumsum()

    # NaN인 부분만 그룹별 개수 계산
    nan_runs = is_nan[is_nan].groupby(groups[is_nan]).size()

    if len(nan_runs) == 0:
        return 0

    return nan_runs.max()


long_gap_info = {}

for Col in SensorCols:

    max_gap = MaxConsecutiveNan(DfClean[Col])

    long_gap_info[Col] = max_gap


long_gap_Df = pd.DataFrame(
    long_gap_info.items(), columns=["sensor", "max_consecutive_missing"]
)

long_gap_Df = long_gap_Df.sort_values("max_consecutive_missing", ascending=False)


print("\n센서별 최대 연속 결측치")
print(long_gap_Df)


# 50개 이상 연속 결측 센서
long_gap_Cols = long_gap_Df[long_gap_Df["max_consecutive_missing"] >= LONG_GAP][
    "sensor"
].tolist()


print("\n50개 이상 연속 결측 센서")
print(long_gap_Cols)


# =========================================================
# 9. 50개 이상 연속 결측 센서 제거
# =========================================================

DfClean = DfClean.drop(columns=long_gap_Cols)


# 센서 목록 다시 갱신
SensorCols = [Col for Col in DfClean.columns if Col.startswith("sensor_")]


# =========================================================
# 10. machine_status 라벨 정리
# =========================================================

if "machine_status" in DfClean.columns:

    DfClean["machine_status"] = (
        DfClean["machine_status"].astype("string").str.strip().str.upper()
    )


# =========================================================
# 11. 남아있는 짧은 결측치 보간
# =========================================================
#
# 이미 50개 이상 긴 결측을 가진 센서는 제거했으므로
# 남은 센서의 짧은 결측을 시간 흐름에 맞춰 보간
#

DfClean = DfClean.set_index("timestamp")


DfClean[SensorCols] = DfClean[SensorCols].interpolate(
    method="time", limit_direction="both"
)


# timestamp 다시 열로 복원
DfClean = DfClean.reset_index()


# =========================================================
# 12. 최종적으로 남은 결측치 확인
# =========================================================

RemainingMissing = DfClean[SensorCols].isna().sum()

print("\n전처리 후 남아있는 결측치")
print(RemainingMissing[RemainingMissing > 0])


# =========================================================
# 13. 최종 중복 재확인
# =========================================================

DuplicateCount = DfClean.duplicated().sum()

print("\n남아있는 완전 중복 행:", DuplicateCount)


# =========================================================
# 14. 시간순 재정렬
# =========================================================

DfClean = DfClean.sort_values("timestamp").reset_index(drop=True)


# =========================================================
# 15. 최종 데이터 상태 확인
# =========================================================

print("\n============================")
print("전처리 완료")
print("============================")

print("원본 데이터 크기:", DfRaw.shape)

print("최종 데이터 크기:", DfClean.shape)

print("\n제거된 결측률 30% 이상 센서:")

print(HighMissingCols)

print("\n제거된 50개 이상 연속 결측 센서:")

print(long_gap_Cols)


print("\n최종 센서 개수:", len(SensorCols))


print("\n전체 결측치:", DfClean.isna().sum().sum())


print("\n라벨 분포")

if "machine_status" in DfClean.columns:
    print(DfClean["machine_status"].value_counts(dropna=False))


print("\n최종 데이터 정보")

DfClean.info()


print("\n센서 기초 통계")

print(DfClean[SensorCols].describe())


# =========================================================
# 16. 전처리 완료 CSV 저장
# =========================================================

DfClean.to_csv("Test1_machine_data.csv", index=False)

print("\nTest1_machine_data.csv 저장 완료")

# -------------------------
import pandas as pd


# =========================================================
# 1. 전처리된 데이터 불러오기
# =========================================================

Df = pd.read_csv("Test2_machine_data.csv")

# Test1을 확인하고 싶다면 아래처럼 변경
# Df = pd.read_csv("Test1_machine_data.csv")


# =========================================================
# 2. timestamp 자료형 변환
# =========================================================

Df["timestamp"] = pd.to_datetime(
    Df["timestamp"],
    errors="coerce"
)


# =========================================================
# 3. 센서 열 찾기
# =========================================================

SensorCols = [
    Col
    for Col in Df.columns
    if Col.startswith("sensor_")
]


# =========================================================
# 4. IQR 방식으로 센서별 이상치 탐지
# =========================================================
#
# Q1 = 하위 25% 지점
# Q3 = 상위 75% 지점
#
# IQR = Q3 - Q1
#
# 정상 범위
# 하한 = Q1 - 1.5 * IQR
# 상한 = Q3 + 1.5 * IQR
#
# 정상 범위를 벗어난 값은 이상치 후보로 판단
# =========================================================

OutlierReport = []

for Col in SensorCols:

    # 1사분위수
    Q1 = Df[Col].quantile(0.25)

    # 3사분위수
    Q3 = Df[Col].quantile(0.75)

    # IQR 계산
    IQR = Q3 - Q1

    # 정상 범위 계산
    LowerBound = Q1 - (1.5 * IQR)
    UpperBound = Q3 + (1.5 * IQR)

    # 이상치 조건
    OutlierMask = (
        (Df[Col] < LowerBound)
        |
        (Df[Col] > UpperBound)
    )

    # 이상치 개수
    OutlierCount = OutlierMask.sum()

    # 이상치 비율
    OutlierRatio = (
        OutlierCount / len(Df)
    ) * 100

    # 결과 저장
    OutlierReport.append({
        "sensor": Col,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "LowerBound": LowerBound,
        "UpperBound": UpperBound,
        "OutlierCount": OutlierCount,
        "OutlierRatio(%)": OutlierRatio
    })


# =========================================================
# 5. 이상치 결과 DataFrame 생성
# =========================================================

OutlierReport = pd.DataFrame(
    OutlierReport
)

# 이상치가 많은 센서 순으로 정렬
OutlierReport = (
    OutlierReport
    .sort_values(
        "OutlierCount",
        ascending=False
    )
    .reset_index(drop=True)
)


# =========================================================
# 6. 센서별 이상치 결과 출력
# =========================================================

print("=== 센서별 IQR 이상치 탐지 결과 ===")

print(
    OutlierReport
)


# =========================================================
# 7. 이상치가 존재하는 센서만 출력
# =========================================================

print("\n=== 이상치가 존재하는 센서 ===")

print(
    OutlierReport[
        OutlierReport["OutlierCount"] > 0
    ][
        [
            "sensor",
            "LowerBound",
            "UpperBound",
            "OutlierCount",
            "OutlierRatio(%)"
        ]
    ]
)


# =========================================================
# 8. 결과 CSV 저장
# =========================================================

OutlierReport.to_csv(
    "Test2_IQR_OutlierReport.csv",
    index=False
)

print(
    "\nTest2_IQR_OutlierReport.csv 저장 완료"
)
