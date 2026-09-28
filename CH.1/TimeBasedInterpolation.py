import pandas as pd


# =========================================================
# 0. 원본 데이터 불러오기
# =========================================================
# 원본 데이터는 보존하고 DfClean에서 전처리를 진행한다.

Df = pd.read_csv("data/sensor.csv")

DfRaw = Df.copy()
DfClean = Df.copy()


# =========================================================
# 1. 불필요한 열 제거
# =========================================================

if "Unnamed: 0" in DfClean.columns:
    DfClean = DfClean.drop(columns=["Unnamed: 0"])


# =========================================================
# 2. timestamp 자료형 변환
# =========================================================
# timestamp를 실제 시간 자료형으로 변환
# 변환할 수 없는 값은 NaT로 변경

DfClean["timestamp"] = pd.to_datetime(
    DfClean["timestamp"],
    errors="coerce"
)

# timestamp가 없는 데이터는 시계열 분석이 불가능하므로 제거
DfClean = DfClean.dropna(
    subset=["timestamp"]
)


# =========================================================
# 3. 시간순 정렬
# =========================================================
# 과거 → 현재 순서로 데이터를 정렬

DfClean = (
    DfClean
    .sort_values("timestamp")
    .reset_index(drop=True)
)


# =========================================================
# 4. 완전히 동일한 중복 행 제거
# =========================================================

DfClean = (
    DfClean
    .drop_duplicates()
    .reset_index(drop=True)
)


# =========================================================
# 5. 센서 열 찾기
# =========================================================

SensorCols = [
    Col
    for Col in DfClean.columns
    if Col.startswith("sensor_")
]


# =========================================================
# 6. 센서값 숫자형 변환
# =========================================================
# 숫자로 변환할 수 없는 값은 NaN으로 처리

for Col in SensorCols:

    DfClean[Col] = pd.to_numeric(
        DfClean[Col],
        errors="coerce"
    )


# =========================================================
# 7. 결측률 30% 이상 센서 제거
# =========================================================
# 전체 데이터 중 결측값이 30% 이상인 센서는
# 신뢰도가 낮다고 판단하여 센서 열 자체를 제거

MissingRate = (
    DfClean[SensorCols]
    .isna()
    .mean()
    * 100
)

HighMissingCols = (
    MissingRate[
        MissingRate >= 30
    ]
    .index
    .tolist()
)

print("\n=== 결측률 30% 이상 제거 센서 ===")
print(HighMissingCols)


DfClean = DfClean.drop(
    columns=HighMissingCols
)


# 제거 후 센서 목록 다시 설정
SensorCols = [
    Col
    for Col in DfClean.columns
    if Col.startswith("sensor_")
]


# =========================================================
# 8. machine_status 라벨 정리
# =========================================================
# normal / Normal / NORMAL 등의 값을
# 모두 동일한 형태로 통일

if "machine_status" in DfClean.columns:

    DfClean["machine_status"] = (
        DfClean["machine_status"]
        .astype("string")
        .str.strip()
        .str.upper()
    )


# =========================================================
# 9. 결측치 보간 전 상태 확인
# =========================================================

BeforeMissing = (
    DfClean[SensorCols]
    .isna()
    .sum()
)

print("\n=== 보간 전 센서별 결측치 ===")

print(
    BeforeMissing[
        BeforeMissing > 0
    ]
)

print(
    "\n보간 전 전체 결측치:",
    BeforeMissing.sum()
)


# =========================================================
# 10. timestamp를 index로 설정
# =========================================================
# 실제 시간 간격을 이용해 결측값을 추정하기 위해
# timestamp를 index로 설정

DfClean = DfClean.set_index(
    "timestamp"
)


# =========================================================
# 11. 앞뒤 센서 흐름을 이용한 결측치 보간
# =========================================================
#
# 예)
#
# 31
# NaN
# 33
#
# ↓
#
# 31
# 32
# 33
#
# method="time"
# → 실제 시간 간격을 고려하여 추정
#
# limit_area="inside"
# → 앞뒤에 실제 값이 존재하는 결측치만 보간

DfClean[SensorCols] = (
    DfClean[SensorCols]
    .interpolate(
        method="time",
        limit_area="inside"
    )
)


# =========================================================
# 12. timestamp를 다시 일반 열로 복원
# =========================================================

DfClean = DfClean.reset_index()


# =========================================================
# 13. 결측치 보간 후 확인
# =========================================================

AfterMissing = (
    DfClean[SensorCols]
    .isna()
    .sum()
)

print("\n=== 보간 후 센서별 결측치 ===")

print(
    AfterMissing[
        AfterMissing > 0
    ]
)

print(
    "\n보간 후 남아있는 전체 결측치:",
    AfterMissing.sum()
)


# =========================================================
# 14. 최종 시간순 정렬
# =========================================================

DfClean = (
    DfClean
    .sort_values("timestamp")
    .reset_index(drop=True)
)


# =========================================================
# 15. 최종 중복 확인
# =========================================================

DuplicateCount = (
    DfClean
    .duplicated()
    .sum()
)

print(
    "\n남아있는 완전 중복 행:",
    DuplicateCount
)


# =========================================================
# 16. IQR 방식 이상치 확인
# =========================================================
# 이상치는 제거하지 않고 후보만 확인한다.
#
# Q1 = 하위 25%
# Q3 = 상위 75%
# IQR = Q3 - Q1
#
# 정상 범위:
# Q1 - 1.5 * IQR ~ Q3 + 1.5 * IQR

OutlierReport = []


for Col in SensorCols:

    Q1 = DfClean[Col].quantile(0.25)
    Q3 = DfClean[Col].quantile(0.75)

    IQR = Q3 - Q1

    LowerBound = Q1 - (1.5 * IQR)
    UpperBound = Q3 + (1.5 * IQR)

    OutlierMask = (
        (DfClean[Col] < LowerBound)
        |
        (DfClean[Col] > UpperBound)
    )

    OutlierCount = OutlierMask.sum()

    OutlierRatio = (
        OutlierCount
        / len(DfClean)
        * 100
    )

    OutlierReport.append({
        "Sensor": Col,
        "LowerBound": LowerBound,
        "UpperBound": UpperBound,
        "OutlierCount": OutlierCount,
        "OutlierRatio(%)": OutlierRatio
    })


OutlierReport = pd.DataFrame(
    OutlierReport
)

OutlierReport = (
    OutlierReport
    .sort_values(
        "OutlierCount",
        ascending=False
    )
    .reset_index(drop=True)
)


# 이상치가 존재하는 센서만 화면에 출력
print("\n=== IQR 이상치가 존재하는 센서 ===")

print(
    OutlierReport[
        OutlierReport["OutlierCount"] > 0
    ]
)


# =========================================================
# 17. 최종 데이터 상태 확인
# =========================================================

print("\n================================")
print("Test2 최종 전처리 완료")
print("================================")

print(
    "원본 데이터 크기:",
    DfRaw.shape
)

print(
    "최종 데이터 크기:",
    DfClean.shape
)

print(
    "\n결측률 30% 이상으로 제거된 센서:"
)

print(
    HighMissingCols
)

print(
    "\n최종 센서 개수:",
    len(SensorCols)
)

print(
    "\n최종 센서 결측치:",
    DfClean[SensorCols]
    .isna()
    .sum()
    .sum()
)

print(
    "\n최종 중복 행:",
    DuplicateCount
)


# =========================================================
# 18. machine_status 최종 확인
# =========================================================

if "machine_status" in DfClean.columns:

    print(
        "\n=== machine_status 분포 ==="
    )

    print(
        DfClean["machine_status"]
        .value_counts(
            dropna=False
        )
    )


# =========================================================
# 19. 센서 기초 통계 확인
# =========================================================

print(
    "\n=== 센서 기초 통계 ==="
)

print(
    DfClean[SensorCols]
    .describe()
)


# =========================================================
# 20. 최종 전처리 데이터 CSV 저장
# =========================================================
# 별도의 결측치/이상치 보고서 파일은 저장하지 않고
# 최종 전처리 데이터 하나만 저장한다.

DfClean.to_csv(
    "Test2_machine_data.csv",
    index=False
)

# print(
#     "\nTest2_machine_data.csv 저장 완료"
# )