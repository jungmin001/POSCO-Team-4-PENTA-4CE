import pandas as pd


# =========================================================
# 0. 원본 데이터 불러오기
# =========================================================

Df = pd.read_csv("data/sensor.csv")

# 원본 데이터 보존
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

DfClean["timestamp"] = pd.to_datetime(
    DfClean["timestamp"],
    errors="coerce"
)

# timestamp 자체가 없는 행 제거
DfClean = DfClean.dropna(
    subset=["timestamp"]
)


# =========================================================
# 3. 시간순 정렬
# =========================================================

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
# 6. 센서값 숫자형으로 변환
# =========================================================

for Col in SensorCols:
    DfClean[Col] = pd.to_numeric(
        DfClean[Col],
        errors="coerce"
    )


# =========================================================
# 7. 결측률 30% 이상 센서 제거
# =========================================================
# 전체 데이터 중 30% 이상이 비어있는 센서는
# 데이터 신뢰도가 낮다고 판단하여 제거

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

print("=== 결측률 30% 이상 제거 센서 ===")
print(HighMissingCols)


DfClean = DfClean.drop(
    columns=HighMissingCols
)


# 제거 이후 센서 목록 다시 갱신
SensorCols = [
    Col
    for Col in DfClean.columns
    if Col.startswith("sensor_")
]


# =========================================================
# 8. machine_status 라벨 정리
# =========================================================

if "machine_status" in DfClean.columns:

    DfClean["machine_status"] = (
        DfClean["machine_status"]
        .astype("string")
        .str.strip()
        .str.upper()
    )


# =========================================================
# 9. 보간 전 결측치 상태 확인
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
# 시간 간격을 고려하여 앞뒤 센서값으로
# 결측치를 추정하기 위해 timestamp를 index로 사용

DfClean = DfClean.set_index(
    "timestamp"
)


# =========================================================
# 11. 앞뒤 센서 흐름을 이용해 결측값 추정
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
# → 실제 timestamp 시간 간격까지 고려
#
# limit_area="inside"
# → 앞뒤에 실제 값이 모두 존재하는 결측만 추정

DfClean[SensorCols] = (
    DfClean[SensorCols]
    .interpolate(
        method="time",
        limit_area="inside"
    )
)


# =========================================================
# 12. timestamp 다시 일반 열로 복원
# =========================================================

DfClean = (
    DfClean
    .reset_index()
)


# =========================================================
# 13. 보간 후 결측치 확인
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
# 14. 시간순 최종 정렬
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
# 16. 최종 데이터 상태 확인
# =========================================================

print("\n================================")
print("Test2 전처리 완료")
print("================================")

print(
    "원본 데이터 크기:",
    DfRaw.shape
)

print(
    "Test2 데이터 크기:",
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
    "\n최종 센서 결측치 수:",
    DfClean[SensorCols]
    .isna()
    .sum()
    .sum()
)


# =========================================================
# 17. machine_status 최종 확인
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
# 18. 센서 기초 통계 확인
# =========================================================

print(
    "\n=== 센서 기초 통계 ==="
)

print(
    DfClean[SensorCols]
    .describe()
)


# =========================================================
# 19. Test2 CSV 저장
# =========================================================

DfClean.to_csv(
    "Test2_machine_data.csv",
    index=False
)

print(
    "\nTest2_machine_data.csv 저장 완료"
)