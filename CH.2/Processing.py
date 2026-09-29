import os
from tkinter import filedialog
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# CSV 파일 로드 및 Df 변환하여 반환
def LoadData(FileLocation):
    Df = pd.read_csv(FileLocation)

    return Df

# 결측치 개수 반환
def CheckMissing(Df):
    MissingCount = Df.isna().sum()
    MissingCount = MissingCount[MissingCount > 0]
    print("결측치 없음" if len(MissingCount) == 0 else f"결측치 개수: {MissingCount}")

    return MissingCount

# 결측치 삭제
def DropMissing(Df):
    ResultDf = Df.dropna()

    return ResultDf

# 결측치 보간
def InterpolateMissing(Df):
    ResultDf = Df.interpolate(method="linear", limit_direction="both")

    return ResultDf

# 결측치 처리 방법 선택
def HandleMissing(Df, Method):
    if Method == 1:
        ResultDf = DropMissing(Df)
    elif Method == 2:
        ResultDf = InterpolateMissing(Df)
    else:
        print("잘못된 입력입니다. 원본을 그대로 반환합니다.")
        ResultDf = Df

    return ResultDf

# 중복 행 개수 반환
def CheckDuplicates(Df):
    DuplicateCount = Df.duplicated(subset=["unit_number", "time_in_cycles"]).sum()
    print("중복 행 없음" if DuplicateCount == 0 else f"중복 행 개수: {DuplicateCount}")

    return DuplicateCount

# 상수/저변동 센서 확인 (완전히 똑같은 센서 vs 거의 안 변하는 센서 구분)
def CheckConstantSensors(Df):
    SensorCols = [Col for Col in Df.columns if Col.startswith("sensor_measurement_")]

    UniqueCounts = Df[SensorCols].nunique()
    ExactConstantCols = UniqueCounts[UniqueCounts == 1].index.tolist()

    StdValues = Df[SensorCols].std()
    LowVarianceCols = StdValues[StdValues < 1e-5].index.tolist()

    print("완전히 똑같은 센서:", ExactConstantCols if ExactConstantCols else "없음")
    print("거의 안 변하는 센서(완전 동일 포함):", LowVarianceCols if LowVarianceCols else "없음")

    return ExactConstantCols, LowVarianceCols

# IQR 방식 이상치 탐지
def CheckOutliersIQR(Df):
    SensorCols = [Col for Col in Df.columns if Col.startswith("sensor_measurement_")]
    Report = []
    OutlierMasks = {}

    for Col in SensorCols:
        Q1 = Df[Col].quantile(0.25)
        Q3 = Df[Col].quantile(0.75)
        IQR = Q3 - Q1
        Lower = Q1 - 1.5 * IQR
        Upper = Q3 + 1.5 * IQR
        OutlierMask = (Df[Col] < Lower) | (Df[Col] > Upper)

        Report.append({"sensor": Col, "outlier_count": OutlierMask.sum()})
        OutlierMasks[Col] = OutlierMask

    ReportDf = pd.DataFrame(Report).sort_values("outlier_count", ascending=False)

    return ReportDf, OutlierMasks

# Rolling Median 방식 이상치 탐지 (엔진별 시간 흐름 기준)
def CheckOutliersRollingMedian(Df, Window=5, Threshold=3):
    SensorCols = [Col for Col in Df.columns if Col.startswith("sensor_measurement_")]
    Report = []
    OutlierMasks = {}

    for Col in SensorCols:
        RollingMedian = Df.groupby("unit_number")[Col].transform(
            lambda X: X.rolling(Window, center=True, min_periods=1).median()
        )
        Deviation = (Df[Col] - RollingMedian).abs()
        OutlierMask = Deviation > (Threshold * Deviation.std())

        Report.append({"sensor": Col, "outlier_count": OutlierMask.sum()})
        OutlierMasks[Col] = OutlierMask

    ReportDf = pd.DataFrame(Report).sort_values("outlier_count", ascending=False)

    return ReportDf, OutlierMasks

# MAD 방식 이상치 탐지
def CheckOutliersMAD(Df, Threshold=3.5):
    SensorCols = [Col for Col in Df.columns if Col.startswith("sensor_measurement_")]
    Report = []
    OutlierMasks = {}

    for Col in SensorCols:
        Median = Df[Col].median()
        Mad = (Df[Col] - Median).abs().median()
        ModifiedZScore = pd.Series(0, index=Df.index) if Mad == 0 else 0.6745 * (Df[Col] - Median) / Mad
        OutlierMask = ModifiedZScore.abs() > Threshold

        Report.append({"sensor": Col, "outlier_count": OutlierMask.sum()})
        OutlierMasks[Col] = OutlierMask

    ReportDf = pd.DataFrame(Report).sort_values("outlier_count", ascending=False)

    return ReportDf, OutlierMasks


# 이상치 처리 여부 선택
def TreatOutliers(Df, OutlierMasks):
    Answer = input("이상치를 결측치로 바꾸고 보간하시겠습니까? (y/n): ").strip().lower()

    if Answer != "y":
        print("이상치를 그대로 유지합니다.")
        return Df

    ResultDf = Df.copy()
    for Col, Mask in OutlierMasks.items():
        ResultDf.loc[Mask, Col] = None

    ResultDf = ResultDf.interpolate(method="linear", limit_direction="both")
    print("이상치를 결측치로 바꾸고 보간했습니다.")

    return ResultDf

# CSV 저장
def SaveCleaned(Df, SavePath):
    Df.to_csv(SavePath, index=False)


# 자동 실행 함수
def Main():
    # GUI 라이브러리 초기화
    Root = tk.Tk()
    Root.withdraw()

    # 파일탐색기 실행 (CSV 파일 외 필터링)
    SelectedFile = filedialog.askopenfilename(initialdir=os.getcwd(), title="CSV 파일을 선택하세요",
                                              filetypes=(("CSV 파일", "*.csv"),))

    Df = LoadData(SelectedFile)

    # 결측치 확인 후 처리 방법 선택
    CheckMissing(Df)
    Method = int(input("결측치 처리 방법을 선택하세요 (1: 삭제, 2: 보간): "))
    Df = HandleMissing(Df, Method)

    CheckDuplicates(Df)

    ExactConstantCols, LowVarianceCols = CheckConstantSensors(Df)
    Df = Df.drop(columns=LowVarianceCols)

    IqrReport, IqrMasks = CheckOutliersIQR(Df)
    print(IqrReport)

    RollingReport, RollingMasks = CheckOutliersRollingMedian(Df)
    print(RollingReport)

    MadReport, MadMasks = CheckOutliersMAD(Df)
    print(MadReport)

    # 이상치 처리는 IQR 기준으로, 할지 말지는 직접 선택
    Df = TreatOutliers(Df, IqrMasks)

    SavePath = SelectedFile.replace(".csv", "_cleaned.csv")
    SaveCleaned(Df, SavePath)


Main()
