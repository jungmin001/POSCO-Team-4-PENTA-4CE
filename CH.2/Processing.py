import os
import sys
from tkinter import filedialog
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# CSV 파일 로드 및 Df 변환하여 반환 (rolling, 보간처럼 행 순서에 의존하는 계산이 있어서 미리 정렬)
def LoadData(FileLocation):
    Df = pd.read_csv(FileLocation)
    Df = Df.sort_values(["unit_number", "time_in_cycles"]).reset_index(drop=True)

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

# 결측치 보간 (엔진별로 따로 보간 - 전체에 한 번에 하면 엔진 끝 값이 다음 엔진 첫 값과 이어져서 보간됨)
def InterpolateMissing(Df):
    ResultDf = Df.copy()
    ValueCols = ResultDf.columns.drop("unit_number")
    ResultDf[ValueCols] = ResultDf.groupby("unit_number")[ValueCols].transform(
        lambda X: X.interpolate(method="linear", limit_direction="both")
    )

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

# 결측치 처리 방법 입력 (1, 2가 아니면 다시 입력)
def AskMissingMethod():
    while True:
        Answer = input("결측치 처리 방법을 선택하세요 (1: 삭제, 2: 보간): ").strip()
        if Answer in ("1", "2"):
            return int(Answer)
        print("1 또는 2를 입력하세요.")

# 중복 행 개수 반환
def CheckDuplicates(Df):
    DuplicateCount = Df.duplicated(subset=["unit_number", "time_in_cycles"]).sum()
    print("중복 행 없음" if DuplicateCount == 0 else f"중복 행 개수: {DuplicateCount}")

    return DuplicateCount

# 상수/저변동 센서 확인 (완전히 똑같은 센서 vs 거의 안 변하는 센서 구분)
def CheckConstantSensors(Df, DominantRatio=0.9):
    SensorCols = [Col for Col in Df.columns if Col.startswith("sensor_measurement_")]

    UniqueCounts = Df[SensorCols].nunique()
    ExactConstantCols = UniqueCounts[UniqueCounts == 1].index.tolist()

    # 최빈값 하나가 전체의 90% 이상을 차지하면 사실상 상수로 취급
    TopValueRatio = Df[SensorCols].apply(lambda Col: Col.value_counts(normalize=True).iloc[0])
    LowVarianceCols = TopValueRatio[TopValueRatio >= DominantRatio].index.tolist()

    print("완전히 똑같은 센서:", ExactConstantCols if ExactConstantCols else "없음")
    print("거의 안 변하는 센서(완전 동일 포함):", LowVarianceCols if LowVarianceCols else "없음")

    return ExactConstantCols, LowVarianceCols

# 엔진별 초반 1/3 구간(정상 상태로 간주) 값만 모아서 반환
def CollectEarlyValues(Df, Col):
    EarlyValues = []

    for _, Group in Df.groupby("unit_number"):
        Group = Group.sort_values("time_in_cycles")
        SplitPoint = len(Group) // 3
        EarlyValues.append(Group[Col].iloc[:SplitPoint])

    return pd.concat(EarlyValues)

# IQR 방식 이상치 탐지 (전체 구간이 아니라 초반 정상 구간 기준으로 범위를 잡음 -
def CheckOutliersIQR(Df):
    SensorCols = [Col for Col in Df.columns if Col.startswith("sensor_measurement_")]
    Report = []
    OutlierMasks = {}

    for Col in SensorCols:
        EarlyValues = CollectEarlyValues(Df, Col)

        Q1 = EarlyValues.quantile(0.25)
        Q3 = EarlyValues.quantile(0.75)
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
        # 기준은 편차 자체의 표준편차 (절댓값의 표준편차를 쓰면 기준이 약 1.8σ로 낮아져서 일반 노이즈까지 잡힘)
        Residual = Df[Col] - RollingMedian
        OutlierMask = Residual.abs() > (Threshold * Residual.std())

        Report.append({"sensor": Col, "outlier_count": OutlierMask.sum()})
        OutlierMasks[Col] = OutlierMask

    ReportDf = pd.DataFrame(Report).sort_values("outlier_count", ascending=False)

    return ReportDf, OutlierMasks

# MAD 방식 이상치 탐지 (IQR과 마찬가지로 초반 정상 구간 기준)
def CheckOutliersMAD(Df, Threshold=3.5):
    SensorCols = [Col for Col in Df.columns if Col.startswith("sensor_measurement_")]
    Report = []
    OutlierMasks = {}

    for Col in SensorCols:
        EarlyValues = CollectEarlyValues(Df, Col)

        Median = EarlyValues.median()
        Mad = (EarlyValues - Median).abs().median()
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

    ResultDf = InterpolateMissing(ResultDf)
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

    # 파일 선택 취소 시 종료 (main.py에서도 다음 단계로 안 넘어감)
    if not SelectedFile:
        sys.exit("파일을 선택하지 않아 종료합니다.")

    Df = LoadData(SelectedFile)

    # 결측치 확인 후 처리 방법 선택
    MissingCount = CheckMissing(Df)
    if len(MissingCount) > 0:
        Method = AskMissingMethod()
        Df = HandleMissing(Df, Method)

    CheckDuplicates(Df)

    ExactConstantCols, LowVarianceCols = CheckConstantSensors(Df)
    Df = Df.drop(columns=LowVarianceCols)

    # IQR 방식
    IqrReport, IqrMasks = CheckOutliersIQR(Df)
    print(IqrReport)

    # Rolling 방식
    RollingReport, RollingMasks = CheckOutliersRollingMedian(Df)
    print(RollingReport)

    # MAD 방식
    MadReport, MadMasks = CheckOutliersMAD(Df)
    print(MadReport)

    # 이상치 처리는 Rolling Median 기준으로, 할지 말지는 직접 선택 (발견된 게 있을 때만 물어봄)
    # IQR/MAD는 초반 정상 구간 기준이라 수명 후반의 열화 신호까지 이상치로 잡음 (지우면 RUL 예측 신호가 사라짐)
    if RollingReport["outlier_count"].sum() > 0:
        Df = TreatOutliers(Df, RollingMasks)
    else:
        print("이상치가 없어 처리를 건너뜁니다.")

    BaseDir = os.path.dirname(os.path.abspath(__file__))
    CleanedPath = os.path.join(BaseDir, "Data", "Processed", "cleaned_data.csv")
    os.makedirs(os.path.dirname(CleanedPath), exist_ok=True)
    SaveCleaned(Df, CleanedPath)


if __name__ == "__main__":
    Main()
