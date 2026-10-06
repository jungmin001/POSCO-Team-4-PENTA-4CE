import os
import sys
from tkinter import filedialog
import pandas as pd
import tkinter as tk

SensorCols = ["WD-SPD", "WD-LOD", "OSC-FRQ", "OSC-STK", "MLD-LVL"]
KeyCols = ["EQP_CD", "SLAB_NO", "MEAS_DT"]


# CSV 파일 로드 및 Df 변환하여 반환 (rolling, 보간처럼 행 순서에 의존하는 계산이 있어서 미리 정렬)
def LoadData(FileLocation):
    Df = pd.read_csv(FileLocation, encoding="utf-8-sig")
    Df["MEAS_DT"] = pd.to_datetime(Df["MEAS_DT"])
    Df = Df.sort_values(KeyCols).reset_index(drop=True)

    return Df

# 현재 남아 있는 센서 컬럼 목록 (저변동 센서를 지운 뒤에도 쓸 수 있게)
def GetSensorCols(Df):
    return [Col for Col in SensorCols if Col in Df.columns]

# 결측치 개수 반환
def CheckMissing(Df):
    MissingCount = Df.isna().sum()
    MissingCount = MissingCount[MissingCount > 0]
    print("결측치 없음" if len(MissingCount) == 0 else f"결측치 개수: {MissingCount}")

    return MissingCount

# 결측치 보간 (슬라브별로 따로 - 전체에 한 번에 하면 앞 슬라브 끝 값이 다음 슬라브 첫 값과 이어져서 보간됨)
def InterpolateMissing(Df):
    ResultDf = Df.copy()
    Cols = GetSensorCols(ResultDf)
    ResultDf[Cols] = ResultDf.groupby("SLAB_NO")[Cols].transform(
        lambda X: X.interpolate(method="linear", limit_direction="both")
    )

    return ResultDf

# 중복 행 개수 반환 (완전히 같은 행 / 같은 설비·슬라브·시각인데 값이 다른 행)
def CheckDuplicates(Df):
    ExactCount = Df.duplicated().sum()
    SameTimeCount = Df.drop_duplicates().duplicated(subset=KeyCols).sum()
    print("완전히 같은 행:", ExactCount)
    print("같은 시각에 값이 다른 행:", SameTimeCount)

    return ExactCount, SameTimeCount

# 중복 정리 (완전히 같은 행은 삭제, 같은 시각에 여러 번 측정된 값은 평균으로 한 행으로 합침)
def MergeDuplicates(Df):
    ResultDf = Df.drop_duplicates()
    ResultDf = ResultDf.groupby(KeyCols, as_index=False)[SensorCols].mean()

    return ResultDf

# 상수/저변동 센서 확인 (완전히 똑같은 센서 vs 거의 안 변하는 센서 구분)
def CheckConstantSensors(Df, DominantRatio=0.9):
    UniqueCounts = Df[SensorCols].nunique()
    ExactConstantCols = UniqueCounts[UniqueCounts == 1].index.tolist()

    # 최빈값 하나가 전체의 90% 이상을 차지하면 사실상 상수로 취급
    TopValueRatio = Df[SensorCols].apply(lambda Col: Col.value_counts(normalize=True).iloc[0])
    LowVarianceCols = TopValueRatio[TopValueRatio >= DominantRatio].index.tolist()

    print("완전히 똑같은 센서:", ExactConstantCols if ExactConstantCols else "없음")
    print("거의 안 변하는 센서(완전 동일 포함):", LowVarianceCols if LowVarianceCols else "없음")

    return ExactConstantCols, LowVarianceCols

# Rolling Median 방식 이상치 탐지 (슬라브별 시간 흐름 기준, 주변 11개 값의 중앙값보다 20% 넘게 튀는 값)
# CC1/CC2는 정상 범위 자체가 달라서 전체 기준(IQR 등)으로 잡으면 정상값까지 이상치로 잡힘
# 실제 데이터에서는 OSC-STK가 6,000 근처에서 28,000~50,000으로 한 번씩 튀는 센서 오류만 잡힘
def CheckOutliersRollingMedian(Df, Window=11, Ratio=0.2):
    Report = []
    OutlierMasks = {}

    for Col in GetSensorCols(Df):
        RollingMedian = Df.groupby("SLAB_NO")[Col].transform(
            lambda X: X.rolling(Window, center=True, min_periods=1).median()
        )
        OutlierMask = (Df[Col] / RollingMedian - 1).abs() > Ratio

        Report.append({"sensor": Col, "outlier_count": OutlierMask.sum()})
        OutlierMasks[Col] = OutlierMask

    ReportDf = pd.DataFrame(Report).sort_values("outlier_count", ascending=False)

    return ReportDf, OutlierMasks

# 이상치를 결측치로 바꾼 뒤 슬라브별 보간
def ReplaceOutliers(Df, OutlierMasks):
    ResultDf = Df.copy()
    for Col, Mask in OutlierMasks.items():
        ResultDf.loc[Mask, Col] = None

    ResultDf = InterpolateMissing(ResultDf)

    return ResultDf

# CSV 저장
def SaveCleaned(Df, SavePath):
    Df.to_csv(SavePath, index=False)


# 자동 실행 함수
def Main():
    # GUI 라이브러리 초기화
    Root = tk.Tk()
    Root.withdraw()

    # 파일탐색기 실행 (CSV 파일 외 필터링) - T-CC-INS01_주편.csv 선택
    BaseDir = os.path.dirname(os.path.abspath(__file__))
    SelectedFile = filedialog.askopenfilename(initialdir=os.path.join(BaseDir, "Data", "Raw"), title="센서 CSV 파일을 선택하세요",
                                              filetypes=(("CSV 파일", "*.csv"),))

    # 파일 선택 취소 시 종료 (Main.py에서도 다음 단계로 안 넘어감)
    if not SelectedFile:
        sys.exit("파일을 선택하지 않아 종료합니다.")

    Df = LoadData(SelectedFile)

    # 결측치 확인 후 있으면 보간
    MissingCount = CheckMissing(Df)
    if len(MissingCount) > 0:
        Df = InterpolateMissing(Df)

    # 중복 확인 후 정리 (같은 시각 값은 평균)
    CheckDuplicates(Df)
    Df = MergeDuplicates(Df)
    print("중복 정리 후 행 수:", len(Df))

    ExactConstantCols, LowVarianceCols = CheckConstantSensors(Df)
    Df = Df.drop(columns=LowVarianceCols)

    # 이상치 탐지 후 결측치로 바꿔서 보간
    RollingReport, RollingMasks = CheckOutliersRollingMedian(Df)
    print(RollingReport)
    Df = ReplaceOutliers(Df, RollingMasks)

    CleanedPath = os.path.join(BaseDir, "Data", "Processed", "cleaned_data.csv")
    os.makedirs(os.path.dirname(CleanedPath), exist_ok=True)
    SaveCleaned(Df, CleanedPath)
    print("저장 완료:", CleanedPath)


if __name__ == "__main__":
    Main()
