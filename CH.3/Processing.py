import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import pandas as pd

SensorCols = ["WD-SPD", "WD-LOD", "OSC-FRQ", "OSC-STK", "MLD-LVL"]
KeyCols = ["EQP_CD", "SLAB_NO", "MEAS_DT"]
GroupCols = ["EQP_CD", "SLAB_NO"]

DataCols = ["MEAS_DT", "EQP_CD", "SLAB_NO"] + SensorCols
OutputCols = DataCols + ["IS_OUTLIER"]


# 저장 기준이 되는 CH.3 폴더 찾기
def GetChapterDir():
    ScriptDir = Path(__file__).resolve().parent

    for Folder in [ScriptDir, *ScriptDir.parents]:
        if Folder.name == "CH.3":
            return Folder

    return ScriptDir / "CH.3"


# CSV 로드 및 정렬
def LoadData(FileLocation):
    Df = pd.read_csv(FileLocation, encoding="utf-8-sig")

    MissingCols = [Col for Col in DataCols if Col not in Df.columns]
    if MissingCols:
        raise ValueError(f"필수 컬럼이 없습니다: {MissingCols}")

    Df = Df[DataCols].copy()

    # 원본 CSV에서의 행 순서 보관
    Df["_ROW_ORDER"] = range(len(Df))

    Df["MEAS_DT"] = pd.to_datetime(Df["MEAS_DT"], errors="coerce")

    for Col in ["EQP_CD", "SLAB_NO"]:
        Df[Col] = Df[Col].astype("string").str.strip().replace("", pd.NA)

    for Col in SensorCols:
        Df[Col] = pd.to_numeric(Df[Col], errors="coerce")

    Df[SensorCols] = Df[SensorCols].replace([float("inf"), float("-inf")], float("nan"))

    return Df.sort_values(KeyCols + ["_ROW_ORDER"]).reset_index(drop=True)


# 결측치가 하나라도 있는 행 전체 삭제
def DeleteMissingRows(Df):
    ResultDf = Df.dropna(subset=DataCols).copy().reset_index(drop=True)

    print("결측치/유효하지 않은 값으로 삭제한 행:", len(Df) - len(ResultDf))

    return ResultDf


# 이상치 표시: 행 삭제 및 값 변경 없음
def MarkOutliers(Df, IQRMultiplier=3.0, ChangeRatio=0.05, Window=11):
    ResultDf = Df.copy()
    OutlierMasks = {}
    Report = []

    EqGroups = ResultDf.groupby("EQP_CD", sort=False).groups

    for Col in SensorCols:
        # 설비별 전체 데이터로 IQR 범위 계산
        Grouped = ResultDf.groupby("EQP_CD", sort=False)[Col]

        Q1 = Grouped.transform("quantile", q=0.25)
        Q3 = Grouped.transform("quantile", q=0.75)
        IQR = Q3 - Q1

        Lower = Q1 - IQRMultiplier * IQR
        Upper = Q3 + IQRMultiplier * IQR

        IQRMask = (ResultDf[Col] < Lower) | (ResultDf[Col] > Upper)

        # 같은 설비·슬래브 내 주변 11행 중앙값
        # 중복 시각을 합치기 전이며 현재 행도 포함
        RollingMedian = ResultDf.groupby(GroupCols, sort=False)[Col].transform(
            lambda X: X.rolling(Window, center=True, min_periods=1).median()
        )

        Difference = (ResultDf[Col] - RollingMedian).abs()

        # 중앙값 대비 차이가 5% 이상인지 확인
        # 중앙값이 0일 때는 0이 아닌 값만 변화 조건 충족
        ChangeMask = (Difference > 0) & (
            Difference >= RollingMedian.abs() * ChangeRatio
        )

        # IQR 범위 이탈 AND 변화율 5% 이상
        OutlierMask = IQRMask & ChangeMask

        OutlierMasks[Col] = OutlierMask
        ResultDf[f"{Col}_OUTLIER"] = OutlierMask

        for Eq, Indexes in EqGroups.items():
            FirstIndex = Indexes[0]

            Report.append(
                {
                    "EQP_CD": Eq,
                    "sensor": Col,
                    "IQR_lower": Lower.loc[FirstIndex],
                    "IQR_upper": Upper.loc[FirstIndex],
                    "outlier_count": int(OutlierMask.loc[Indexes].sum()),
                }
            )

    # 센서 하나라도 이상치이면 해당 행을 이상치로 표시
    ResultDf["IS_OUTLIER"] = pd.DataFrame(OutlierMasks, index=ResultDf.index).any(
        axis=1
    )

    ReportDf = pd.DataFrame(
        Report, columns=["EQP_CD", "sensor", "IQR_lower", "IQR_upper", "outlier_count"]
    ).sort_values(["EQP_CD", "outlier_count"], ascending=[True, False])

    return ResultDf, ReportDf


# 같은 시각의 대표값 선택
def MergeDuplicates(Df):
    UnknownEqp = set(Df["EQP_CD"].unique()) - {"CC1", "CC2"}
    if UnknownEqp:
        raise ValueError(f"처리 규칙이 없는 설비입니다: {UnknownEqp}")

    GroupSizes = Df.groupby(KeyCols).size()
    if (GroupSizes > 2).any():
        raise ValueError(
            "같은 설비·슬래브·시각에 3행 이상 남아 있습니다. " "처리 규칙을 확인하세요."
        )

    Df = Df.sort_values(KeyCols + ["_ROW_ORDER"])

    AllOutliers = Df.groupby(KeyCols, sort=False)["IS_OUTLIER"].transform("all")

    # 정상 행이 있으면 정상 행만 대표값 후보로 선택
    # 전부 이상치이면 이상치 행도 후보로 유지
    # 입력 데이터의 행은 삭제하지 않음
    Candidates = Df.loc[~Df["IS_OUTLIER"] | AllOutliers].copy()

    # CC1: 첫 번째 후보 행 사용
    CC1 = Candidates.loc[Candidates["EQP_CD"].eq("CC1")].drop_duplicates(
        subset=KeyCols, keep="first"
    )[OutputCols]

    # CC2: 후보 행의 센서별 평균
    # 한 행만 후보이면 그 값 그대로 사용
    # 전부 이상치이면 평균을 내도 이상치 표시 유지
    Aggregations = {Col: "mean" for Col in SensorCols}
    Aggregations["IS_OUTLIER"] = "any"

    CC2 = (
        Candidates.loc[Candidates["EQP_CD"].eq("CC2")]
        .groupby(KeyCols, as_index=False, sort=False)
        .agg(Aggregations)
    )

    ResultDf = pd.concat([CC1, CC2[OutputCols]], ignore_index=True)

    return ResultDf.sort_values(KeyCols).reset_index(drop=True)


# 최종 CSV 저장
def SaveCleaned(Df, SavePath):
    SavePath = Path(SavePath)
    SavePath.parent.mkdir(parents=True, exist_ok=True)

    Df.to_csv(SavePath, index=False, encoding="utf-8-sig")


def Main():
    ChapterDir = GetChapterDir()

    Root = tk.Tk()
    Root.withdraw()

    try:
        SelectedFile = filedialog.askopenfilename(
            initialdir=str(ChapterDir / "Data" / "Raw"),
            title="센서 CSV 파일을 선택하세요",
            filetypes=(("CSV 파일", "*.csv"),),
        )
    finally:
        Root.destroy()

    if not SelectedFile:
        sys.exit("파일을 선택하지 않아 종료합니다.")

    Df = LoadData(SelectedFile)
    print("원본 행 수:", len(Df))

    # 1. 결측치 행 삭제
    Df = DeleteMissingRows(Df)

    # 2. 이상치 표시
    MarkedDf, OutlierReport = MarkOutliers(Df)

    print("\n설비·센서별 이상치 판정 결과:")
    print(OutlierReport.to_string(index=False))

    print("\n대표값 선택 전 이상치 행 수:", int(MarkedDf["IS_OUTLIER"].sum()))

    # 3. 설비별 규칙으로 대표값 선택
    CleanedDf = MergeDuplicates(MarkedDf)

    print("\n최종 행 수:", len(CleanedDf))
    print("설비별 최종 행 수:")
    print(CleanedDf.groupby("EQP_CD").size().to_string())

    print("\n최종 데이터에 유지된 이상치 행 수:")
    print(CleanedDf.groupby("EQP_CD")["IS_OUTLIER"].sum().to_string())

    # CH.3/Data/Processed/cleaned_data.csv
    CleanedPath = ChapterDir / "Data" / "Processed" / "cleaned_data.csv"

    SaveCleaned(CleanedDf, CleanedPath)
    print("\n저장 완료:", CleanedPath)

    # MarkedDf는 대표값 선택 전 데이터이며 별도로 저장하지 않음
    return CleanedDf, MarkedDf


if __name__ == "__main__":
    Main()
