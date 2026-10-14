import pandas as pd

from . import config, data

FlagCols = ["scenario", "time", "kind", "column", "value"]
MaintCols = ["scenario", "start", "end"]
FillCols = ["scenario", "time", "column", "value", "gap_rows"]


# 직전 Window개 값의 중앙값과 그 편차의 MAD (직전 값만 쓰므로 예측 시점 이후의 정보가 섞이지 않음)
# 중앙값은 Window행, MAD는 그 편차가 Window개 쌓여야 계산되므로 앞 2*Window-1행은 NaN
def HampelStats(Series, Window=None):
    Window = config.Window if Window is None else Window

    Median = Series.rolling(Window, min_periods=Window).median().shift(1)
    Mad = (Series - Median).abs().rolling(Window, min_periods=Window).median().shift(1) * 1.4826

    return Median, Mad


# 판정할 수 있는 행 (기준값과 척도가 모두 계산되는 행)
def HampelChecked(Series, Window=None):
    Median, Mad = HampelStats(Series, Window)

    return Median.notna() & Mad.notna()


# 직전 Window개 값의 중앙값에서 NSigma * 척도 이상 벗어난 값 (Hampel 필터), 판정할 수 없는 행은 이상치로 보지 않음
def HampelFlags(Series, Window=None, NSigma=None, RelFloor=None):
    NSigma = config.DetectNSigma if NSigma is None else NSigma
    RelFloor = config.RelFloor if RelFloor is None else RelFloor

    Median, Mad = HampelStats(Series, Window)

    # 거의 일정한 신호의 작은 변화가 잡히지 않도록 척도 하한을 중앙값의 RelFloor 비율로 둠
    Floor = RelFloor * Median.abs()
    Scale = Mad.where(Mad > Floor, Floor).clip(lower=1e-9)

    return ((Series - Median).abs() > NSigma * Scale) & Median.notna() & Mad.notna()


# 시나리오 하나를 열어 시각순으로 정렬한 표 (설비 ID가 설정돼 있으면 그 설비만)
def OpenScenario(ScenarioId, Source=None, Manifest=None):
    Source = config.Source if Source is None else Source
    Df = data.ReadScenario(Source, ScenarioId, Manifest=Manifest)

    if config.EqpCode is not None and config.EqpCol in Df.columns:
        Df = Df[Df[config.EqpCol] == config.EqpCode]

    Df[config.TimeCol] = pd.to_datetime(Df[config.TimeCol], utc=True)
    SortCols = [config.TimeCol] + [Col for Col in config.TieCols if Col in Df.columns]

    return Df.sort_values(SortCols, kind="stable").reset_index(drop=True)


# 시나리오 하나의 정비 구간 (설비 ID가 설정돼 있으면 그 설비만), 정비가 없으면 빈 표
def ReadMaintenance(ScenarioId, Manifest=None):
    Df = data.ReadScenario(config.MaintSource, ScenarioId, Manifest=Manifest)

    if Df.empty:
        return pd.DataFrame(columns=MaintCols)

    if config.EqpCode is not None:
        Df = Df[Df[config.MaintEqpCol] == config.EqpCode]

    return pd.DataFrame({"scenario": ScenarioId,
                         "start": pd.to_datetime(Df[config.MaintStartCol], utc=True),
                         "end": pd.to_datetime(Df[config.MaintEndCol], utc=True)}, columns=MaintCols)


# 센서 컬럼: 제외 목록에 없는 숫자 컬럼 전부
def GetSensors(Df):
    return [Col for Col in Df.select_dtypes("number").columns if Col not in config.ExcludeCols]


# 정지·결측·정비 구간 여부
# 결측: 센서 컬럼에 NaN이 있거나 품질 컬럼이 정상값이 아닌 행
def GetMasks(Df, Maint):
    Sensors = GetSensors(Df)

    IsStopped = (Df[config.RunCol] == config.StopValue) if config.RunCol in Df.columns else pd.Series(False, index=Df.index)

    IsMissing = Df[Sensors].isna().any(axis=1)
    if config.QualityCol in Df.columns:
        IsMissing = IsMissing | (Df[config.QualityCol] != config.QualityOk)

    IsMaint = pd.Series(False, index=Df.index)
    for _, Row in Maint.iterrows():
        End = Row["end"] if pd.notna(Row["end"]) else Df[config.TimeCol].max() + pd.Timedelta(seconds=1)
        IsMaint |= (Df[config.TimeCol] >= Row["start"]) & (Df[config.TimeCol] < End)

    return Sensors, IsStopped, IsMissing, IsMaint


# 짧은 결측(연속 FillMaxGap행 이하)만 바로 앞 값으로 채움 (입력 표는 바꾸지 않고 채운 복사본을 만듦)
# 더 긴 결측과 맨 앞 결측은 채우지 않음, 반환: 채운 칸 목록, 채운 뒤에도 NaN이 남은 행 수, 채운 복사본
def FillShortGaps(Df, Sensors, ScenarioId):
    Filled, StillNa, FilledDf = [], pd.Series(False, index=Df.index), Df.copy()

    for Col in Sensors:
        IsNa = Df[Col].isna()

        if not IsNa.any():
            continue

        GapLength = IsNa.groupby((~IsNa).cumsum()).transform("sum")
        Previous = Df[Col].ffill()
        CanFill = IsNa & (GapLength <= config.FillMaxGap) & Previous.notna()

        Filled.append(pd.DataFrame({"scenario": ScenarioId, "time": Df.loc[CanFill, config.TimeCol], "column": Col,
                                    "value": Previous[CanFill], "gap_rows": GapLength[CanFill]}, columns=FillCols))
        StillNa |= IsNa & ~CanFill
        FilledDf.loc[CanFill, Col] = Previous[CanFill]

    return (pd.concat(Filled, ignore_index=True) if Filled else pd.DataFrame(columns=FillCols)), int(StillNa.sum()), FilledDf


# 정지·결측·정비 구간 행은 비교에서 빼고 나머지 행에서 센서별 이상치를 찾음 (값은 바꾸지 않음)
# 반환: 판정 대상 행, 판정 가능한 행 수, 센서별 이상치 여부 (모든 행 기준, 판정 대상이 아닌 행은 False)
def FindOutliers(Df, Sensors, IsStopped, IsMissing, IsMaint):
    Valid = Df.loc[~IsStopped & ~IsMissing & ~IsMaint]
    Groups = [Col for Col in config.GroupCols if Col in Df.columns]

    # 판정할 수 있는 행 수 (센서마다 같으므로 첫 센서로 계산)
    Checked = Valid.groupby(Groups, sort=False)[Sensors[0]].transform(HampelChecked) if Groups else HampelChecked(Valid[Sensors[0]])

    Outliers = {}
    for Col in Sensors:
        if Groups:
            IsOutlier = Valid.groupby(Groups, sort=False)[Col].transform(HampelFlags)
        else:
            IsOutlier = HampelFlags(Valid[Col])

        Outliers[Col] = IsOutlier.fillna(False).astype(bool).reindex(Df.index, fill_value=False)

    return Valid, int(Checked.fillna(False).astype(bool).sum()), Outliers


# 시나리오 하나 점검
# 반환: 요약 한 줄, 결측·이상치 행 목록
def Inspect(Df, ScenarioId, Maint):
    Sensors, IsStopped, IsMissing, IsMaint = GetMasks(Df, Maint)

    FillDf, Unfilled, _ = FillShortGaps(Df, Sensors, ScenarioId)
    Valid, CheckedRows, Outliers = FindOutliers(Df, Sensors, IsStopped, IsMissing, IsMaint)

    Summary = {"scenario": ScenarioId, "rows": len(Df), "stopped": int(IsStopped.sum()), "missing": int(IsMissing.sum()),
               "maintenance": int(IsMaint.sum()), "filled_cells": len(FillDf), "unfilled_rows": Unfilled,
               "valid_rows": len(Valid), "checked_rows": CheckedRows}
    Flags = [pd.DataFrame({"scenario": ScenarioId, "time": Df.loc[IsMissing, config.TimeCol], "kind": "missing",
                           "column": "", "value": float("nan")}, columns=FlagCols)]

    AnyOutlier = pd.Series(False, index=Df.index)

    for Col in Sensors:
        AnyOutlier |= Outliers[Col]

        Summary[f"outlier_{Col}"] = int(Outliers[Col].sum())
        Summary[f"min_{Col}"] = Valid[Col].min()
        Summary[f"max_{Col}"] = Valid[Col].max()
        Summary[f"mean_{Col}"] = Valid[Col].mean()

        Hit = Df.loc[Outliers[Col]]
        Flags.append(pd.DataFrame({"scenario": ScenarioId, "time": Hit[config.TimeCol], "kind": "outlier",
                                   "column": Col, "value": Hit[Col]}, columns=FlagCols))

    Summary["outlier_rows"] = int(AnyOutlier.sum())

    return Summary, pd.concat(Flags, ignore_index=True)


# 원자료 종류 하나의 모든 시나리오를 같은 방식으로 점검
def Detect():
    Manifest = data.LoadManifest()
    Split = data.GetScenarioSplit(Manifest=Manifest)
    Scenarios = data.GetSourceScenarios(config.Source, Manifest=Manifest)

    Summaries, Flags = [], []

    for Index, ScenarioId in enumerate(Scenarios, start=1):
        Summary, Flag = Inspect(OpenScenario(ScenarioId, Manifest=Manifest), ScenarioId, ReadMaintenance(ScenarioId, Manifest))

        Summaries.append({**Summary, "split": Split[ScenarioId]})
        Flags.append(Flag)

        print(f"\r    {Index}/{len(Scenarios)} 시나리오 점검 완료", end="", flush=True)

    print(flush=True)

    return pd.DataFrame(Summaries), pd.concat(Flags, ignore_index=True) if Flags else pd.DataFrame(columns=FlagCols)


def Main():
    SummaryDf, FlagDf = Detect()

    OutCols = [Col for Col in SummaryDf.columns if Col.startswith("outlier_") and Col != "outlier_rows"]
    Rows = SummaryDf["rows"].sum()

    print(f"[{config.Name}] 시나리오 {len(SummaryDf)}개, 행 {int(Rows):,}개, 센서 {len(OutCols)}개")
    print(f"정지 행 {int(SummaryDf['stopped'].sum()):,}개 ({SummaryDf['stopped'].sum() / Rows * 100:.2f}%), "
          f"정비 구간 행 {int(SummaryDf['maintenance'].sum()):,}개, 결측 행 {int(SummaryDf['missing'].sum()):,}개")
    print(f"짧은 결측으로 채운 칸 {int(SummaryDf['filled_cells'].sum()):,}개, 채운 뒤에도 NaN이 남은 행 {int(SummaryDf['unfilled_rows'].sum()):,}개")
    Valid, Checked = SummaryDf["valid_rows"].sum(), SummaryDf["checked_rows"].sum()
    print(f"이상치 판정 대상 행 {int(Valid):,}개 중 판정 가능한 행 {int(Checked):,}개 ({Checked / Valid * 100:.1f}%), "
          f"이상치가 하나라도 있는 행 {int(SummaryDf['outlier_rows'].sum()):,}개")
    print("센서별 이상치 행 수:")
    print(SummaryDf[OutCols].sum().astype(int).to_string())

    return SummaryDf, FlagDf


if __name__ == "__main__":
    Main()
