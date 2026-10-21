import csv

import numpy as np
import pandas as pd

from . import config


def TablePath(Equip):
    return config.OutDir / f"features_{Equip}.csv"


# 센서 컬럼: 제외 목록과 접두사에 없는 숫자 열
def GetSensorCols(Df):
    return [Col for Col in Df.select_dtypes("number").columns
            if Col not in config.ExcludeCols and not Col.startswith(config.ExcludePrefixes)]


# 한 구간의 센서 하나 요약: 평균, 표준편차, 최솟값, 최댓값, 마지막 값, 시간당 기울기
def Summarize(Time, Value):
    Ok = ~np.isnan(Value)
    Time, Value = Time[Ok], Value[Ok]

    if len(Value) == 0:
        return [np.nan] * 6

    Slope = np.nan
    if len(Value) >= config.MinSlopePoints and Time.std() > 0:
        Hours = (Time - Time.mean()) / 3600
        Slope = float(((Hours * (Value - Value.mean())).sum()) / (Hours ** 2).sum())

    return [Value.mean(), Value.std(), Value.min(), Value.max(), Value[-1], Slope]


StatNames = ["mean", "std", "min", "max", "last", "slope"]


# 한 시나리오, 한 설비의 정답 시점마다 특징 한 줄씩 만듦
# 시점 Now의 특징은 시각이 Now 이하인 행만 사용 (정지·결측·정비 구간 행은 센서 요약에서 제외하고 비율로 따로 기록)
def BuildScenario(ScenarioId, Equip, Split):
    Folder = config.ScenarioDir / ScenarioId
    # 정답 열(GT_)은 읽지 않음
    Sensor = pd.read_csv(Folder / f"{Equip}.csv", usecols=lambda Col: not Col.startswith("GT_"))
    Labels = pd.read_csv(Folder / "labels.csv")
    Labels = Labels[Labels["EQP_CD"] == Sensor["EQP_CD"].iloc[0]]

    Cols = GetSensorCols(Sensor)
    Time = Sensor["T_SEC"].to_numpy(float)
    Values = Sensor[Cols].to_numpy(float)
    Flags = {Name: Sensor[Col].to_numpy(bool) for Name, Col in
             (("stop", "IS_STOPPED"), ("missing", "IS_MISSING"), ("maint", "IS_MAINTENANCE"), ("outlier", "IS_OUTLIER"))}
    Valid = ~(Flags["stop"] | Flags["missing"] | Flags["maint"])
    MaintTimes = Time[Flags["maint"]]

    Rows = []

    for _, Label in Labels.iterrows():
        Now = float(Label["T_SEC"])
        End = int(np.searchsorted(Time, Now, side="right"))

        Row = {"scenario": ScenarioId, "split": Split, "equip": Equip, "time": Label["OBSERVED_AT"], "T_SEC": Now,
               "RISK_6H": Label["GT_RISK_6H"], "RISK_24H": Label["GT_RISK_24H"]}

        # 마지막 정비 이후 경과 시간 (정비가 없었으면 72시간으로 둠)
        Past = MaintTimes[MaintTimes <= Now]
        Row["had_maint"] = int(len(Past) > 0)
        Row["hours_since_maint"] = (Now - Past.max()) / 3600 if len(Past) else 72.0

        for Hours in config.WindowHours:
            Start = int(np.searchsorted(Time, Now - Hours * 3600, side="right"))
            Slice = slice(Start, End)
            Count = End - Start

            for Name, Flag in Flags.items():
                Row[f"w{Hours}_{Name}"] = float(Flag[Slice].mean()) if Count else np.nan

            Mask = Valid[Slice]
            for Index, Col in enumerate(Cols):
                for StatName, Stat in zip(StatNames, Summarize(Time[Slice][Mask], Values[Slice][Mask, Index])):
                    Row[f"w{Hours}_{Col}_{StatName}"] = Stat

        Rows.append(Row)

    return Rows


# 설비 하나의 모든 정답 시나리오(train·validation)의 특징 표를 만들어 저장
def Build(Equip):
    with open(config.ScenarioDir / "index.csv", encoding="utf-8") as File:
        Scenarios = [(Row["scenario"], Row["split"]) for Row in csv.DictReader(File)
                     if Row["split"] != "test" and (config.ScenarioDir / Row["scenario"] / "labels.csv").exists()]

    Rows = []
    for Index, (ScenarioId, Split) in enumerate(Scenarios, start=1):
        Rows += BuildScenario(ScenarioId, Equip, Split)
        print(f"\r    {Equip}: {Index}/{len(Scenarios)} 시나리오 특징 생성 완료", end="", flush=True)

    print(flush=True)

    Table = pd.DataFrame(Rows)
    config.OutDir.mkdir(parents=True, exist_ok=True)
    Table.to_csv(TablePath(Equip), index=False)

    print(f"특징 표 저장: {TablePath(Equip)} ({len(Table):,}행, 특징 {len(Table.columns) - len(config.MetaCols)}개)")

    return Table
