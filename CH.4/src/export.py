import numpy as np
import pandas as pd

from . import config, data, detect

ScenarioDir = data.OutDir / config.ScenarioDirName


# 정답지의 상태 구간(START_AT <= 시각 < END_AT)을 각 행에 붙임, 구간 밖이면 빈칸
def AttachTruth(Out, ScenarioId, Times):
    for Col in config.TruthCols:
        Out[config.TruthPrefix + Col] = ""

    Diagnosis = data.LoadTruth("diagnostics", ScenarioId, config.EqpCode).sort_values(config.TruthStartCol)

    Starts = Diagnosis[config.TruthStartCol].to_numpy(dtype="datetime64[ns]")
    Ends = Diagnosis[config.TruthEndCol].to_numpy(dtype="datetime64[ns]")
    Time = Times.to_numpy(dtype="datetime64[ns]")

    Position = np.searchsorted(Starts, Time, side="right") - 1
    IsInside = (Position >= 0) & (Time < Ends[np.clip(Position, 0, None)])

    for Col in config.TruthCols:
        Values = Diagnosis[Col].fillna("").to_numpy(dtype=object)
        Out[config.TruthPrefix + Col] = np.where(IsInside, Values[np.clip(Position, 0, None)], "")


# 시나리오 하나, 설비 하나의 전처리 완료 표: 짧은 결측은 채우고, 행마다 표시와 경과 시간을 붙임
# 행은 삭제하지 않고 원본 열은 그대로 두며, 정답지 상태 열은 train·validation 시나리오에만 채움
def BuildTable(ScenarioId, Start, Split, Manifest):
    Df = detect.OpenScenario(ScenarioId, Manifest=Manifest)
    Maint = detect.ReadMaintenance(ScenarioId, Manifest)

    Sensors, IsStopped, IsMissing, IsMaint = detect.GetMasks(Df, Maint)
    _, _, Filled = detect.FillShortGaps(Df, Sensors, ScenarioId)
    _, _, Outliers = detect.FindOutliers(Df, Sensors, IsStopped, IsMissing, IsMaint)

    AnyOutlier = pd.Series(False, index=Df.index)
    for Col in Sensors:
        AnyOutlier |= Outliers[Col]

    Out = Filled.copy()
    Out[config.TimeCol] = Df[config.TimeCol].dt.strftime(config.TimeFormat)
    Out.insert(0, "ROW_ID", np.arange(1, len(Df) + 1))
    Out.insert(1, "T_SEC", (Df[config.TimeCol] - Start).dt.total_seconds())

    for Key, Mask in (("missing", IsMissing), ("stopped", IsStopped), ("maintenance", IsMaint), ("outlier", AnyOutlier)):
        Out[config.FlagNames[Key]] = Mask.astype(int)

    if Split == "test":
        for Col in config.TruthCols:
            Out[config.TruthPrefix + Col] = ""
    else:
        AttachTruth(Out, ScenarioId, Df[config.TimeCol])

    return Out


# 흐름 정보 한 종류의 표: 시각은 센서 파일과 같은 형식으로 바꾸고 경과 초 열을 붙임, 정답 종류는 열 이름에 GT_를 붙임
def BuildFlowTable(Kind, ScenarioId, Start, Manifest):
    return BuildRawTable(data.RawDir / "flow" / Kind / f"{ScenarioId}.csv", config.FlowTimeCols[Kind], Start,
                         config.FlowKeyCols if Kind in config.FlowTruthKinds else None)


# 정답 한 종류의 표 (train·validation 시나리오에만 있음), 키 열을 뺀 모든 열 이름에 GT_를 붙임
def BuildAnswerTable(Kind, ScenarioId, Start, Manifest):
    return BuildRawTable(data.RawDir / Kind / f"{ScenarioId}.csv", config.AnswerTimeCols[Kind], Start,
                         config.AnswerKeyCols[Kind])


# 정비 이력 한 종류의 표 (원본 글자 그대로, 행이 없는 시나리오는 열 이름만 있는 빈 표)
def BuildHistoryTable(Kind, ScenarioId, Start, Manifest):
    return AddTimes(data.ReadScenario(Kind, ScenarioId, Manifest=Manifest, AsText=True), config.HistoryTimeCols[Kind], Start)


# 원본 CSV를 읽어 정답 열 이름을 바꾸고(TruthKeys가 있을 때) 시각 형식과 경과 초 열을 맞춤, 파일이 없으면 None
def BuildRawTable(Path_, TimeCols, Start, TruthKeys):
    if not Path_.exists():
        return None

    # 값을 숫자로 바꾸지 않고 원본 글자 그대로 읽음 (빈칸이 있는 열이 0 -> 0.0처럼 바뀌는 것을 막음)
    Df = pd.read_csv(Path_, dtype=str, keep_default_na=False, encoding="utf-8-sig")

    if TruthKeys is not None:
        Df = Df.rename(columns={Col: config.TruthPrefix + Col for Col in Df.columns if Col not in TruthKeys})

    return AddTimes(Df, TimeCols, Start)


# 시각 열을 센서 파일과 같은 형식으로 바꾸고 시나리오 시작 이후의 경과 초 열을 붙임, 맨 앞에 ROW_ID를 넣음
def AddTimes(Df, TimeCols, Start):
    Df = Df.copy()

    for Col, SecCol in TimeCols.items():
        Time = pd.to_datetime(Df[Col], utc=True)
        Df[Col] = Time.dt.strftime(config.TimeFormat)
        Df[SecCol] = (Time - Start).dt.total_seconds()

    Df.insert(0, "ROW_ID", range(1, len(Df) + 1))

    return Df


# 시나리오별 폴더에 표를 종류마다 파일 하나씩 저장, Builders의 반환이 None이면 저장하지 않음
# flow(흐름 정보): Scenarios/S0001/flow/{종류}.csv
# answers(정답 diagnostics·labels·targets): 원본처럼 폴더 없이 Scenarios/S0001/{종류}.csv
# history(정비 이력 G 6종): Scenarios/S0001/history/{종류}.csv
TableGroups = {"flow": ("flow", config.FlowTimeCols, BuildFlowTable),
               "answers": ("", config.AnswerTimeCols, BuildAnswerTable),
               "history": ("history", config.HistoryTimeCols, BuildHistoryTable)}


# index.csv의 행 수 열 이름
def RowKey(Group, Kind):
    Folder = TableGroups[Group][0]

    return f"rows_{Folder}_{Kind}" if Folder else f"rows_{Kind}"


def ExportGroup(Info, Manifest, Group):
    Folder, Kinds, Build = TableGroups[Group]

    for Index, (ScenarioId, Row) in enumerate(Info.items(), start=1):
        Start = pd.Timestamp(Row["start_at"])
        OutDir = ScenarioDir / ScenarioId / Folder
        OutDir.mkdir(parents=True, exist_ok=True)

        for Kind in Kinds:
            Table = Build(Kind, ScenarioId, Start, Manifest)

            if Table is not None:
                Table.to_csv(OutDir / f"{Kind}.csv", index=False)
                Row[RowKey(Group, Kind)] = len(Table)

        print(f"\r    {Group}: {Index}/{len(Info)} 시나리오 저장 완료", end="", flush=True)

    print(flush=True)


# 시나리오별 폴더에 설비마다 파일 하나씩 저장: Scenarios/S0001/C.csv, 이름에 "flow", "answers", "history"를 주면 해당 표 그룹 저장
# 이름을 주지 않으면 모든 설비와 표 그룹, index.csv는 기존 내용에 이번에 만든 열을 합쳐서 저장
def Export(Names=None):
    ScenarioDir.mkdir(parents=True, exist_ok=True)

    Manifest = data.LoadManifest()
    Info = {S["id"]: {"scenario": S["id"], "split": S["split"], "start_at": S["startAt"]} for S in Manifest["scenarios"]}

    for Name in (Names or list(config.Profiles) + list(TableGroups)):
        if Name in TableGroups:
            ExportGroup(Info, Manifest, Name)
            continue

        config.Use(Name)
        Scenarios = data.GetSourceScenarios(config.Source, Manifest=Manifest)

        for Index, ScenarioId in enumerate(Scenarios, start=1):
            Start = pd.Timestamp(Info[ScenarioId]["start_at"])
            Table = BuildTable(ScenarioId, Start, Info[ScenarioId]["split"], Manifest)

            (ScenarioDir / ScenarioId).mkdir(exist_ok=True)
            Table.to_csv(ScenarioDir / ScenarioId / f"{Name}.csv", index=False)
            Info[ScenarioId][f"rows_{Name}"] = len(Table)

            print(f"\r    {Name}: {Index}/{len(Scenarios)} 시나리오 저장 완료", end="", flush=True)

        print(flush=True)

    IndexPath = ScenarioDir / "index.csv"
    Index = pd.DataFrame(Info.values()).set_index("scenario")

    if IndexPath.exists():
        Index = Index.combine_first(pd.read_csv(IndexPath).set_index("scenario"))

    # 열 순서: 설비(config.Profiles 순서) 다음에 표 그룹 (flow, answers, history)
    RowCols = [f"rows_{Name}" for Name in config.Profiles] + [RowKey(Group, Kind) for Group, (_, Kinds, _) in TableGroups.items() for Kind in Kinds]
    # 아직 만들지 않은 열은 빈칸으로 둠
    Index = Index.reindex(columns=["split", "start_at"] + RowCols)
    Index[RowCols] = Index[RowCols].astype("Int64")

    Index.reset_index().to_csv(IndexPath, index=False)


def Main(Names=None):
    Export(Names)

    print(f"저장 위치: {ScenarioDir}")


if __name__ == "__main__":
    Main()
