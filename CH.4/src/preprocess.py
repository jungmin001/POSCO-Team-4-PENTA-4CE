import pandas as pd

from . import data


# 결측 행 표시 (QUALITY_FLAG가 ok가 아니거나 숫자 컬럼에 NaN이 있는 행)
# 행 삭제와 값 대체는 하지 않음, 대체 기준은 train 시나리오에서 정함
def MarkMissing(Df):
    ResultDf = Df.copy()

    IsMissing = ResultDf.select_dtypes("number").isna().any(axis=1)

    if "QUALITY_FLAG" in ResultDf.columns:
        IsMissing = IsMissing | (ResultDf["QUALITY_FLAG"] != "ok")

    ResultDf["IS_MISSING"] = IsMissing

    return ResultDf


# 정지 행 표시 (RUNNING=0), D·E·F 특징 파일에는 RUNNING이 없어 모두 False
def MarkStopped(Df):
    ResultDf = Df.copy()

    if "RUNNING" in ResultDf.columns:
        ResultDf["IS_STOPPED"] = ResultDf["RUNNING"] == 0
    else:
        ResultDf["IS_STOPPED"] = False

    return ResultDf


# 정비 구간 행 표시 (STRT_DT <= MEAS_DT < END_DT)
# END_DT가 공란인 작업은 관측 종료까지 이어진 것으로 봄
def MarkMaintenance(Df, MaintenanceDf, Equip):
    ResultDf = Df.copy()
    EqpCd = data.Equipment[Equip][0]

    EqpMaintenance = MaintenanceDf[MaintenanceDf["EQP_TAG"] == EqpCd]
    ObservationEnd = ResultDf["MEAS_DT"].max() + pd.Timedelta(seconds=1)

    IsMaintenance = pd.Series(False, index=ResultDf.index)

    for _, Row in EqpMaintenance.iterrows():
        EndDt = Row["END_DT"] if pd.notna(Row["END_DT"]) else ObservationEnd
        IsMaintenance |= (ResultDf["MEAS_DT"] >= Row["STRT_DT"]) & (ResultDf["MEAS_DT"] < EndDt)

    ResultDf["IS_MAINTENANCE"] = IsMaintenance

    return ResultDf


# 한 시나리오, 한 설비의 센서를 읽어 결측·정지·정비 구간을 표시
# 원본 행이나 센서값은 삭제·변경하지 않음
def PreprocessScenario(ScenarioId, Equip, Root=data.RawDir):
    Df = data.LoadSensor(ScenarioId, Equip, Root)
    MaintenanceDf = data.LoadMaintenance(ScenarioId, Root)

    Df = MarkMissing(Df)
    Df = MarkStopped(Df)
    Df = MarkMaintenance(Df, MaintenanceDf, Equip)

    return Df


# 설비별 전처리 결과를 Data/Processed/{설비}.csv로 저장하고
# 결측·정지·정비 구간 행 수를 요약, 시나리오를 주지 않으면 360개 전체
# 메모리를 아끼기 위해 시나리오마다 이어서 씀
def Main(Scenarios=None):
    if Scenarios is None:
        Scenarios = list(data.GetScenarioSplit())

    ProcessedDir = data.BaseDir.parent / "Data" / "Processed"
    ProcessedDir.mkdir(parents=True, exist_ok=True)

    Report = []
    PrintEvery = 60

    for EquipIndex, Equip in enumerate(data.Equipment, start=1):
        Total = Missing = Stopped = Maintenance = 0

        # CH.4/Data/Processed/C.csv
        SavePath = ProcessedDir / f"{Equip}.csv"
        SavePath.unlink(missing_ok=True)

        print(f"[{EquipIndex}/{len(data.Equipment)}] 설비 {Equip} 시작", flush=True)

        for Index, ScenarioId in enumerate(Scenarios, start=1):
            Df = PreprocessScenario(ScenarioId, Equip)
            Df.to_csv(SavePath, mode="a", header=(Index == 1), index=False, encoding="utf-8")

            Total += len(Df)
            Missing += int(Df["IS_MISSING"].sum())
            Stopped += int(Df["IS_STOPPED"].sum())
            Maintenance += int(Df["IS_MAINTENANCE"].sum())

            if Index % PrintEvery == 0 or Index == len(Scenarios):
                print(f"    {Index}/{len(Scenarios)} 시나리오 완료", flush=True)

        Report.append(
            {
                "equipment": Equip,
                "rows": Total,
                "missing": Missing,
                "stopped": Stopped,
                "maintenance": Maintenance,
            }
        )

    print(f"시나리오 {len(Scenarios)}개 전처리 표시 결과:")
    print(pd.DataFrame(Report).to_string(index=False))

    return Report
