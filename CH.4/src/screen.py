import pandas as pd

from . import config, data

ScreenCols = ["scenario", "split", "column", "stat", "value", "z"]


# 2단계: 시나리오별 요약(정지 행 제외 min/max/mean)을 train 분포와 비교해서 평소와 다른 시나리오를 찾음
# 평소 범위는 train 시나리오로만 정함
def Screen(SummaryDf, Stats=config.ScreenStats, NSigma=config.ScreenNSigma):
    Sensors = sorted(Col[len("outlier_"):] for Col in SummaryDf.columns if Col.startswith("outlier_") and Col != "outlier_rows")

    Flags = []

    for Col in Sensors:
        for Stat in Stats:
            Key = f"{Stat}_{Col}"
            Train = SummaryDf.loc[SummaryDf["split"] == "train", Key]
            Median = Train.median()
            Mad = (Train - Median).abs().median() * 1.4826

            # 분포가 한 값에 몰려 있으면 판정하지 않음
            if pd.isna(Mad) or Mad == 0:
                continue

            Z = (SummaryDf[Key] - Median) / Mad
            IsFlagged = Z.abs() > NSigma

            if not IsFlagged.any():
                continue

            Out = SummaryDf.loc[IsFlagged, ["scenario", "split"]].copy()
            Out["column"] = Col
            Out["stat"] = Stat
            Out["value"] = SummaryDf.loc[IsFlagged, Key]
            Out["z"] = Z[IsFlagged]
            Flags.append(Out[ScreenCols])

    return pd.concat(Flags, ignore_index=True) if Flags else pd.DataFrame(columns=ScreenCols)


# 정답지(진단)와 비교해서 훑어보기 결과가 열화 시나리오와 얼마나 겹치는지 확인 (분석용, 입력으로 쓰지 않음)
def Validate(SummaryDf, ScreenDf):
    Labeled = SummaryDf.loc[SummaryDf["split"] != "test", "scenario"]

    Degraded = set()
    for ScenarioId in Labeled:
        Diagnosis = data.LoadTruth("diagnostics", ScenarioId, config.EqpCode)
        if Diagnosis[config.HealthCol].isin(config.DegradedStates).any():
            Degraded.add(ScenarioId)

    OutCols = [Col for Col in SummaryDf.columns if Col.startswith("outlier_") and Col != "outlier_rows"]
    HasOutlier = set(SummaryDf.loc[SummaryDf[OutCols].sum(axis=1) > 0, "scenario"])
    Screened = set(ScreenDf["scenario"])

    print(f"정답이 있는 시나리오 {len(Labeled)}개 중 열화 {len(Degraded & set(Labeled))}개")
    for Name, Picked in (("훑어보기 의심", Screened), ("이상치 있음", HasOutlier)):
        Picked = Picked & set(Labeled)
        print(f"  {Name}: {len(Picked)}개 (열화 시나리오 {len(Picked & Degraded)}개, 비열화 {len(Picked - Degraded)}개)")


def Main(SummaryDf):
    ScreenDf = Screen(SummaryDf)

    print(f"의심 (시나리오, 센서, 통계) {len(ScreenDf)}건, 시나리오 {ScreenDf['scenario'].nunique()}개")

    if len(ScreenDf):
        print(ScreenDf.groupby(["column", "stat", "split"]).size().to_string())

    Validate(SummaryDf, ScreenDf)

    return ScreenDf
