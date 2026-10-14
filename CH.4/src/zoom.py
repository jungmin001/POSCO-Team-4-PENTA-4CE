import numpy as np
import pandas as pd

from . import config, data, detect

ZoomCols = ["scenario", "column", "why", "time", "value", "local_median", "local_mad", "local_z", "window_rows"]


# 의심 지점 앞뒤 ZoomRows행을 보고, 그 값이 주변과 얼마나 다른지 계산 (설명용이라 앞뒤를 모두 봄)
# 정지·결측 행과 지점 자신은 주변 기준에서 뺌
def LocalContext(Df, Valid, Index, Col, Rows=config.ZoomRows):
    Lo, Hi = max(0, Index - Rows), min(len(Df), Index + Rows + 1)
    Near = Df.loc[Lo:Hi - 1, Col]
    Near = Near[Near.index.isin(Valid) & (Near.index != Index)]

    Median = Near.median()
    Mad = (Near - Median).abs().median() * 1.4826
    Value = Df.loc[Index, Col]

    return Median, Mad, (Value - Median) / Mad if Mad > 0 else float("nan"), len(Near)


# 3단계: 이상치와 훑어보기에서 걸린 지점(센서의 최솟값·최댓값이 있는 곳) 주변을 자세히 봄
def Zoom(FlagDf, ScreenDf):
    Manifest = data.LoadManifest()

    Targets = {}

    for _, Row in FlagDf[FlagDf["kind"] == "outlier"].iterrows():
        Targets.setdefault(Row["scenario"], []).append((Row["column"], "outlier", Row["time"], Row["value"]))

    for _, Row in ScreenDf[ScreenDf["stat"].isin(["min", "max"])].iterrows():
        Targets.setdefault(Row["scenario"], []).append((Row["column"], Row["stat"], None, None))

    Seen = set()

    Rows = []

    for Index, (ScenarioId, Items) in enumerate(Targets.items(), start=1):
        Df = detect.OpenScenario(ScenarioId, Manifest=Manifest)
        _, IsStopped, IsMissing, IsMaint = detect.GetMasks(Df, detect.ReadMaintenance(ScenarioId, Manifest))
        Valid = Df.index[~IsStopped & ~IsMissing & ~IsMaint]

        for Col, Why, Time, Value in Items:
            if Why == "outlier":
                Same = (Df[config.TimeCol] == pd.Timestamp(Time)) & np.isclose(Df[Col], Value)
                Position = Df.index[Same][0]
            else:
                Position = Df.loc[Valid, Col].idxmax() if Why == "max" else Df.loc[Valid, Col].idxmin()

            # 같은 지점이 이상치이면서 최솟값·최댓값이기도 하면 한 번만 봄
            if (ScenarioId, Col, Position) in Seen:
                continue
            Seen.add((ScenarioId, Col, Position))

            Median, Mad, Z, Count = LocalContext(Df, Valid, Position, Col)
            Rows.append({"scenario": ScenarioId, "column": Col, "why": Why, "time": Df.loc[Position, config.TimeCol],
                         "value": Df.loc[Position, Col], "local_median": Median, "local_mad": Mad, "local_z": Z,
                         "window_rows": Count})

        print(f"\r    {Index}/{len(Targets)} 시나리오 주변 탐지 완료", end="", flush=True)

    print(flush=True)

    return pd.DataFrame(Rows, columns=ZoomCols)


def Main(FlagDf, ScreenDf):
    ZoomDf = Zoom(FlagDf, ScreenDf)

    print(f"주변을 본 지점 {len(ZoomDf)}개")

    if len(ZoomDf):
        Top = ZoomDf.reindex(ZoomDf["local_z"].abs().sort_values(ascending=False).index).head(10)
        NumCols = Top.select_dtypes("number").columns
        Top[NumCols] = Top[NumCols].round(4)
        print(Top.to_string(index=False))

    return ZoomDf
