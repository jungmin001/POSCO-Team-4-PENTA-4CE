import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jungmin import config, features

import matplotlib

if not config.Show:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

# 센서별 시간 변화 그림 (CH.4 폴더에서 실행)
#   python jungmin/plot_sensors.py                  연주 C, 열화 시나리오 S0002와 정상 시나리오 S0001 비교
#   python jungmin/plot_sensors.py S0058 S0147 C    시나리오와 설비를 지정
#   python jungmin/plot_sensors.py noshow           창을 띄우지 않고 저장만 함
Korean = any("AppleGothic" in Font.name for Font in font_manager.fontManager.ttflist)
if Korean:
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False

Text = ({"value": "값 (정지·결측·정비 제외)", "change": "1시간 평균의 시간당 변화량", "hour": "시나리오 시작 후 시간(h)",
         "degraded": "열화·고장", "maint": "정비", "raw": "원값", "hourly": "1시간 평균", "has": "열화 있음", "normal": "정상"}
        if Korean else
        {"value": "value (excl. stop/missing/maint.)", "change": "change per hour of hourly mean", "hour": "hours since start",
         "degraded": "degraded/failed", "maint": "maintenance", "raw": "raw", "hourly": "hourly mean", "has": "degraded", "normal": "normal"})


# 상태 구간 (연속된 같은 상태의 시작·끝 시간)
def StateSpans(Hours, State):
    Spans, Start = [], 0
    for Index in range(1, len(State) + 1):
        if Index == len(State) or State[Index] != State[Start]:
            Spans.append((Hours[Start], Hours[Index - 1], State[Start]))
            Start = Index

    return Spans


def Load(ScenarioId, Equip):
    Df = pd.read_csv(config.ScenarioDir / ScenarioId / f"{Equip}.csv")
    Cols = [Col for Col in features.GetSensorCols(Df) if Col != "LOAD"]
    Valid = ~(Df["IS_STOPPED"].astype(bool) | Df["IS_MISSING"].astype(bool) | Df["IS_MAINTENANCE"].astype(bool))
    Hourly = Df.loc[Valid].groupby((Df.loc[Valid, "T_SEC"] // 3600).astype(int))[Cols].mean().reindex(range(72))

    return Df, Cols, Valid, Hourly


def Main(Args):
    Equip = next((Name for Name in Args if Name in config.Equipments), config.DefaultEquipments[0])
    Scenarios = [Name for Name in Args if Name.startswith("S")] or ["S0002", "S0001"]
    Data = {ScenarioId: Load(ScenarioId, Equip) for ScenarioId in Scenarios}
    Cols = Data[Scenarios[0]][1]

    for Kind in ("value", "change"):
        Fig, Axes = plt.subplots(len(Cols), len(Scenarios), figsize=(6.5 * len(Scenarios), 1.9 * len(Cols)),
                                 sharex=True, sharey="row", squeeze=False)

        for Column, ScenarioId in enumerate(Scenarios):
            Df, _, Valid, Hourly = Data[ScenarioId]
            Hours = Df["T_SEC"].to_numpy() / 3600
            States = Df["GT_HEALTH_STATE"].fillna("").to_numpy() if "GT_HEALTH_STATE" in Df else np.array([""] * len(Df))
            HasDegraded = bool(np.isin(States, ["degraded", "failed"]).any())

            for Row, Col in enumerate(Cols):
                Ax = Axes[Row][Column]

                for Start, End, State in StateSpans(Hours, States):
                    if State in ("degraded", "failed"):
                        Ax.axvspan(Start, End, color="red", alpha=0.15, linewidth=0)
                    elif State == "maintenance":
                        Ax.axvspan(Start, End, color="gray", alpha=0.3, linewidth=0)

                if Kind == "value":
                    Ax.plot(Hours[Valid.to_numpy()][::6], Df.loc[Valid, Col].to_numpy()[::6], color="tab:blue", alpha=0.25, linewidth=0.5)
                    Ax.plot(Hourly.index + 0.5, Hourly[Col], color="tab:blue", linewidth=1.6)
                else:
                    Ax.axhline(0, color="black", linewidth=0.5)
                    Ax.plot(Hourly.index[1:] + 0.5, Hourly[Col].diff().iloc[1:], color="tab:orange", linewidth=1.4, marker="o", markersize=2.5)

                Ax.grid(alpha=0.3)
                if Column == 0:
                    Ax.set_ylabel(Col, fontsize=9)
                if Row == 0:
                    Ax.set_title(f"{ScenarioId} ({Text['has'] if HasDegraded else Text['normal']})")

            Axes[-1][Column].set_xlabel(Text["hour"])
            Axes[-1][Column].set_xlim(0, 72)

        Fig.suptitle(f"{Equip}: " + Text[Kind], y=0.995)
        Axes[0][-1].legend(handles=[Patch(color="red", alpha=0.3, label=Text["degraded"]), Patch(color="gray", alpha=0.4, label=Text["maint"])],
                           loc="upper right", fontsize=8)
        Fig.tight_layout()
        Path_ = config.OutDir / f"sensor_{Kind}_{Equip}.png"
        Fig.savefig(Path_, dpi=110)
        print(f"저장: {Path_}")

        if not config.Show:
            plt.close(Fig)

    # 센서별 1시간 평균 변화량의 크기: 정상 시나리오와 열화 시나리오 비교
    Rows = []
    for Col in Cols:
        Row = {"센서": Col}
        for ScenarioId in Scenarios:
            Change = Data[ScenarioId][3][Col].diff()
            Row[f"{ScenarioId} 변화량 표준편차"] = Change.std()
            Row[f"{ScenarioId} 최대 변화량"] = Change.abs().max()
        Rows.append(Row)

    pd.set_option("display.width", 200)
    print("\n[1시간 평균의 시간당 변화량 크기]")
    print(pd.DataFrame(Rows).round(4).to_string(index=False))

    # 저장한 그림을 창으로도 보여줌 (창을 모두 닫으면 종료)
    if config.Show:
        plt.show()


if __name__ == "__main__":
    Main(sys.argv[1:])
