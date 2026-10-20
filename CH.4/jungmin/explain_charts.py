import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jungmin import config, plot_sensors

import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

# 차트를 읽는 법을 그림 위에 직접 적은 해설판 (CH.4 폴더에서 실행, 먼저 python jungmin/Main.py가 끝나 있어야 함)
#   python jungmin/explain_charts.py            해설 그림 4장을 저장하고 창으로 보여줌
#   python jungmin/explain_charts.py noshow     저장만 함
Box = dict(boxstyle="round,pad=0.4", fc="#fffbe6", ec="#d4a900", lw=1)
Arrow = dict(arrowstyle="->", color="#c0392b", lw=1.8)


def Note(Ax, Text, Xy, XyText, Color="#fffbe6"):
    Ax.annotate(Text, xy=Xy, xytext=XyText, fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc=Color, ec="#d4a900", lw=1),
                arrowprops=Arrow, ha="left", va="center", zorder=10)


def Shade(Ax, Df):
    Hours = Df["T_SEC"].to_numpy() / 3600
    States = Df["GT_HEALTH_STATE"].fillna("").to_numpy()
    for Start, End, State in plot_sensors.StateSpans(Hours, States):
        if State in ("degraded", "failed"):
            Ax.axvspan(Start, End, color="red", alpha=0.15, linewidth=0)
        elif State == "maintenance":
            Ax.axvspan(Start, End, color="gray", alpha=0.3, linewidth=0)


# 1) 센서 값: 정상과 열화 비교
def SensorValues(Equip, Bad, Good):
    Data = {ScenarioId: plot_sensors.Load(ScenarioId, Equip) for ScenarioId in (Bad, Good)}
    Cols = ["CONTROL_DELAY", "OSC-STK", "WD-SPD"]

    Fig, Axes = plt.subplots(3, 2, figsize=(14, 9.5), sharex=True, sharey="row")
    for Column, ScenarioId in enumerate((Bad, Good)):
        Df, _, _, Hourly = Data[ScenarioId]
        for Row, Col in enumerate(Cols):
            Ax = Axes[Row][Column]
            Shade(Ax, Df)
            Ax.plot(Hourly.index + 0.5, Hourly[Col], color="tab:blue", linewidth=2)
            Ax.grid(alpha=0.3)
            if Column == 0:
                Ax.set_ylabel(f"{Col}\n(센서 값)")
        Axes[0][Column].set_title(f"{ScenarioId}  " + ("열화가 있는 시나리오" if ScenarioId == Bad else "정상 시나리오"), fontsize=13)
        Axes[-1][Column].set_xlabel("시나리오 시작 후 시간 (h)")
    Axes[-1][0].set_xlim(0, 72)
    Axes[2][0].set_ylim(0.72, 1.34)

    Note(Axes[0][0], "붉은 배경 = 열화 구간", (12, 0.15), (3, 0.195))
    Note(Axes[0][0], "회색 배경 = 정비로 설비가 멈춘 구간\n(이 구간의 값은 계산에서 제외)", (49.5, 0.175), (21, 0.105))
    Note(Axes[0][0], "열화되면 계속 올라감\n0.13 → 0.21", (44, 0.208), (54.5, 0.115))
    Note(Axes[0][1], "정상은 0.08~0.11에서\n거의 일정", (36, 0.083), (40, 0.165))
    Note(Axes[1][0], "열화되면 서서히 내려감\n4.95 → 4.7", (43, 4.73), (3, 4.74))
    Note(Axes[1][1], "정상은 4.95 근처에서\n일정하게 진동", (30, 5.0), (36, 4.83))
    Note(Axes[2][0], "이 센서는 정상과 열화가 거의 같음\n→ 열화를 구분하는 데 도움이 안 됨", (33, 1.12), (12, 1.25))

    Fig.suptitle("센서 값 읽는 법: 같은 센서를 왼쪽(열화)과 오른쪽(정상)에서 비교하세요", fontsize=14, y=0.995)
    Fig.tight_layout()
    Path_ = config.OutDir / f"explain_1_sensor_value_{Equip}.png"
    Fig.savefig(Path_, dpi=105)

    return Path_


# 2) 센서 변화량
def SensorChange(Equip, Bad, Good):
    Data = {ScenarioId: plot_sensors.Load(ScenarioId, Equip) for ScenarioId in (Bad, Good)}
    Cols = ["WD-LOD", "CONTROL_DELAY"]

    Fig, Axes = plt.subplots(2, 2, figsize=(14, 7.5), sharex=True, sharey="row")
    for Column, ScenarioId in enumerate((Bad, Good)):
        Df, _, _, Hourly = Data[ScenarioId]
        for Row, Col in enumerate(Cols):
            Ax = Axes[Row][Column]
            Shade(Ax, Df)
            Ax.axhline(0, color="black", linewidth=1)
            Ax.plot(Hourly.index[1:] + 0.5, Hourly[Col].diff().iloc[1:], color="tab:orange", linewidth=2, marker="o", markersize=3)
            Ax.grid(alpha=0.3)
            if Column == 0:
                Ax.set_ylabel(f"{Col}\n(1시간 동안 변한 양)")
        Axes[0][Column].set_title(f"{ScenarioId}  " + ("열화가 있는 시나리오" if ScenarioId == Bad else "정상 시나리오"), fontsize=13)
        Axes[-1][Column].set_xlabel("시나리오 시작 후 시간 (h)")
    Axes[-1][0].set_xlim(0, 72)
    Axes[0][0].set_ylim(-9, 14)
    Axes[1][0].set_ylim(-0.009, 0.0125)

    Note(Axes[0][0], "검은 선 = 0 (변화 없음)\n위: 값이 올라가는 중\n아래: 값이 내려가는 중", (46, 0), (30, 11.5))
    Note(Axes[0][0], "오르내림 폭이 정상과 비슷\n(최대 약 ±7)", (11.5, 7.5), (1, 11.5))
    Note(Axes[0][1], "정상도 오르내림 폭이 비슷\n(최대 약 ±6)", (23.5, 6.0), (4, 11.5))
    Note(Axes[1][0], "변화 폭이 정상과 거의 같음", (32.5, 0.0064), (3, 0.0105))
    Fig.text(0.5, 0.012, "해석: 변화 속도(이 그림)로는 정상과 열화가 구분되지 않음  →  구분은 '값의 수준과 한쪽으로 쌓이는 이동'(앞 그림)에서 나옴",
             ha="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="#e8f6e8", ec="#3c9d3c"))
    Fig.suptitle("센서 변화량 읽는 법: 값이 1시간 동안 얼마나 변했는가", fontsize=14, y=0.995)
    Fig.tight_layout(rect=(0, 0.05, 1, 1))
    Path_ = config.OutDir / f"explain_2_sensor_change_{Equip}.png"
    Fig.savefig(Path_, dpi=105)

    return Path_


def LoadPredictions(Tag, Model="forest"):
    Pred = pd.read_csv(config.OutDir / f"predictions_{Tag}.csv")
    Result = pd.read_csv(config.OutDir / f"results_{Tag}.csv").set_index("model")
    Y = Pred.iloc[:, 2].astype(int).to_numpy()

    return Y, Pred[f"p_{Model}"].to_numpy(), float(Result.loc[Model, "threshold@train선택임계값"])


# 3) PR 곡선
def PRCurve(Tag):
    Y, P, Chosen = LoadPredictions(Tag)
    Precision, Recall, _ = precision_recall_curve(Y, P)
    Ap = average_precision_score(Y, P)
    Pred = P >= Chosen
    Tp, Fp, Fn = int((Pred & (Y == 1)).sum()), int((Pred & (Y == 0)).sum()), int((~Pred & (Y == 1)).sum())
    ChosenPrecision, ChosenRecall = Tp / (Tp + Fp), Tp / (Tp + Fn)

    Fig, Ax = plt.subplots(figsize=(11, 7))
    Ax.fill_between(Recall, Precision, step="post", color="tab:orange", alpha=0.12)
    Ax.step(Recall, Precision, where="post", color="tab:orange", linewidth=2.2, label="랜덤 포레스트")
    Ax.axhline(Y.mean(), color="gray", linestyle="--", linewidth=1.5)
    Ax.scatter([ChosenRecall], [ChosenPrecision], s=130, color="#c0392b", zorder=6)
    Ax.set_xlim(0, 1.02)
    Ax.set_ylim(0, 1.05)
    Ax.set_xlabel("재현율  =  실제 고장 중 미리 잡은 비율  (오른쪽일수록 놓치는 고장이 적음)", fontsize=11)
    Ax.set_ylabel("정밀도  =  경보 중 진짜 고장 비율  (위쪽일수록 헛경보가 적음)", fontsize=11)
    Ax.set_title("PR 곡선 읽는 법 (연주 24시간 위험, validation)", fontsize=14)
    Ax.grid(alpha=0.3)

    Note(Ax, "오른쪽 위로 갈수록 좋은 모델", (0.99, 1.04), (0.50, 0.99))
    Note(Ax, f"회색 점선 = 아무 생각 없이 찍을 때의 수준 ({Y.mean():.2f})\n주황 곡선이 이 선보다 훨씬 위에 있으면 잘 맞히는 것", (0.5, Y.mean()), (0.05, 0.03))
    Note(Ax, f"빨간 점 = train으로 정한 경보 기준 ({Chosen:.2f})\n경보 {Tp + Fp}번 중 {Tp}번 적중 (정밀도 {ChosenPrecision:.2f})\n실제 고장 {Tp + Fn}건 중 {Tp}건 잡음 (재현율 {ChosenRecall:.2f})",
         (ChosenRecall, ChosenPrecision), (0.12, 0.46))
    Note(Ax, "더 많이 잡으려고 하면(오른쪽)\n헛경보가 늘어 정밀도가 급히 떨어짐", (0.88, 0.37), (0.60, 0.20))
    Ax.text(0.12, 0.215, f"주황색 면적 = PR-AUC {Ap:.2f}\n(1에 가까울수록 좋음, 찍기는 {Y.mean():.2f})", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="tab:orange"))
    Fig.tight_layout()
    Path_ = config.OutDir / f"explain_3_pr_curve_{Tag}.png"
    Fig.savefig(Path_, dpi=105)

    return Path_


# 4) 임계값 곡선
def ThresholdCurve(Tag):
    Y, P, Chosen = LoadPredictions(Tag)
    Grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    Pred = P[None, :] >= Grid[:, None]
    Tp, Fp = (Pred & (Y == 1)).sum(axis=1), (Pred & (Y == 0)).sum(axis=1)
    Precision = np.where(Tp + Fp > 0, Tp / np.maximum(Tp + Fp, 1), np.nan)
    Recall = Tp / Y.sum()
    F1 = 2 * Precision * Recall / np.maximum(Precision + Recall, 1e-12)
    PredChosen = P >= Chosen
    TpC, FpC = int((PredChosen & (Y == 1)).sum()), int((PredChosen & (Y == 0)).sum())
    PrecisionC, RecallC = TpC / (TpC + FpC), TpC / int(Y.sum())

    Fig, Ax = plt.subplots(figsize=(12, 7))
    Ax.axvspan(0, Chosen, color="#d6eaf8", alpha=0.5, linewidth=0)
    Ax.axvspan(Chosen, 1, color="#fdebd0", alpha=0.5, linewidth=0)
    Ax.plot(Grid, Precision, color="tab:blue", linewidth=2.4, label="정밀도 (경보가 맞은 비율)")
    Ax.plot(Grid, Recall, color="tab:orange", linewidth=2.4, label="재현율 (고장을 잡은 비율)")
    Ax.plot(Grid, F1, color="tab:green", linewidth=2, linestyle=":", label="F1 (둘의 균형)")
    Ax.axvline(Chosen, color="gray", linestyle="--", linewidth=1.8)
    Ax.set_xlim(0, 1)
    Ax.set_ylim(0, 1.2)
    Ax.set_xlabel("경보 기준 (임계값): 모델이 계산한 고장 확률이 이 값 이상이면 경보를 울림", fontsize=11)
    Ax.set_ylabel("점수 (1에 가까울수록 좋음)", fontsize=11)
    Ax.set_title("임계값 곡선 읽는 법 (연주 24시간 위험, 랜덤 포레스트, validation)", fontsize=14)
    Ax.grid(alpha=0.3)
    Ax.legend(loc="lower center", ncol=3, fontsize=10)

    Note(Ax, "기준을 올리면 경보가 까다로워져서\n정밀도(파랑)는 올라감", (0.9, Precision[89]), (0.58, 1.03))
    Note(Ax, "대신 놓치는 고장이 늘어서\n재현율(주황)은 내려감", (0.85, Recall[84]), (0.62, 0.30))
    Note(Ax, f"점선 = train으로 정한 기준 ({Chosen:.2f})\n정밀도 {PrecisionC:.2f}, 재현율 {RecallC:.2f}\n(경보 {TpC + FpC}번 중 {TpC}번 적중,\n고장 {int(Y.sum())}건 중 {TpC}건 잡음)",
         (Chosen, RecallC), (0.03, 0.40))
    Ax.text(0.02, 1.185, "← 기준 낮음: 경보가 자주 울림 (고장은 많이 잡지만 헛경보도 많음)", fontsize=10, va="top", color="#1b4f72")
    Ax.text(0.98, 1.185, "기준 높음 →: 경보가 가끔 울림 (헛경보는 적지만 고장을 놓침)", fontsize=10, ha="right", va="top", color="#935116")
    Fig.tight_layout()
    Path_ = config.OutDir / f"explain_4_threshold_curve_{Tag}.png"
    Fig.savefig(Path_, dpi=105)

    return Path_


def Main(Args):
    Equip = next((Name for Name in Args if Name in config.Equipments), config.DefaultEquipments[0])
    Paths = [SensorValues(Equip, "S0002", "S0001"), SensorChange(Equip, "S0002", "S0001"),
             PRCurve(f"{Equip}_24H"), ThresholdCurve(f"{Equip}_24H")]

    for Path_ in Paths:
        print(f"저장: {Path_}")

    if config.Show:
        plt.show()


if __name__ == "__main__":
    Main(sys.argv[1:])
