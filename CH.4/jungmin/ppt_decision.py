from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

import ppt_font


# 정비 의사결정 설명 이미지 (PPT용, 배경 투명), CH.4 폴더에서 실행
#   python jungmin/ppt_decision.py
#
# 저장 (jungmin/outputs/ppt)
#   5_decision.png        글자가 진한 회색 (밝은 슬라이드용)
#   5_decision_light.png  글자가 흰색 (어두운 슬라이드용)
# 숫자는 jungmin/outputs/unreal/decisions_C_check.csv의 한 시점(위험이 높은 순간)에서 읽음, 금액은 가정값(단위 만원)
OutDir = Path(__file__).resolve().parent / "outputs" / "ppt"
CheckPath = Path(__file__).resolve().parent / "outputs" / "unreal" / "decisions_C_check.csv"
Scenario, Time = "S0222", "2027-06-05T06:00:00Z"

Gray, Blue = "#b9c2cc", "#2f6fdb"
ppt_font.UseMatplotlib()


def Draw(Name, Ink, Sub):
    Row = pd.read_csv(CheckPath).query("Scenario == @Scenario and Time == @Time").iloc[0]
    Options = [("기다린다", Row["Cost_wait"], "wait"), ("6시간 뒤에 교체한다", Row["Cost_replace_6h"], "replace_6h"), ("바로 교체한다", Row["Cost_replace_now"], "replace_now")]
    Best = Row["Recommended"]

    Fig = plt.figure(figsize=(11, 4.7), dpi=200)
    Ax = Fig.add_axes([0, 0, 1, 1])
    Ax.set_xlim(0, 11)
    Ax.set_ylim(0, 4.7)
    Ax.axis("off")

    Ax.text(0.3, 4.3, "고장 위험이 높은 순간의 예", ha="left", va="center", color=Ink, fontsize=19, fontweight="bold")
    Ax.text(0.3, 3.82, f"24시간 안에 고장 날 가능성 {Row['P24H'] * 100:.0f}%  ·  6시간 안 {Row['P6H'] * 100:.0f}%", ha="left", va="center", color=Sub, fontsize=13.5)

    Max, X0, BarW = max(Cost for _, Cost, _ in Options), 3.35, 5.3
    for Index, (Label, Cost, Key) in enumerate(Options):
        Y = 2.75 - Index * 0.95
        Pick = Key == Best
        Ax.text(0.3, Y, Label, ha="left", va="center", color=Ink, fontsize=16 if Pick else 15, fontweight="bold" if Pick else "regular")
        Ax.add_patch(FancyBboxPatch((X0, Y - 0.28), max(0.12, Cost / Max * BarW), 0.56, boxstyle="round,pad=0,rounding_size=0.1", fc=Blue if Pick else Gray, ec="none"))
        Ax.text(X0 + Cost / Max * BarW + 0.18, Y, f"{Cost:,.0f}만원", ha="left", va="center", color=Blue if Pick else Sub, fontsize=15, fontweight="bold" if Pick else "regular")
        if Pick:
            Pill = X0 + Cost / Max * BarW + 1.95
            Ax.add_patch(FancyBboxPatch((Pill, Y - 0.2), 0.95, 0.4, boxstyle="round,pad=0,rounding_size=0.2", fc=Blue, ec="none"))
            Ax.text(Pill + 0.475, Y, "추천", ha="center", va="center", color="white", fontsize=13)

    Ax.text(0.3, 0.2, "예상 손실 비교 · 금액은 가정값이며 실제 비용을 넣으면 다시 계산됩니다", ha="left", va="center", color=Sub, fontsize=11.5)
    OutDir.mkdir(parents=True, exist_ok=True)
    Fig.savefig(OutDir / Name, transparent=True)
    plt.close(Fig)
    print(f"저장: {OutDir / Name}")


def Main():
    Draw("5_decision.png", "#1f2933", "#6b7683")
    Draw("5_decision_light.png", "#ffffff", "#c3cbd3")


if __name__ == "__main__":
    Main()
