import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, detect

import ppt_font


# 챕터 4 전처리 설명 이미지 (PPT용), CH.4 폴더에서 실행
#   python jungmin/ppt_preprocess.py
#
# 저장 (jungmin/outputs/ppt)
#   1_flow.png         전처리 순서
#   2_criteria.png     표시하는 4가지 기준
#   3_missing.png      결측 처리 (3행 이하만 채움)
#   4_outlier.png      이상치 판정 (직전 11개 중앙값 기준)
# 흰 배경, 글자는 진한 회색, 강조색은 파랑 하나(이상치만 주황)로 통일
OutDir = Path(__file__).resolve().parent / "outputs" / "ppt"

Ink, Gray, Light, Line = "#1f2933", "#6b7683", "#f2f4f6", "#d5dae0"
Blue, Orange = "#2f6fdb", "#e8672a"

ppt_font.UseMatplotlib()


def Canvas(Width, Height):
    Fig = plt.figure(figsize=(Width, Height), dpi=200)
    Ax = Fig.add_axes([0, 0, 1, 1])
    Ax.set_xlim(0, Width)
    Ax.set_ylim(0, Height)
    Ax.axis("off")
    return Fig, Ax


def Box(Ax, X, Y, W, H, Fill=Light, Edge="none", Round=0.12, Lw=1.2):
    Ax.add_patch(FancyBboxPatch((X, Y), W, H, boxstyle=f"round,pad=0,rounding_size={Round}", fc=Fill, ec=Edge, lw=Lw))


def Save(Fig, Name):
    OutDir.mkdir(parents=True, exist_ok=True)
    Fig.savefig(OutDir / Name, facecolor="white")
    plt.close(Fig)
    print(f"저장: {OutDir / Name}")


# ---------------------------------------------------------------------------
# 1. 전처리 순서
# ---------------------------------------------------------------------------
def Flow():
    Fig, Ax = Canvas(13.6, 4.2)
    Steps = [("원본 읽기", "시나리오·설비별로\n시각순 정렬"), ("구간 표시", "정지 · 결측 · 정비"), ("결측 채우기", "연속 3행 이하만\n직전 값으로"),
             ("이상치 표시", "직전 11개 값의\n중앙값 기준"), ("저장", "행마다 표시 열 4개\n+ 경과 초")]
    W, H, Gap, X0, Y = 2.2, 2.0, 0.42, 0.4, 1.45

    for Index, (Title, Sub) in enumerate(Steps):
        X = X0 + Index * (W + Gap)
        Box(Ax, X, Y, W, H, Fill=Light)
        Ax.add_patch(plt.Circle((X + 0.42, Y + H - 0.42), 0.2, color=Blue))
        Ax.text(X + 0.42, Y + H - 0.42, str(Index + 1), ha="center", va="center", color="white", fontsize=12, fontweight="bold")
        Ax.text(X + W / 2, Y + 1.0, Title, ha="center", va="center", color=Ink, fontsize=17, fontweight="bold")
        Ax.text(X + W / 2, Y + 0.42, Sub, ha="center", va="center", color=Gray, fontsize=11.5, linespacing=1.5)
        if Index < len(Steps) - 1:
            Ax.add_patch(FancyArrowPatch((X + W + 0.06, Y + H / 2), (X + W + Gap - 0.06, Y + H / 2), arrowstyle="-|>", mutation_scale=14, color=Gray, lw=1.6))

    Box(Ax, X0, 0.3, 5 * W + 4 * Gap, 0.7, Fill="white", Edge=Line)
    Ax.text(X0 + (5 * W + 4 * Gap) / 2, 0.65, "행은 지우지 않고, 이상치는 값을 바꾸지 않고 표시만 붙입니다", ha="center", va="center", color=Ink, fontsize=14)
    Save(Fig, "1_flow.png")


# ---------------------------------------------------------------------------
# 2. 표시하는 4가지 기준
# ---------------------------------------------------------------------------
def Criteria():
    Items = [("정지", "가동 여부가\n꺼져 있음", "RUNNING = 0", "IS_STOPPED"),
             ("결측", "센서 값이 비었거나\n품질 표시가 정상이 아님", "값 없음 또는 품질 불량", "IS_MISSING"),
             ("정비 구간", "정비 이력에 적힌\n시작부터 종료 사이", "시작 이상, 종료 미만", "IS_MAINTENANCE"),
             ("이상치", "직전 11개 값의 중앙값에서\n크게 벗어난 값", "중앙값에서 10 × 척도 이상", "IS_OUTLIER")]
    W, H, Gap, X0, Y = 3.0, 3.5, 0.35, 0.4, 1.2
    Total = 2 * X0 + len(Items) * W + (len(Items) - 1) * Gap
    Fig, Ax = Canvas(Total, 5.0)

    for Index, (Title, Desc, Rule, Col) in enumerate(Items):
        X = X0 + Index * (W + Gap)
        Accent = Orange if Index == 3 else Blue
        Box(Ax, X, Y, W, H, Fill=Light)
        Ax.add_patch(plt.Rectangle((X + 0.3, Y + H - 0.55), 0.5, 0.07, color=Accent, lw=0))
        Ax.text(X + 0.3, Y + H - 1.0, Title, ha="left", va="center", color=Ink, fontsize=20, fontweight="bold")
        Ax.text(X + 0.3, Y + H - 1.85, Desc, ha="left", va="center", color=Gray, fontsize=12, linespacing=1.55)
        Ax.text(X + 0.3, Y + 1.0, Rule, ha="left", va="center", color=Ink, fontsize=12.5)
        Box(Ax, X + 0.3, Y + 0.25, W - 0.6, 0.45, Fill="white", Edge=Line, Round=0.08, Lw=1)
        Ax.text(X + W / 2, Y + 0.475, Col, ha="center", va="center", color=Accent, fontsize=11.5, fontweight="medium")

    Ax.text(Total / 2, 0.55, "정지 · 결측 · 정비 구간의 행은 이상치 판정에서 제외합니다", ha="center", va="center", color=Ink, fontsize=14)
    Save(Fig, "2_criteria.png")


# ---------------------------------------------------------------------------
# 3. 결측 처리
# ---------------------------------------------------------------------------
def Missing():
    Fig, Ax = Canvas(13, 4.6)
    Panels = [("연속 3행 이하", "직전 값으로 채움", 3, True), ("연속 4행 이상", "채우지 않고 결측으로 표시", 4, False)]
    Base = np.array([4.0, 4.3, 4.1, 4.4, 4.2, 4.5, 4.3, 4.6, 4.4, 4.7, 4.5, 4.6])

    for Index, (Title, Sub, Gap, Fill) in enumerate(Panels):
        X0 = 0.5 + Index * 6.4
        Ax.text(X0, 4.05, Title, ha="left", va="center", color=Ink, fontsize=19, fontweight="bold")
        Ax.text(X0, 3.55, Sub, ha="left", va="center", color=Blue if Fill else Orange, fontsize=14)
        Box(Ax, X0, 0.55, 5.9, 2.6, Fill=Light)

        Pos = np.linspace(X0 + 0.45, X0 + 5.45, len(Base))
        Y = 0.9 + (Base - 3.8) / 1.1 * 1.9
        Start = 4
        Missing = range(Start, Start + Gap)
        Prev = Y[Start - 1]

        for Id in range(len(Base)):
            if Id in Missing:
                Ax.add_patch(plt.Circle((Pos[Id], Y[Id]), 0.13, fc="white", ec=Gray, lw=1.4, ls=(0, (2, 2))))
                if Fill:
                    Ax.plot([Pos[Id] - 0.16, Pos[Id] + 0.16], [Prev, Prev], color=Blue, lw=3, solid_capstyle="round")
            else:
                Ax.add_patch(plt.Circle((Pos[Id], Y[Id]), 0.13, color=Ink, lw=0))

        if Fill:
            Ax.text(Pos[Start], 0.72, "직전 값 반복", ha="left", va="center", color=Blue, fontsize=11.5)
        else:
            Ax.text(Pos[Start], 0.72, "빈 칸 유지 + IS_MISSING = 1", ha="left", va="center", color=Orange, fontsize=11.5)

    Ax.text(6.5, 0.22, "짧은 끊김은 이어 주고, 길게 끊긴 구간은 채우지 않습니다", ha="center", va="center", color=Ink, fontsize=14)
    Save(Fig, "3_missing.png")


# ---------------------------------------------------------------------------
# 4. 이상치 판정 (실제 전처리 함수로 밴드 계산)
# ---------------------------------------------------------------------------
def Outlier():
    Rng = np.random.default_rng(7)
    N = 52
    Signal = pd.Series(100 + np.cumsum(Rng.normal(0, 0.25, N)) * 0.6 + Rng.normal(0, 0.9, N))
    Spike = 38
    Signal.iloc[Spike] += 14

    Median, Mad = detect.HampelStats(Signal, config.Window)
    Floor = config.RelFloor * Median.abs()
    Scale = Mad.where(Mad > Floor, Floor).clip(lower=1e-9)
    Upper, Lower = Median + config.DetectNSigma * Scale, Median - config.DetectNSigma * Scale
    Hit = ((Signal - Median).abs() > config.DetectNSigma * Scale) & Median.notna() & Mad.notna()

    Fig = plt.figure(figsize=(13, 4.9), dpi=200)
    Ax = Fig.add_axes([0.05, 0.2, 0.9, 0.62])
    Ax.set_ylim(min(Lower[Median.notna() & Mad.notna()].min(), Signal.min()) - 1.5, Signal.max() + 3)
    Idx = np.arange(N)
    Ok = Median.notna() & Mad.notna()
    Ax.fill_between(Idx[Ok], Lower[Ok], Upper[Ok], color=Blue, alpha=0.12, lw=0, label="정상 범위")
    Ax.plot(Idx[Ok], Median[Ok], color=Blue, lw=1.6, label="직전 11개 중앙값")
    Ax.plot(Idx, Signal, color=Gray, lw=1.2, zorder=2)
    Ax.scatter(Idx[~Hit], Signal[~Hit], s=22, color=Ink, zorder=3)
    Ax.scatter(Idx[Hit], Signal[Hit], s=120, color=Orange, zorder=4)
    Ax.annotate("이상치\n(범위 밖)", xy=(Spike, Signal.iloc[Spike]), xytext=(Spike + 3.2, Signal.iloc[Spike] - 1.2), color=Orange, fontsize=14, va="center", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=Orange, lw=1.2))

    Low = float(Signal.iloc[Spike - 11:Spike].min()) - 1.6
    Left, Right = Spike - 11 - 0.3, Spike - 1 + 0.3
    Ax.plot([Left, Left, Right, Right], [Low + 0.8, Low, Low, Low + 0.8], color=Blue, lw=1.8, clip_on=False)
    Ax.text((Left + Right) / 2, Low - 0.9, "이 11개로 중앙값을 계산", ha="center", va="top", color=Blue, fontsize=12.5)
    for Side in ("top", "right", "left"):
        Ax.spines[Side].set_visible(False)
    Ax.spines["bottom"].set_color(Line)
    Ax.set_yticks([])
    Ax.set_xticks([])
    Ax.set_xlim(-1, N)
    Ax.legend(loc="upper left", frameon=False, fontsize=12, labelcolor=Gray, ncol=2)

    Fig.text(0.05, 0.93, "이상치 판정", fontsize=19, fontweight="bold", color=Ink, ha="left", va="center")
    Fig.text(0.05, 0.12, f"각 값은 바로 앞 {config.Window}개 값의 중앙값에서 {config.DetectNSigma:g} × 척도(MAD) 이상 벗어나면 이상치입니다. 미래 값은 쓰지 않습니다.",
             fontsize=13.5, color=Ink, ha="left", va="center")
    Fig.savefig(OutDir / "4_outlier.png", facecolor="white")
    plt.close(Fig)
    print(f"저장: {OutDir / '4_outlier.png'}")


def Main():
    OutDir.mkdir(parents=True, exist_ok=True)
    Flow()
    Criteria()
    Missing()
    Outlier()


if __name__ == "__main__":
    Main()
