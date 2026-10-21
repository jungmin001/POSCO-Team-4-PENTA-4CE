from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import ppt_font


# 서비스 시스템 구조도 (PPT용, 배경 투명), CH.4 폴더에서 실행
#   python jungmin/ppt_system.py
#
# 저장 (jungmin/outputs/ppt)
#   6_system.png        글자가 진한 회색 (밝은 슬라이드용)
#   6_system_light.png  글자가 흰색 (어두운 슬라이드용)
OutDir = Path(__file__).resolve().parent / "outputs" / "ppt"

Blue = "#2f6fdb"
ppt_font.UseMatplotlib()

Steps = [("데이터", ["설비 7대의 센서", "정비 기록"]),
         ("분석 · 예측", ["이상 알아채기", "원인 짚기", "고장 미리 알리기"]),
         ("디지털 트윈 화면", ["3D 설비 지도", "위험을 색으로 표시", "선택지 비용 비교"]),
         ("작업 지시 · 기록", ["담당자 · 마감 지정", "결정 기록"])]


def Draw(Name, Ink, Sub, Fill, Line):
    W, H, Gap, X0, Y, Total = 2.75, 2.7, 0.5, 0.35, 1.55, 13.6
    Fig = plt.figure(figsize=(Total, 5.3), dpi=200)
    Ax = Fig.add_axes([0, 0, 1, 1])
    Ax.set_xlim(0, Total)
    Ax.set_ylim(0, 5.3)
    Ax.axis("off")


    for Index, (Title, Lines) in enumerate(Steps):
        X = X0 + Index * (W + Gap)
        Ax.add_patch(FancyBboxPatch((X, Y), W, H, boxstyle="round,pad=0,rounding_size=0.14", fc=Fill, ec=Line, lw=1.2))
        Ax.add_patch(plt.Circle((X + 0.42, Y + H - 0.42), 0.2, color=Blue))
        Ax.text(X + 0.42, Y + H - 0.42, str(Index + 1), ha="center", va="center", color="white", fontsize=12, fontweight="bold")
        Ax.text(X + W / 2, Y + H - 1.1, Title, ha="center", va="center", color=Ink, fontsize=17, fontweight="bold")
        for Row, Text in enumerate(Lines):
            Ax.text(X + W / 2, Y + H - 1.7 - Row * 0.37, Text, ha="center", va="center", color=Sub, fontsize=12.5)
        if Index < len(Steps) - 1:
            Ax.add_patch(FancyArrowPatch((X + W + 0.08, Y + H / 2), (X + W + Gap - 0.08, Y + H / 2), arrowstyle="-|>", mutation_scale=15, color=Sub, lw=1.7))

    # 되돌아가는 화살표: 기록을 실제 결과와 비교해 분석에 반영 (④ → ②)
    X2, X4 = X0 + 1 * (W + Gap) + W / 2, X0 + 3 * (W + Gap) + W / 2
    Low = 0.95
    Ax.plot([X4, X4, X2], [Y - 0.05, Low, Low], color=Blue, lw=1.8, solid_capstyle="round")
    Ax.add_patch(FancyArrowPatch((X2, Low), (X2, Y - 0.05), arrowstyle="-|>", mutation_scale=15, color=Blue, lw=1.8))
    Ax.text((X2 + X4) / 2, Low - 0.32, "기록을 실제 결과와 비교해 다음 예측에 반영", ha="center", va="center", color=Blue, fontsize=13)

    Ax.text(0.35, 0.2, "시연 방식: 미리 계산한 예측을 시간순으로 재생합니다  ·  작업 지시는 시연용 입력입니다  ·  금액은 가정값입니다", ha="left", va="center", color=Sub, fontsize=11.5)

    OutDir.mkdir(parents=True, exist_ok=True)
    Fig.savefig(OutDir / Name, transparent=True)
    plt.close(Fig)
    print(f"저장: {OutDir / Name}")


def Main():
    Draw("6_system.png", "#1f2933", "#6b7683", (0.95, 0.96, 0.97, 0.9), "#d5dae0")
    Draw("6_system_light.png", "#ffffff", "#c3cbd3", (1, 1, 1, 0.07), (1, 1, 1, 0.25))


if __name__ == "__main__":
    Main()
