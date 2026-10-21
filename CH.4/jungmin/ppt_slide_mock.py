import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw

import ppt_font


# "서비스 시스템 구현" 슬라이드 배치 미리보기, CH.4 폴더에서 실행
#   python jungmin/ppt_slide_mock.py [슬라이드 양식 이미지 경로]
#
# 저장 (jungmin/outputs/ppt)
#   6_system_wide.png   가로로 납작한 구조도 (배경 투명, 슬라이드에 그대로 붙여 넣는 용도)
#   7_slide_mock.png    비어 있는 양식 위에 내용을 올린 미리보기 (화면 캡처 자리는 점선 칸)
OutDir = Path(__file__).resolve().parent / "outputs" / "ppt"
TemplatePath = OutDir / "slide_template.png"

Blue, Ink, Sub, Line = "#2f6fdb", "#1f2933", "#6b7683", "#d5dae0"
ppt_font.UseMatplotlib()

Steps = [("데이터", ["설비 7대의 센서", "정비 기록"]),
         ("분석 · 예측", ["이상 알아채기", "원인 짚기", "고장 미리 알리기"]),
         ("디지털 트윈 화면", ["3D 설비 지도", "위험을 색으로 표시", "선택지 비용 비교"]),
         ("작업 지시 · 기록", ["담당자 · 마감 지정", "결정 기록"])]



def WideDiagram():
    Total, Height, W, H, Gap, X0, Y = 13.6, 2.8, 3.1, 1.85, 0.4, 0.0, 0.8
    Fig = plt.figure(figsize=(Total, Height), dpi=200)
    Ax = Fig.add_axes([0, 0, 1, 1])
    Ax.set_xlim(0, Total)
    Ax.set_ylim(0, Height)
    Ax.axis("off")

    for Index, (Title, Lines) in enumerate(Steps):
        X = X0 + Index * (W + Gap)
        Ax.add_patch(FancyBboxPatch((X, Y), W, H, boxstyle="round,pad=0,rounding_size=0.14", fc=(0.95, 0.96, 0.97, 0.95), ec=Line, lw=1.2))
        Ax.add_patch(plt.Circle((X + 0.36, Y + H - 0.36), 0.17, color=Blue))
        Ax.text(X + 0.36, Y + H - 0.36, str(Index + 1), ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        Ax.text(X + W / 2 + 0.1, Y + H - 0.5, Title, ha="center", va="center", color=Ink, fontsize=16, fontweight="bold")
        for Row, Text in enumerate(Lines):
            Ax.text(X + W / 2, Y + H - 0.98 - Row * 0.33, Text, ha="center", va="center", color=Sub, fontsize=12)
        if Index < len(Steps) - 1:
            Ax.add_patch(FancyArrowPatch((X + W + 0.07, Y + H / 2), (X + W + Gap - 0.07, Y + H / 2), arrowstyle="-|>", mutation_scale=14, color=Sub, lw=1.6))

    X2, X4, Low = X0 + 1 * (W + Gap) + W / 2, X0 + 3 * (W + Gap) + W / 2, 0.42
    Ax.plot([X4, X4, X2], [Y - 0.04, Low, Low], color=Blue, lw=1.7, solid_capstyle="round")
    Ax.add_patch(FancyArrowPatch((X2, Low), (X2, Y - 0.04), arrowstyle="-|>", mutation_scale=14, color=Blue, lw=1.7))
    Ax.text((X2 + X4) / 2, Low - 0.2, "기록을 실제 결과와 비교해 다음 예측에 반영", ha="center", va="center", color=Blue, fontsize=12.5)

    OutDir.mkdir(parents=True, exist_ok=True)
    Fig.savefig(OutDir / "6_system_wide.png", transparent=True)
    plt.close(Fig)


def Dashed(Draw, Box, Color, Dash=14, Gap=9, Width=3):
    X0, Y0, X1, Y1 = Box
    for X in range(X0, X1, Dash + Gap):
        Draw.line([(X, Y0), (min(X + Dash, X1), Y0)], fill=Color, width=Width)
        Draw.line([(X, Y1), (min(X + Dash, X1), Y1)], fill=Color, width=Width)
    for Y in range(Y0, Y1, Dash + Gap):
        Draw.line([(X0, Y), (X0, min(Y + Dash, Y1))], fill=Color, width=Width)
        Draw.line([(X1, Y), (X1, min(Y + Dash, Y1))], fill=Color, width=Width)


def Mock():
    Slide = Image.open(TemplatePath).convert("RGBA")
    Draw = ImageDraw.Draw(Slide)
    Left, Right = 150, 1852

    # 한 줄 메시지
    Draw.text((Left, 348), "데이터에서 예측, 추천, 작업 지시까지 이어지는 순환 시스템", font=ppt_font.Pil(30), fill=(31, 41, 51, 255))

    # 구조도
    Diagram = Image.open(OutDir / "6_system_wide.png").convert("RGBA")
    Width = Right - Left
    Diagram = Diagram.resize((Width, round(Diagram.height * Width / Diagram.width)), Image.LANCZOS)
    Top = 392
    Slide.alpha_composite(Diagram, (Left, Top))

    # 화면 캡처 자리 두 칸
    Y0, Y1 = Top + Diagram.height + 20, 1036
    Half = (Width - 30) // 2
    Labels = [("언리얼 화면 캡처 1", "종합 현황 (3D 설비 지도 · 위험 색 · 진단)"), ("언리얼 화면 캡처 2", "선택지 비용 비교 · 작업 지시")]
    for Index, (Title, Sub2) in enumerate(Labels):
        X0 = Left + Index * (Half + 30)
        Draw.rounded_rectangle((X0, Y0, X0 + Half, Y1), radius=14, fill=(255, 255, 255, 235))
        Dashed(Draw, (X0, Y0, X0 + Half, Y1), (160, 170, 182, 255))
        Draw.text((X0 + Half // 2, (Y0 + Y1) // 2 - 18), Title, font=ppt_font.Pil(28), fill=(31, 41, 51, 255), anchor="mm")
        Draw.text((X0 + Half // 2, (Y0 + Y1) // 2 + 22), Sub2, font=ppt_font.Pil(22), fill=(107, 118, 131, 255), anchor="mm")

    # 각주
    Draw.text((Left, 1054), "시연 방식: 미리 계산한 예측을 시간순으로 재생  ·  작업 지시는 시연용 입력  ·  금액은 가정값", font=ppt_font.Pil(20), fill=(107, 118, 131, 255))

    Slide.convert("RGB").save(OutDir / "7_slide_mock.png")
    print(f"저장: {OutDir / '7_slide_mock.png'}")


def Main():
    if len(sys.argv) > 1:
        OutDir.mkdir(parents=True, exist_ok=True)
        Image.open(sys.argv[1]).convert("RGB").save(TemplatePath)

    WideDiagram()
    Mock()


if __name__ == "__main__":
    Main()
