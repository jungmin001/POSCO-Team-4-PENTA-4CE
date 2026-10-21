import json
import sys
from pathlib import Path

import ppt_font
from PIL import Image, ImageDraw


# "경제적 효과 및 기대 성과" 슬라이드 배치 미리보기, CH.4 폴더에서 실행
#   python jungmin/effect_metrics.py                         숫자 먼저 계산
#   python jungmin/ppt_effect_mock.py [슬라이드 양식 이미지 경로]
#
# 저장 (jungmin/outputs/ppt)
#   8_effect_mock.png   비어 있는 양식 위에 내용을 올린 미리보기 (PPT에서 글상자로 다시 만들 때 참고)
# 칸마다 근거의 종류를 표시한다: 측정(검증 결과), 가정(가정 금액으로 환산), 기대(아직 측정하지 못한 예상)
OutDir = Path(__file__).resolve().parent / "outputs" / "ppt"
TemplatePath = OutDir / "slide_template_2.png"

Ink, Gray, Light, Blue = (31, 41, 51, 255), (107, 118, 131, 255), (213, 218, 224, 255), (47, 111, 219, 255)


def F(Size, Bold=False):
    return ppt_font.Pil(Size, "bold" if Bold else "regular")


def Text(Draw, Xy, Value, Size, Fill=Ink, Bold=False, Anchor="la"):
    Draw.text(Xy, Value, font=F(Size, Bold), fill=Fill, anchor=Anchor)


def Tag(Draw, X, Y, Label, Style):
    Width = 92
    if Style == "fill":
        Draw.rounded_rectangle((X, Y, X + Width, Y + 44), radius=22, fill=Blue)
        Text(Draw, (X + Width / 2, Y + 22), Label, 24, (255, 255, 255, 255), Anchor="mm")
    else:
        Color = Blue if Style == "blue" else Gray
        Draw.rounded_rectangle((X, Y, X + Width, Y + 44), radius=22, outline=Color, width=3, fill=(255, 255, 255, 255))
        Text(Draw, (X + Width / 2, Y + 22), Label, 24, Color, Anchor="mm")


def Card(Draw, X, Y, W, H, Label, Style, Title):
    Draw.rounded_rectangle((X, Y, X + W, Y + H), radius=16, fill=(255, 255, 255, 240), outline=Light, width=2)
    Tag(Draw, X + 36, Y + 32, Label, Style)
    Text(Draw, (X + 148, Y + 54), Title, 32, Ink, Bold=True, Anchor="lm")


def Main():
    Metrics = json.loads((OutDir / "effect_metrics.json").read_text(encoding="utf-8"))
    if len(sys.argv) > 1:
        Image.open(sys.argv[1]).convert("RGB").save(TemplatePath)

    Slide = Image.open(TemplatePath).convert("RGBA")
    Draw = ImageDraw.Draw(Slide)
    Left, Right = 128, 1893
    Gap = 30
    W = (Right - Left - 2 * Gap) // 3
    Top, Bottom = 400, 1012

    Text(Draw, (Left, 352), f"고장의 {Metrics['warned_rate']:.0%}를 미리 알렸고, 고장 1건을 막으면 약 {Metrics['saving_per_failure']:,.0f}만원(가정)을 아낍니다", 30)

    # 1. 측정
    X = Left
    Card(Draw, X, Top, W, Bottom - Top, "측정", "fill", "검증으로 확인한 것")
    Rows = [(f"{Metrics['warned_rate']:.0%}", f"실제 고장 {Metrics['failures']}건 중 {Metrics['warned']}건,\n고장 전에 경보가 울림"),
            (f"약 {Metrics['lead_median_h']:.0f}시간", "첫 경보부터 고장까지\n(중앙값)"),
            (f"{Metrics['quiet_alarmed_rate']:.0%}", "고장이 없는 설비에서\n경보가 울린 비율")]
    for Index, (Big, Small) in enumerate(Rows):
        Y = Top + 125 + Index * 160
        Text(Draw, (X + 36, Y), Big, 64, Blue if Index == 0 else Ink, Bold=True)
        Text(Draw, (X + 36, Y + 82), Small, 24, Gray)

    # 2. 가정
    X = Left + W + Gap
    Card(Draw, X, Top, W, Bottom - Top, "가정", "blue", "비용으로 환산하면")
    Inner = W - 72
    Bars = [("고장 난 뒤에 대응", Metrics["cost_after_failure"], Light), ("미리 교체", Metrics["cost_replace_now"], Blue)]
    for Index, (Label, Cost, Color) in enumerate(Bars):
        Y = Top + 140 + Index * 120
        Text(Draw, (X + 36, Y), Label, 26, Ink)
        Text(Draw, (X + W - 36, Y), f"약 {Cost:,.0f}만원", 26, Blue if Index == 1 else Gray, Anchor="ra")
        Draw.rounded_rectangle((X + 36, Y + 44, X + 36 + Inner * Cost / Bars[0][1], Y + 80), radius=10, fill=Color)
    Y = Top + 140 + 2 * 120 + 20
    Text(Draw, (X + 36, Y), "1건당 절감", 26, Gray)
    Text(Draw, (X + 36, Y + 44), f"약 {Metrics['saving_per_failure']:,.0f}만원", 58, Blue, Bold=True)
    Text(Draw, (X + 36, Y + 126), "고장 1건을 미리 막을 때, 가정 금액 기준", 22, Gray)

    # 3. 기대
    X = Left + 2 * (W + Gap)
    Card(Draw, X, Top, W, Bottom - Top, "기대", "gray", "기대 성과")
    Items = ["갑작스러운 정지 줄이기", "고장 전에 계획 정비", "원인 후보로 점검 범위 좁히기", "설비를 늘려도 같은 방식"]
    for Index, Item in enumerate(Items):
        Y = Top + 150 + Index * 108
        Draw.ellipse((X + 36, Y + 10, X + 56, Y + 30), fill=Blue)
        Text(Draw, (X + 76, Y), Item, 30, Ink)
        Draw.line([(X + 36, Y + 62), (X + W - 36, Y + 62)], fill=Light, width=2) if Index < len(Items) - 1 else None

    Text(Draw, (Left, 1042), f"검증 시나리오 {Metrics['scenarios']}개(설비 {Metrics['equipments']}대) 기준  ·  금액은 가정값  ·  실제 효과는 현장 데이터로 다시 확인해야 합니다", 21, Gray)

    OutDir.mkdir(parents=True, exist_ok=True)
    Slide.convert("RGB").save(OutDir / "8_effect_mock.png")
    print(f"저장: {OutDir / '8_effect_mock.png'}")


if __name__ == "__main__":
    Main()
