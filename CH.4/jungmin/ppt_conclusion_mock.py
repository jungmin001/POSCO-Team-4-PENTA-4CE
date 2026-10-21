import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

import ppt_effect_mock as Base
from ppt_effect_mock import Blue, Gray, Ink, Light, Text


# "결론 및 향후 발전 방향" 슬라이드 배치 미리보기, CH.4 폴더에서 실행
#   python jungmin/effect_metrics.py                             숫자 먼저 계산
#   python jungmin/ppt_conclusion_mock.py [슬라이드 양식 이미지 경로]
#
# 저장 (jungmin/outputs/ppt)
#   9_conclusion_mock.png   비어 있는 양식 위에 내용을 올린 미리보기 (PPT에서 글상자로 다시 만들 때 참고)
# 왼쪽은 지금까지 만든 것, 오른쪽은 한계를 앞으로의 계획에 이어서 "지금 → 앞으로"로 적는다
OutDir = Path(__file__).resolve().parent / "outputs" / "ppt"
TemplatePath = OutDir / "slide_template_3.png"

Next = [("가상 데이터로 검증", "실제 설비 데이터로 검증", ""),
        ("미리 계산한 예측을 재생", "센서를 실시간으로 연결", ""),
        ("가정한 금액으로 계산", "실제 비용으로 효과 측정", ""),
        ("일부 예측은 참고용", "사례를 늘려 정확도 보강", "6시간 예측 · 일부 원인 진단")]


def Main():
    Metrics = json.loads((OutDir / "effect_metrics.json").read_text(encoding="utf-8"))
    if len(sys.argv) > 1:
        Image.open(sys.argv[1]).convert("RGB").save(TemplatePath)

    Slide = Image.open(TemplatePath).convert("RGBA")
    Draw = ImageDraw.Draw(Slide)
    Left, Right, Gap = 128, 1893, 30
    LeftW = 560
    Top, Bottom = 400, 1012

    Text(Draw, (Left, 352), "미리 알리는 데서 끝나지 않고 결정까지 이어지는 시스템을 만들었고, 다음은 현장에서 확인하는 단계입니다", 30)

    # 왼쪽: 지금까지 만든 것
    Base.Card(Draw, Left, Top, LeftW, Bottom - Top, "결론", "fill", "지금까지 만든 것")
    Items = [("하나로 연결", "이상 알아채기 · 원인 짚기 ·\n고장 미리 알리기"),
             (f"고장의 {Metrics['warned_rate']:.0%}를 미리 알림", f"검증에서 보통 {Metrics['lead_median_h']:.0f}시간 전\n(고장 {Metrics['failures']}건 중 {Metrics['warned']}건)"),
             ("결정까지 이어지는 화면", "비용 비교 · 작업 지시 · 기록")]
    for Index, (Big, Small) in enumerate(Items):
        Y = Top + 135 + Index * 150
        Draw.ellipse((Left + 36, Y + 6, Left + 62, Y + 32), fill=Blue)
        Text(Draw, (Left + 49, Y + 19), "✓", 18, (255, 255, 255, 255), Bold=True, Anchor="mm")
        Text(Draw, (Left + 84, Y), Big, 32, Ink, Bold=True)
        Text(Draw, (Left + 84, Y + 52), Small, 24, Gray)

    # 오른쪽: 지금 → 앞으로
    X0 = Left + LeftW + Gap
    W = Right - X0
    Base.Card(Draw, X0, Top, W, Bottom - Top, "다음", "gray", "앞으로 할 일")
    BoxL, BoxR = X0 + 36, X0 + W - 36
    LeftBoxW, ArrowW = 470, 96
    RightBoxX = BoxL + LeftBoxW + ArrowW
    Text(Draw, (BoxL + 14, Top + 140), "지금", 24, Gray)
    Text(Draw, (RightBoxX + 14, Top + 140), "앞으로", 24, Blue, Bold=True)
    for Index, (Now, After, Sub) in enumerate(Next):
        Y = Top + 182 + Index * 100
        Draw.rounded_rectangle((BoxL, Y, BoxL + LeftBoxW, Y + 80), radius=12, fill=(242, 244, 246, 255))
        Text(Draw, (BoxL + 24, Y + 40), Now, 27, Gray, Anchor="lm")
        Draw.polygon([(BoxL + LeftBoxW + 26, Y + 28), (BoxL + LeftBoxW + 62, Y + 40), (BoxL + LeftBoxW + 26, Y + 52)], fill=Blue)
        Draw.rounded_rectangle((RightBoxX, Y, BoxR, Y + 80), radius=12, fill=(255, 255, 255, 255), outline=Blue, width=3)
        Text(Draw, (RightBoxX + 24, Y + (28 if Sub else 40)), After, 28, Ink, Bold=True, Anchor="lm")
        if Sub:
            Text(Draw, (RightBoxX + 24, Y + 58), Sub, 20, Gray, Anchor="lm")

    Text(Draw, (Left, 1042), "검증 시나리오 기준  ·  현장에 적용하기 전에는 실제 데이터로 다시 확인해야 합니다", 21, Gray)

    OutDir.mkdir(parents=True, exist_ok=True)
    Slide.convert("RGB").save(OutDir / "9_conclusion_mock.png")
    print(f"저장: {OutDir / '9_conclusion_mock.png'}")


if __name__ == "__main__":
    Main()
